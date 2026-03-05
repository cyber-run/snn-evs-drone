"""
Train the LGMD SNN on synthetic looming event sequences.

Loss design:
  - PRIMARY: window-level Pearson correlation between mean PRE-LIF EXCITATION
    per window and mean label per window.  The excitation signal is continuous
    and dense, giving strong gradients to exc_conv even when very few neurons
    spike (the looming signal fires only ~18/5590 edge pixels at init).
  - AUXILIARY: same correlation on spike-based DCMD output.  At init DCMD is
    near-zero so this adds almost nothing, but as training progresses and more
    neurons start firing it provides the proper SNN-level signal.
  - SUPPRESSION: small L1 penalty on mean excitation (keeps background quiet).

Data pipeline:
  - Events stored in RAM, windows encoded on-the-fly (fine dt_us without OOM)
  - On-the-fly augmentation: spatial flips, polarity swap, noise, dropout
  - Weighted sampling: oversamples looming windows to balance the dataset
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import h5py
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from snn.models.lgmd_net import LGMDNet
from snn.models.event_encoder import EventEncoder, angular_velocity_label


# ── Augmentation ─────────────────────────────────────────────────────────────

class EventAugmentor:
    """
    On-the-fly augmentation for encoded event frames (T, 2, H, W).

    Augmentations (all stochastic, applied independently per sample):
      - Horizontal flip: spatial invariance for looming from any direction
      - Vertical flip: same reasoning
      - Polarity swap: ON<->OFF channels (dark-on-bright vs bright-on-dark)
      - Additive noise: random binary events (simulates hot pixels / background)
      - Event dropout: randomly zero a fraction of events (sim-to-real gap)
    """

    def __init__(self, hflip=0.5, vflip=0.3, polarity_swap=0.2,
                 noise_rate=0.005, dropout_rate=0.05):
        self.hflip = hflip
        self.vflip = vflip
        self.polarity_swap = polarity_swap
        self.noise_rate = noise_rate
        self.dropout_rate = dropout_rate

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Augment a (T, 2, H, W) binary event frame tensor."""
        if torch.rand(1).item() < self.hflip:
            frames = frames.flip(-1)
        if torch.rand(1).item() < self.vflip:
            frames = frames.flip(-2)
        if torch.rand(1).item() < self.polarity_swap:
            frames = frames[:, [1, 0]]
        if self.noise_rate > 0:
            noise = (torch.rand_like(frames) < self.noise_rate).float()
            frames = (frames + noise).clamp_(0, 1)
        if self.dropout_rate > 0:
            mask = (torch.rand_like(frames) > self.dropout_rate).float()
            frames = frames * mask
        return frames


# ── Loss helpers ─────────────────────────────────────────────────────────────

def _pearson_window(x: torch.Tensor, lbl: torch.Tensor) -> torch.Tensor:
    """
    Batch-level Pearson correlation: compare window-mean x to window-mean label.
    x, lbl: (T, B).  Returns scalar in [-1, 1]; higher is better.
    """
    xw  = x.mean(dim=0)    # (B,)
    lw  = lbl.mean(dim=0)
    d   = xw  - xw.mean()
    l   = lw  - lw.mean()
    if l.norm() < 1e-6:
        return torch.tensor(0.0, device=x.device)
    return (d * l).sum() / (d.norm() * l.norm() + 1e-8)


def combined_loss(dcmd: torch.Tensor, exc_mean: torch.Tensor,
                  target: torch.Tensor,
                  exc_weight: float = 0.8,
                  bg_penalty: float = 0.05) -> torch.Tensor:
    """
    Weighted sum of excitation-level + spike-level window Pearson losses.
    """
    lbl_std = target.mean(dim=0).std()

    if lbl_std > 0.005:
        exc_corr  = _pearson_window(exc_mean, target)
        dcmd_corr = _pearson_window(dcmd,     target)
        corr_loss = (exc_weight * (1.0 - exc_corr)
                     + (1.0 - exc_weight) * (1.0 - dcmd_corr))
    else:
        corr_loss = torch.tensor(0.0, device=dcmd.device)

    supp = bg_penalty * exc_mean.mean()   # suppress false excitation
    return corr_loss + supp


def pearson_val(x: torch.Tensor, lbl: torch.Tensor) -> float:
    """Window-level Pearson r (float). Used for reporting."""
    return _pearson_window(x, lbl).item()


# ── Dataset ──────────────────────────────────────────────────────────────────

class LoomingDataset(Dataset):
    """
    Lazy-encoding dataset: raw events in RAM, encoded per window on-the-fly.

    Compared to pre-encoding all windows to RAM, this:
      - Uses ~100x less memory (events only, not dense frames)
      - Supports fine dt_us / large n_bins without OOM
      - Enables on-the-fly augmentation
      - Adds ~10ms latency per sample (bincount on ~10K events)
    """

    def __init__(self, events, label, encoder, n_bins=50, stride_bins=10,
                 augmentor=None):
        order  = np.argsort(events[:, 0], kind="stable")
        events = events[order]
        self.ts = events[:, 0].astype(np.float64)
        self.xs = events[:, 1].astype(np.int32)
        self.ys = events[:, 2].astype(np.int32)
        self.ps = events[:, 3].astype(np.int32)
        self.encoder = encoder
        self.augmentor = augmentor
        self.n_bins = n_bins

        t_start      = float(self.ts[0])
        t_end        = float(self.ts[-1])
        n_total_bins = int((t_end - t_start) / encoder.dt_us)

        self.windows = []   # window start times (µs)
        labels = []
        for i in range(0, n_total_bins - n_bins, stride_bins):
            ws = t_start + i * encoder.dt_us
            lb = label[i:i + n_bins]
            if len(lb) == n_bins:
                self.windows.append(ws)
                labels.append(torch.from_numpy(lb.copy()))

        self.labels = torch.stack(labels) if labels else torch.empty(0)
        self._mean_labels = (self.labels.mean(dim=1)
                             if len(self.labels) > 0 else torch.empty(0))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        ws = self.windows[i]
        we = ws + self.n_bins * self.encoder.dt_us

        lo = int(np.searchsorted(self.ts, ws, side="left"))
        hi = int(np.searchsorted(self.ts, we, side="left"))

        frames = self.encoder._encode_columns(
            self.ts[lo:hi], self.xs[lo:hi],
            self.ys[lo:hi], self.ps[lo:hi], ws, self.n_bins)

        if self.augmentor is not None:
            frames = self.augmentor(frames)

        return frames, self.labels[i]

    def sample_weights(self, loom_weight=5.0):
        """Per-window weights for WeightedRandomSampler (higher for looming)."""
        return torch.where(self._mean_labels > 0.05, loom_weight, 1.0).tolist()


# ── Label helpers ─────────────────────────────────────────────────────────────

def make_label_from_trajectory(events, h5_path, encoder):
    with h5py.File(h5_path, "r") as f:
        traj = f["obstacle_positions"][:]
        dp   = f["drone_hover_position"][:]
        r    = float(f["obstacle_radius"][()])
        dt   = float(f["sim_dt"][()])
        launch_step = int(f["launch_step"][()]) if "launch_step" in f else 0

    # TODO: remove once all data regenerated with subsampled trajectories.
    # Auto-detect trajectory double-sampling from physics substep rate:
    # Pegasus calls backend.update() at physics rate (typically 2x render rate).
    # If trajectory was NOT subsampled at save time, dt_stored = 1/FPS but each
    # point is actually 1/(2*FPS) apart, making trajectory appear 2x too long.
    t0_us = float(events[0, 0])
    t1_us = float(events[-1, 0])
    event_dur_s  = (t1_us - t0_us) * 1e-6
    traj_dur_s   = len(traj) * dt
    if traj_dur_s > event_dur_s * 1.3:
        factor = round(traj_dur_s / event_dur_s)
        dt = dt / factor

    dth = angular_velocity_label(traj, dp, r, dt)   # shape (T_sim,)

    # Physical time for each trajectory sample (seconds from sim start)
    traj_t_s  = np.arange(len(dth), dtype=np.float64) * dt

    # Physical time axis for the event bins (seconds from recording start)
    n_bins = int((t1_us - t0_us) / encoder.dt_us)
    event_t_s = np.linspace(t0_us, t1_us, n_bins) * 1e-6

    # Interpolate dtheta/dt onto the event time axis
    lab = np.interp(event_t_s, traj_t_s, dth).astype(np.float32)
    return lab / (lab.max() + 1e-6)


def make_label_from_event_rate(events, encoder):
    from scipy.ndimage import uniform_filter1d
    n  = int((float(events[-1, 0]) - float(events[0, 0])) / encoder.dt_us)
    bi = np.clip(((events[:, 0] - events[0, 0]) / encoder.dt_us).astype(np.int32), 0, n-1)
    c  = np.bincount(bi, minlength=n).astype(np.float32)
    c  = uniform_filter1d(c, size=5)
    return c / (c.max() + 1e-6)


def _load_single(path, encoder, n_bins, stride_bins, augmentor=None, tag=""):
    pfx = f"[{tag}] " if tag else "  "
    print(f"{pfx}{path}", flush=True)
    with h5py.File(path, "r") as f:
        ev  = f["events"][:]
        has = "obstacle_positions" in f
    lab  = (make_label_from_trajectory(ev, path, encoder) if has
            else make_label_from_event_rate(ev, encoder))
    frac = float((lab > 0.05).mean()) * 100
    src  = "analytical" if has else "event-rate"
    print(f"{pfx}  {len(ev):,} events  |  looming: {frac:.1f}%  ({src})", flush=True)
    t0 = time.time()
    ds = LoomingDataset(ev, lab, encoder, n_bins, stride_bins, augmentor)
    print(f"{pfx}  {len(ds)} windows in {time.time()-t0:.1f}s", flush=True)
    return ds


# ── Eval ──────────────────────────────────────────────────────────────────────

def _run_eval(model, loader, device):
    model.eval()
    adc, aex, alb = [], [], []
    with torch.no_grad():
        for fr, lb in loader:
            fr = fr.permute(1, 0, 2, 3, 4).to(device, non_blocking=True)
            lb = lb.permute(1, 0).to(device, non_blocking=True)
            dc, _, ex = model(fr)
            adc.append(dc.cpu()); aex.append(ex.cpu()); alb.append(lb.cpu())
    dc  = torch.cat(adc, 1); ex = torch.cat(aex, 1); lb = torch.cat(alb, 1)

    v_loss = combined_loss(dc, ex, lb).item()
    v_exc  = pearson_val(ex, lb)
    v_dcmd = pearson_val(dc, lb)

    # Discrimination: looming vs background window-mean excitation
    lw = lb.mean(0); ew = ex.mean(0)
    mask = lw > 0.05
    ex_l = ew[mask].mean().item()  if mask.any()  else 0.0
    ex_b = ew[~mask].mean().item() if (~mask).any() else 0.0
    return v_loss, v_exc, v_dcmd, ex_l, ex_b


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    ds = args.pool   # spatial downsample in encoder (replaces model's AvgPool2d)
    enc = EventEncoder(height=args.height, width=args.width,
                       dt_us=args.dt_us, mode="binary",
                       spatial_downsample=ds)

    augmentor = EventAugmentor() if args.augment else None

    h5s = args.h5 if isinstance(args.h5, list) else [args.h5]
    stride = args.stride_bins
    print(f"\nTrain ({len(h5s)}):", flush=True)
    train_datasets = [_load_single(p, enc, args.n_bins, stride, augmentor, "tr")
                      for p in h5s]
    tr = ConcatDataset(train_datasets)
    print(f"  {len(tr)} windows  batch={args.batch}  "
          f"~{len(tr)//args.batch} batches/ep", flush=True)

    va = None
    if args.val_h5:
        vs = args.val_h5 if isinstance(args.val_h5, list) else [args.val_h5]
        print(f"\nVal ({len(vs)}):", flush=True)
        val_datasets = [_load_single(p, enc, args.n_bins, stride, None, "va")
                        for p in vs]
        va = ConcatDataset(val_datasets)
        print(f"  {len(va)} windows held out", flush=True)

    # Weighted sampling: oversample looming windows to balance the dataset
    weights = []
    for tds in train_datasets:
        weights.extend(tds.sample_weights(loom_weight=args.loom_weight))
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    nw = min(args.num_workers, 4) if device.type == "cuda" else 0
    tl = DataLoader(tr, batch_size=args.batch, sampler=sampler,
                    num_workers=nw, pin_memory=(device.type == "cuda"),
                    persistent_workers=(nw > 0))
    vl = (DataLoader(va, batch_size=args.batch, shuffle=False,
                     num_workers=nw, pin_memory=(device.type == "cuda"),
                     persistent_workers=(nw > 0)) if va else None)

    model = LGMDNet(height=args.height // ds, width=args.width // ds,
                    pool_factor=1, tau_mem=args.tau).to(device)
    np_ = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {np_} trainable params  LIF_thresh={model.lgmd_lif.v_threshold}"
          f"  tau={args.tau}  exc_weight={args.exc_weight}", flush=True)
    if augmentor:
        print(f"Augmentation: hflip={augmentor.hflip} vflip={augmentor.vflip} "
              f"pol_swap={augmentor.polarity_swap} noise={augmentor.noise_rate} "
              f"dropout={augmentor.dropout_rate}", flush=True)

    opt  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    hdr  = (f"{'Ep':>5}  {'TrLoss':>8}  {'VaLoss':>8}  {'ExCorr':>8}  {'DcCorr':>8}  "
            f"{'Ex_loom':>8}  {'Ex_bg':>6}  {'LR':>8}  {'s':>4}")
    print(f"\nTraining {args.epochs} epochs\n  {hdr}", flush=True)
    print("  " + "-" * len(hdr), flush=True)

    best_corr = -9.0; best_ep = 0; t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train(); tr_loss = 0.0
        for fr, lb in tl:
            fr = fr.permute(1, 0, 2, 3, 4).to(device, non_blocking=True)
            lb = lb.permute(1, 0).to(device, non_blocking=True)
            opt.zero_grad()
            dc, _, ex = model(fr)
            loss = combined_loss(dc, ex, lb, exc_weight=args.exc_weight,
                                 bg_penalty=args.bg_penalty)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        sch.step()
        tr_loss /= max(len(tl), 1)

        if ep % args.log_every == 0 or ep == 1:
            lr_ = sch.get_last_lr()[0]; el = time.time() - t0
            if vl:
                vl_, vex, vdc, exl, exb = _run_eval(model, vl, device)
                m = " *" if vex > best_corr else ""
                if vex > best_corr:
                    best_corr = vex; best_ep = ep
                    if args.save:
                        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), args.save)
                print(f"  {ep:5d}  {tr_loss:8.5f}  {vl_:8.5f}  {vex:8.4f}  "
                      f"{vdc:8.4f}  {exl:8.6f}  {exb:6.4f}  {lr_:8.2e}  "
                      f"{el:3.0f}s{m}", flush=True)
            else:
                print(f"  {ep:5d}  {tr_loss:8.5f}  {'—':>8}  {'—':>8}  "
                      f"{'—':>8}  {'—':>8}  {'—':>6}  {lr_:8.2e}  {el:3.0f}s",
                      flush=True)

    print(f"\n  Done {time.time()-t0:.0f}s  best ExCorr={best_corr:.4f} @ ep {best_ep}",
          flush=True)
    if args.save and not vl:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5",     required=True, nargs="+")
    p.add_argument("--val_h5", nargs="+", default=None)

    # Encoding
    p.add_argument("--height", type=int,   default=260)
    p.add_argument("--width",  type=int,   default=346)
    p.add_argument("--dt_us",  type=float, default=2000.0,
                   help="Time bin width in us (default 2ms)")
    p.add_argument("--n_bins", type=int,   default=50,
                   help="Time bins per window (default 50 = 100ms at 2ms bins)")
    p.add_argument("--stride_bins", type=int, default=10,
                   help="Stride between windows in bins (default 10 = 80%% overlap)")

    # Model
    p.add_argument("--pool",   type=int,   default=4)
    p.add_argument("--tau",    type=float, default=5.0,
                   help="LIF membrane time constant (default 5.0 = 10ms at 2ms bins)")

    # Training
    p.add_argument("--epochs", type=int,   default=200)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--batch",  type=int,   default=32)
    p.add_argument("--exc_weight",  type=float, default=0.8)
    p.add_argument("--bg_penalty",  type=float, default=0.05)
    p.add_argument("--loom_weight", type=float, default=5.0,
                   help="Oversampling weight for looming windows (default 5x)")
    p.add_argument("--log_every",   type=int,   default=10)

    # Augmentation
    p.add_argument("--augment", action="store_true",
                   help="Enable on-the-fly data augmentation")

    # System
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save",   default="results/lgmd_weights.pt")

    train(p.parse_args())
