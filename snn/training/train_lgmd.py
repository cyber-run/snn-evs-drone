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

The split between primary and auxiliary is controlled by --exc_weight (default 0.8).
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import h5py
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from snn.models.lgmd_net import LGMDNet
from snn.models.event_encoder import EventEncoder, angular_velocity_label


# ── Loss helpers ──────────────────────────────────────────────────────────────

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

    For batches where all labels ≈ 0 the Pearson terms collapse and only the
    suppression term applies, keeping background windows quiet.
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


# ── Dataset ───────────────────────────────────────────────────────────────────

class LoomingDataset(Dataset):
    """All windows pre-encoded to RAM."""

    def __init__(self, events, label, encoder, n_bins=20, stride_bins=10):
        order  = np.argsort(events[:, 0], kind="stable")
        events = events[order]
        ts_f64 = events[:, 0].astype(np.float64)
        xs_i32 = events[:, 1].astype(np.int32)
        ys_i32 = events[:, 2].astype(np.int32)
        ps_i32 = events[:, 3].astype(np.int32)
        t_start      = float(ts_f64[0])
        t_end        = float(ts_f64[-1])
        n_total_bins = int((t_end - t_start) / encoder.dt_us)

        fl, ll = [], []
        for i in range(0, n_total_bins - n_bins, stride_bins):
            ws = t_start + i * encoder.dt_us
            we = ws + n_bins * encoder.dt_us
            lb = label[i:i + n_bins]
            if len(lb) == n_bins:
                lo = int(np.searchsorted(ts_f64, ws, side="left"))
                hi = int(np.searchsorted(ts_f64, we, side="left"))
                fl.append(encoder._encode_columns(
                    ts_f64[lo:hi], xs_i32[lo:hi],
                    ys_i32[lo:hi], ps_i32[lo:hi], ws, n_bins))
                ll.append(torch.from_numpy(lb.copy()))

        self.all_frames = torch.stack(fl) if fl else torch.empty(0)
        self.all_labels = torch.stack(ll) if ll else torch.empty(0)

    def __len__(self):   return len(self.all_frames)
    def __getitem__(self, i): return self.all_frames[i], self.all_labels[i]


# ── Label helpers ─────────────────────────────────────────────────────────────

def make_label_from_trajectory(events, h5_path, encoder):
    with h5py.File(h5_path, "r") as f:
        traj = f["obstacle_positions"][:]
        dp   = f["drone_hover_position"][:]
        r    = float(f["obstacle_radius"][()])
        dt   = float(f["sim_dt"][()])
        launch_step = int(f["launch_step"][()]) if "launch_step" in f else 0

    # Guard against trajectory double-sampling: Pegasus calls backend.update()
    # at the physics substep rate (typically 2× the render rate).  If the
    # trajectory was NOT subsampled at save time, dt_stored = 1/FPS but each
    # point is actually 1/(2*FPS) apart → the trajectory appears 2× too long.
    # Detect this by comparing traj duration to event recording duration and
    # correct dt accordingly.
    t0_us = float(events[0, 0])
    t1_us = float(events[-1, 0])
    event_dur_s  = (t1_us - t0_us) * 1e-6
    traj_dur_s   = len(traj) * dt
    # Allow up to 30% overshoot (trajectory may run a little longer than events)
    if traj_dur_s > event_dur_s * 1.3:
        factor = round(traj_dur_s / event_dur_s)
        dt = dt / factor   # correct to actual per-sample interval

    dth = angular_velocity_label(traj, dp, r, dt)   # shape (T_sim,)

    # Physical time for each trajectory sample (seconds from sim start)
    traj_t_s  = np.arange(len(dth), dtype=np.float64) * dt

    # Physical time axis for the event bins (seconds from recording start)
    n_bins = int((t1_us - t0_us) / encoder.dt_us)
    event_t_s = np.linspace(t0_us, t1_us, n_bins) * 1e-6

    # Interpolate dθ/dt onto the event time axis; np.interp clamps at edges.
    lab = np.interp(event_t_s, traj_t_s, dth).astype(np.float32)
    return lab / (lab.max() + 1e-6)


def make_label_from_event_rate(events, encoder):
    from scipy.ndimage import uniform_filter1d
    n  = int((float(events[-1, 0]) - float(events[0, 0])) / encoder.dt_us)
    bi = np.clip(((events[:, 0] - events[0, 0]) / encoder.dt_us).astype(np.int32), 0, n-1)
    c  = np.bincount(bi, minlength=n).astype(np.float32)
    c  = uniform_filter1d(c, size=5)
    return c / (c.max() + 1e-6)


def _load_single(path, encoder, n_bins, tag=""):
    pfx = f"[{tag}] " if tag else "  "
    print(f"{pfx}{path}", flush=True)
    with h5py.File(path, "r") as f:
        ev  = f["events"][:]
        has = "obstacle_positions" in f
    lab  = make_label_from_trajectory(ev, path, encoder) if has else make_label_from_event_rate(ev, encoder)
    frac = float((lab > 0.05).mean()) * 100
    src  = "analytical" if has else "event-rate"
    print(f"{pfx}  {len(ev):,} events  |  looming: {frac:.1f}%  ({src})", flush=True)
    t0 = time.time()
    ds = LoomingDataset(ev, lab, encoder, n_bins, stride_bins=n_bins // 2)
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
    enc = EventEncoder(height=args.height, width=args.width,
                       dt_us=args.dt_us, mode="binary")

    h5s = args.h5 if isinstance(args.h5, list) else [args.h5]
    print(f"\nTrain ({len(h5s)}):", flush=True)
    tr = ConcatDataset([_load_single(p, enc, args.n_bins, "tr") for p in h5s])
    print(f"  {len(tr)} windows  batch={args.batch}  {len(tr)//args.batch} batches/ep", flush=True)

    va = None
    if args.val_h5:
        vs = args.val_h5 if isinstance(args.val_h5, list) else [args.val_h5]
        print(f"\nVal ({len(vs)}):", flush=True)
        va = ConcatDataset([_load_single(p, enc, args.n_bins, "va") for p in vs])
        print(f"  {len(va)} windows held out", flush=True)

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    num_workers=0, pin_memory=(device.type == "cuda"))
    vl = (DataLoader(va, batch_size=args.batch, shuffle=False,
                     num_workers=0, pin_memory=(device.type == "cuda")) if va else None)

    model = LGMDNet(height=args.height, width=args.width,
                    pool_factor=args.pool, tau_mem=args.tau).to(device)
    np_ = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {np_} trainable params  LIF_thresh={model.lgmd_lif.v_threshold}"
          f"  exc_weight={args.exc_weight}", flush=True)

    opt  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    hdr  = f"{'Ep':>5}  {'TrLoss':>8}  {'VaLoss':>8}  {'ExCorr':>8}  {'DcCorr':>8}  {'Ex_loom':>8}  {'Ex_bg':>6}  {'LR':>8}  {'s':>4}"
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
    p.add_argument("--height", type=int,   default=260)
    p.add_argument("--width",  type=int,   default=346)
    p.add_argument("--dt_us",  type=float, default=5000.0)
    p.add_argument("--n_bins", type=int,   default=20)
    p.add_argument("--pool",   type=int,   default=4)
    p.add_argument("--tau",    type=float, default=2.0)
    p.add_argument("--epochs", type=int,   default=300)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--batch",  type=int,   default=64)
    p.add_argument("--exc_weight",  type=float, default=0.8,
                   help="Weight for excitation vs spike Pearson loss (default 0.8)")
    p.add_argument("--bg_penalty", type=float, default=0.05)
    p.add_argument("--log_every",  type=int,   default=10)
    p.add_argument("--save",   default="results/lgmd_weights.pt")
    train(p.parse_args())
