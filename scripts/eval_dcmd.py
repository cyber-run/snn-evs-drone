"""
DCMD signal visualisation over a full event recording.

For a trained LGMDNet + a recorded H5 file, slides a causal n_bins window
over the full event stream and plots:
  - Event rate per bin (grey, left axis)
  - Normalised dθ/dt label (blue, right axis)
  - DCMD output — smoothed (red, right axis)
  - Vertical line at launch_step

Usage:
    python scripts/eval_dcmd.py \
        --h5  /tmp/evasion_head_on_events/events.h5 \
        --weights results/lgmd_weights.pt \
        --out results/eval_dcmd_head_on.png
"""

import argparse
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snn.models.event_encoder import EventEncoder
from snn.models.lgmd_net import LGMDNet
from snn.training.train_lgmd import make_label_from_trajectory


def eval_recording(h5_path: str, weights_path: str,
                   dt_us: float = 10_000.0,
                   n_bins: int = 20,
                   pool: int = 4,
                   batch_size: int = 128,
                   smooth_window: int = 5,
                   device: torch.device | None = None) -> dict:
    """
    Slide a causal n_bins window over the recording and collect DCMD per bin.

    Returns a dict with keys:
        t_s          (N,)   time axis in seconds from recording start
        event_rate   (N,)   events per bin (normalised to [0,1])
        label        (N,)   normalised dθ/dt (0=background, 1=peak loom)
        dcmd         (N,)   raw DCMD output per window
        dcmd_smooth  (N,)   causal moving-average of dcmd
        launch_t_s   float  launch time in seconds from recording start
        profile      str    h5 filename stem
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load events ───────────────────────────────────────────────────────────
    with h5py.File(h5_path, "r") as f:
        events = f["events"][:]
        launch_step = int(f["launch_step"][()]) if "launch_step" in f else None
        sim_dt      = float(f["sim_dt"][()])    if "sim_dt"       in f else None

    events = events[np.argsort(events[:, 0], kind="stable")]
    t0_us  = float(events[0, 0])
    t1_us  = float(events[-1, 0])

    # ── Encoder ───────────────────────────────────────────────────────────────
    enc = EventEncoder(dt_us=dt_us, spatial_downsample=pool)
    H, W = enc.enc_height, enc.enc_width

    # Encode full recording into per-bin frames: (N_bins, 2, H, W)
    n_total = int((t1_us - t0_us) / dt_us)
    ts_f64  = events[:, 0].astype(np.float64)
    xs_i32  = events[:, 1].astype(np.int32)
    ys_i32  = events[:, 2].astype(np.int32)
    ps_i32  = events[:, 3].astype(np.int32)

    # Event rate per bin (for plotting)
    bin_idx  = np.clip(((ts_f64 - t0_us) / dt_us).astype(np.int32), 0, n_total - 1)
    ev_count = np.bincount(bin_idx, minlength=n_total).astype(np.float32)
    ev_rate  = ev_count / (ev_count.max() + 1e-6)

    # Encode each bin individually (fast: one bincount over all events)
    all_frames = enc.encode(events, t_start_us=t0_us,
                            t_end_us=t0_us + n_total * dt_us)  # (N_bins, 2, H, W)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LGMDNet(height=H, width=W, pool_factor=1)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    # ── Sliding causal window → DCMD per position ────────────────────────────
    # Window at position i uses frames [i-n_bins+1 .. i], output = DCMD at last step.
    # Positions 0..n_bins-2 are padded with zeros at the start.
    dcmd_out = np.zeros(n_total, dtype=np.float32)

    # Build list of window tensors (batch for efficiency)
    starts = range(0, n_total)
    windows = []
    for i in starts:
        lo  = max(0, i - n_bins + 1)
        pad = n_bins - (i - lo + 1)
        w   = all_frames[lo:i + 1]          # (≤n_bins, 2, H, W)
        if pad > 0:
            w = torch.cat([torch.zeros(pad, 2, H, W), w], dim=0)
        windows.append(w)

    # Batch inference
    for b_start in range(0, n_total, batch_size):
        b_end   = min(b_start + batch_size, n_total)
        batch   = torch.stack(windows[b_start:b_end])    # (B, n_bins, 2, H, W)
        batch   = batch.permute(1, 0, 2, 3, 4).to(device)  # (T, B, 2, H, W)
        with torch.no_grad():
            dc, _, _ = model(batch)                      # (T, B)
        dcmd_out[b_start:b_end] = dc[-1].cpu().numpy()

    # Smooth: causal moving average
    kernel = np.ones(smooth_window) / smooth_window
    dcmd_smooth = np.convolve(dcmd_out, kernel, mode="full")[:n_total]

    # ── Label ─────────────────────────────────────────────────────────────────
    label = make_label_from_trajectory(events, h5_path, enc, binary=False)
    # Interpolate label to match n_total bins if needed
    if len(label) != n_total:
        x_src = np.linspace(0, 1, len(label))
        x_dst = np.linspace(0, 1, n_total)
        label = np.interp(x_dst, x_src, label).astype(np.float32)

    # ── Time axis and launch marker ───────────────────────────────────────────
    t_s = (np.arange(n_total) * dt_us + t0_us) * 1e-6

    launch_t_s = None
    if launch_step is not None and sim_dt is not None:
        launch_t_s = launch_step * sim_dt   # seconds from sim start
        # Offset to recording time: recording starts at t=0 in event timestamps
        # but sim starts earlier (warmup). Align via label peak.
        label_peak_bin   = int(np.argmax(label))
        label_peak_t_s   = float(t_s[label_peak_bin])
        # Use raw launch time as absolute marker
        launch_t_s = launch_step * sim_dt

    profile = Path(h5_path).parent.name.replace("evasion_", "").replace("_events", "")

    return dict(
        t_s=t_s, event_rate=ev_rate, label=label,
        dcmd=dcmd_out, dcmd_smooth=dcmd_smooth,
        launch_t_s=launch_t_s, profile=profile,
    )


def plot_dcmd(result: dict, out_path: str) -> None:
    """Render and save the DCMD signal plot."""
    t   = result["t_s"]
    er  = result["event_rate"]
    lbl = result["label"]
    dc  = result["dcmd"]
    dcs = result["dcmd_smooth"]
    lt  = result["launch_t_s"]
    title = result["profile"]

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()

    # Event rate (grey bars, left axis)
    ax1.fill_between(t, er, color="silver", alpha=0.6, label="Event rate")
    ax1.set_ylabel("Event rate (normalised)", color="grey")
    ax1.tick_params(axis="y", labelcolor="grey")
    ax1.set_ylim(0, 1.05)

    # dθ/dt label (blue, right axis)
    ax2.plot(t, lbl, color="steelblue", linewidth=1.5, label="dθ/dt label")

    # DCMD (raw faint, smoothed bold, red, right axis)
    ax2.plot(t, dc,  color="tomato", linewidth=0.8, alpha=0.4)
    ax2.plot(t, dcs, color="red",    linewidth=2.0, label="DCMD (smoothed)")
    ax2.set_ylabel("LGMD / dθ/dt (normalised)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(bottom=0)

    # Launch marker
    if lt is not None:
        ax1.axvline(lt, color="black", linestyle="--", linewidth=1.2,
                    label=f"Launch (t={lt:.2f}s)")

    ax1.set_xlabel("Time (s from recording start)")
    ax1.set_title(f"LGMD DCMD signal — profile: {title}")

    # Merge legends
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=9)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",      required=True,
                   help="Path to events.h5 recording")
    p.add_argument("--weights", required=True,
                   help="Path to saved model weights (.pt)")
    p.add_argument("--out",     default="results/eval_dcmd.png",
                   help="Output plot path")
    p.add_argument("--dt_us",   type=float, default=10_000.0)
    p.add_argument("--n_bins",  type=int,   default=20)
    p.add_argument("--pool",    type=int,   default=4)
    p.add_argument("--smooth",  type=int,   default=5,
                   help="Causal smoothing window in bins")
    p.add_argument("--batch",   type=int,   default=128,
                   help="Inference batch size (number of windows per forward pass)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading: {args.h5}")

    result = eval_recording(
        h5_path=args.h5,
        weights_path=args.weights,
        dt_us=args.dt_us,
        n_bins=args.n_bins,
        pool=args.pool,
        batch_size=args.batch,
        smooth_window=args.smooth,
        device=device,
    )

    # Print quick stats
    dc   = result["dcmd"]
    lbl  = result["label"]
    loom = lbl > 0.5
    print(f"Profile:        {result['profile']}")
    print(f"Bins:           {len(dc)}")
    print(f"DCMD loom mean: {dc[loom].mean():.4f}" if loom.any() else "DCMD loom mean: n/a")
    print(f"DCMD bg mean:   {dc[~loom].mean():.4f}" if (~loom).any() else "DCMD bg mean:  n/a")

    plot_dcmd(result, args.out)


if __name__ == "__main__":
    main()
