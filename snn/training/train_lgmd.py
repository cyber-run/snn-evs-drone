"""
Train the LGMD SNN on synthetic looming event sequences.

Training signal: Option A — analytical dθ/dt label from known obstacle trajectories.
The network learns to produce a DCMD spike rate proportional to angular expansion rate.

Usage:
  python snn/training/train_lgmd.py --h5 /tmp/sim_events_obstacles/events.h5
  python snn/training/train_lgmd.py --h5 /tmp/sim_events_obstacles/events.h5 \
      --epochs 50 --lr 1e-3 --save results/lgmd_weights.pt

Input HDF5 must have an 'events' dataset of shape (N, 4) uint32
and optionally an 'obstacle_trajectory' dataset of shape (M, 3) float32.

If no trajectory data is available, the script falls back to a heuristic label
derived from the temporal event rate (rising rate → looming stimulus).
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from snn.models.lgmd_net import LGMDNet
from snn.models.event_encoder import EventEncoder, angular_velocity_label


# ── Dataset ───────────────────────────────────────────────────────────────────

class LoomingDataset(Dataset):
    """
    Sliding-window dataset over a single event recording.

    Each sample is a (T, 2, H, W) spike frame sequence with a (T,) dθ/dt label.
    Windows overlap by 50% to maximise training data from a single run.
    """

    def __init__(
        self,
        events: np.ndarray,
        label: np.ndarray,        # (T_total,) dθ/dt at dt_us resolution
        encoder: EventEncoder,
        n_bins: int = 20,         # time bins per sample
        stride_bins: int = 10,    # stride for sliding window
    ):
        self.events = events
        self.encoder = encoder
        self.n_bins = n_bins

        t_start = float(events[0, 0])
        t_end = float(events[-1, 0])
        n_total_bins = int((t_end - t_start) / encoder.dt_us)

        # Build list of (t_start_us, label_slice) for each window
        self.windows = []
        for i in range(0, n_total_bins - n_bins, stride_bins):
            ws_us = t_start + i * encoder.dt_us
            we_us = ws_us + n_bins * encoder.dt_us
            lbl = label[i:i + n_bins]
            if len(lbl) == n_bins:
                self.windows.append((ws_us, we_us, lbl.copy()))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ws_us, we_us, lbl = self.windows[idx]
        frames = self.encoder.encode(self.events, t_start_us=ws_us, t_end_us=we_us)
        # frames: (T, 2, H, W)
        label_t = torch.from_numpy(lbl)   # (T,)
        return frames, label_t


# ── Label generation ──────────────────────────────────────────────────────────

def make_label_from_trajectory(events, h5_path, encoder):
    """Load trajectory from HDF5 and compute analytical dθ/dt label."""
    with h5py.File(h5_path, "r") as f:
        traj = f["obstacle_trajectory"][:]   # (M, 3) positions at sim dt
        drone_pos = f["drone_position"][:]   # (3,) fixed hover position
        obs_radius = float(f["obstacle_radius"][()])
        sim_dt = float(f["sim_dt"][()])

    dtheta = angular_velocity_label(traj, drone_pos, obs_radius, sim_dt)

    # Resample dtheta to encoder time bins
    t_start = float(events[0, 0])
    t_end = float(events[-1, 0])
    n_bins = int((t_end - t_start) / encoder.dt_us)
    orig_t = np.linspace(0, 1, len(dtheta))
    new_t  = np.linspace(0, 1, n_bins)
    label = np.interp(new_t, orig_t, dtheta).astype(np.float32)

    # Normalise to [0, 1] for stable training
    label = label / (label.max() + 1e-6)
    return label


def make_label_from_event_rate(events, encoder):
    """
    Heuristic fallback: event rate in each bin, normalised.
    Rising event rate correlates with looming for most approach trajectories.
    """
    t_start = float(events[0, 0])
    t_end = float(events[-1, 0])
    n_bins = int((t_end - t_start) / encoder.dt_us)
    counts = np.zeros(n_bins, dtype=np.float32)

    ts = events[:, 0].astype(np.float64)
    bin_idx = ((ts - t_start) / encoder.dt_us).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    np.add.at(counts, bin_idx, 1)

    # Smooth and normalise
    from scipy.ndimage import uniform_filter1d
    counts = uniform_filter1d(counts, size=5)
    counts = counts / (counts.max() + 1e-6)
    return counts


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading events from {args.h5}")
    with h5py.File(args.h5, "r") as f:
        events = f["events"][:]
    print(f"  {len(events):,} events loaded")

    encoder = EventEncoder(
        height=args.height,
        width=args.width,
        dt_us=args.dt_us,
        mode="binary",
    )

    # Build label
    with h5py.File(args.h5, "r") as f:
        has_traj = "obstacle_trajectory" in f

    if has_traj:
        print("Using analytical dθ/dt label from trajectory data")
        label = make_label_from_trajectory(events, args.h5, encoder)
    else:
        print("No trajectory data found — using event-rate heuristic label")
        label = make_label_from_event_rate(events, encoder)

    dataset = LoomingDataset(events, label, encoder,
                             n_bins=args.n_bins, stride_bins=args.n_bins // 2)
    print(f"  {len(dataset)} training windows")

    if len(dataset) == 0:
        print("ERROR: No windows generated. Check event count and dt_us setting.")
        return

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    model = LGMDNet(
        height=args.height,
        width=args.width,
        pool_factor=args.pool,
        tau_mem=args.tau,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    print(f"\nTraining LGMD SNN for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for frames, lbl in loader:
            # frames: (B, T, 2, H, W) → need (T, B, 2, H, W)
            frames = frames.permute(1, 0, 2, 3, 4).to(device)
            lbl = lbl.permute(1, 0).to(device)   # (T, B)

            optimiser.zero_grad()
            dcmd_spikes, _ = model(frames)        # (T, B)

            # Normalise spike count to [0, 1] for MSE against normalised label
            dcmd_norm = dcmd_spikes / (dcmd_spikes.max() + 1e-6)
            loss = loss_fn(dcmd_norm, lbl)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(loader), 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"\nWeights saved to {args.save}")

    print("Training complete.")
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",      required=True, help="Path to events.h5")
    parser.add_argument("--height",  type=int,   default=260)
    parser.add_argument("--width",   type=int,   default=346)
    parser.add_argument("--dt_us",   type=float, default=5000.0,
                        help="Time bin width in microseconds (default 5ms)")
    parser.add_argument("--n_bins",  type=int,   default=20,
                        help="Time bins per training window")
    parser.add_argument("--pool",    type=int,   default=4,
                        help="Spatial pooling factor")
    parser.add_argument("--tau",     type=float, default=2.0,
                        help="LIF membrane time constant")
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--save",    default="results/lgmd_weights.pt",
                        help="Path to save model weights")
    args = parser.parse_args()
    train(args)
