"""
Visualise synthetic event stream from v2e HDF5 output.

Usage:
  python events/visualise_events.py --h5 /tmp/sim_events_obstacles/events.h5
  python events/visualise_events.py --h5 /tmp/sim_events_obstacles/events.h5 --save /tmp/event_vis.mp4

Event format: [timestamp_us, x, y, polarity]
"""

import argparse
import numpy as np
import h5py
import cv2
from tqdm import tqdm


def load_events(h5_path):
    with h5py.File(h5_path, "r") as f:
        events = f["events"][:]  # (N, 4) uint32: [t_us, x, y, p]
    print(f"Loaded {len(events):,} events")
    print(f"  Time range: {events[0,0]/1e6:.3f}s — {events[-1,0]/1e6:.3f}s")
    print(f"  ON events:  {(events[:,3]==1).sum():,}")
    print(f"  OFF events: {(events[:,3]==0).sum():,}")
    return events


def events_to_frame(events, width, height, t_start_us, t_end_us):
    """Accumulate events in [t_start, t_end] into an RGB frame."""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 128  # grey background
    mask = (events[:, 0] >= t_start_us) & (events[:, 0] < t_end_us)
    chunk = events[mask]
    if len(chunk) == 0:
        return frame
    on  = chunk[chunk[:, 3] == 1]
    off = chunk[chunk[:, 3] == 0]
    # ON = white, OFF = black
    frame[on[:, 2].clip(0, height-1), on[:, 1].clip(0, width-1)] = [255, 255, 255]
    frame[off[:, 2].clip(0, height-1), off[:, 1].clip(0, width-1)] = [0, 0, 0]
    return frame


def render_video(events, width, height, window_ms=33, fps=30, save_path=None):
    """Render event stream as video with accumulation windows."""
    t_start = int(events[0, 0])
    t_end   = int(events[-1, 0])
    window_us = int(window_ms * 1000)
    t_step_us = int(1e6 / fps)

    frames = []
    for t in tqdm(range(t_start, t_end, t_step_us), desc="Rendering", unit="frame"):
        frame = events_to_frame(events, width, height, t, t + window_us)

        # Overlay stats
        n_events = ((events[:, 0] >= t) & (events[:, 0] < t + window_us)).sum()
        cv2.putText(frame, f"t={t/1e6:.3f}s  events={n_events}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        frames.append(frame)

    if save_path:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (width, height))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Saved to {save_path}")
    return frames


def print_stats(events, width, height, bin_ms=100):
    """Print event rate statistics over time."""
    t_start = events[0, 0]
    t_end   = events[-1, 0]
    bin_us  = bin_ms * 1000
    print(f"\nEvent rate ({bin_ms}ms bins):")
    for t in range(int(t_start), int(t_end), int(bin_us)):
        mask = (events[:, 0] >= t) & (events[:, 0] < t + bin_us)
        n = mask.sum()
        on_frac = events[mask, 3].mean() if n > 0 else 0
        bar = "█" * min(int(n / 5000), 40)
        print(f"  {t/1e6:.2f}s: {n:6,} events  ON={on_frac:.2f}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="Path to events.h5")
    parser.add_argument("--width",  type=int, default=346)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--window_ms", type=float, default=33,
                        help="Accumulation window in ms per rendered frame")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save", help="Save rendered video to this path")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print stats only, skip video render")
    args = parser.parse_args()

    events = load_events(args.h5)
    print_stats(events, args.width, args.height)

    if not args.stats_only:
        save = args.save or args.h5.replace("events.h5", "event_vis.mp4")
        render_video(events, args.width, args.height,
                     window_ms=args.window_ms, fps=args.fps, save_path=save)
