"""
Event stream encoder: raw HDF5 events → spike frame tensors for LGMD input.

Converts (N, 4) uint32 [timestamp_us, x, y, polarity] events into
(T, B, 2, H, W) float tensors suitable for LGMDNet.

Two encoding modes:
  'binary'  — each pixel fires 0 or 1 per time bin (ignores event count)
  'count'   — raw event count per pixel per bin, normalised to [0, 1]

Binary is closer to biology and avoids magnitude scaling issues.
"""

import numpy as np
import torch


class EventEncoder:
    """
    Encodes a flat event array into a sequence of spike frames.

    Args:
        height:      sensor height in pixels
        width:       sensor width in pixels
        dt_us:       time bin width in microseconds
        mode:        'binary' or 'count'
        clip_count:  for 'count' mode — clip counts at this value before normalising
    """

    def __init__(
        self,
        height: int = 260,
        width: int = 346,
        dt_us: float = 5000.0,   # 5 ms default — covers ~5 looming edge expansions
        mode: str = "binary",
        clip_count: int = 5,
    ):
        self.height = height
        self.width = width
        self.dt_us = dt_us
        self.mode = mode
        self.clip_count = clip_count

    def encode(self, events: np.ndarray,
               t_start_us: float | None = None,
               t_end_us: float | None = None) -> torch.Tensor:
        """
        Encode events into a spike frame sequence.

        Args:
            events:     (N, 4) uint32 — [timestamp_us, x, y, polarity]
            t_start_us: start of encoding window (default: first event)
            t_end_us:   end of encoding window (default: last event)

        Returns:
            frames: (T, 2, H, W) float32 tensor
                    dim 1: channel 0 = ON, channel 1 = OFF
        """
        if t_start_us is None:
            t_start_us = float(events[0, 0])
        if t_end_us is None:
            t_end_us = float(events[-1, 0])

        n_bins = max(1, int((t_end_us - t_start_us) / self.dt_us))
        frames = np.zeros((n_bins, 2, self.height, self.width), dtype=np.float32)

        ts = events[:, 0].astype(np.float64)
        xs = events[:, 1].astype(np.int32)
        ys = events[:, 2].astype(np.int32)
        ps = events[:, 3].astype(np.int32)

        # Filter to time window first (avoids processing full array each call)
        mask = (ts >= t_start_us) & (ts < t_end_us)
        ts, xs, ys, ps = ts[mask], xs[mask], ys[mask], ps[mask]

        if len(ts) > 0:
            # Vectorised bin assignment
            bin_idx = ((ts - t_start_us) / self.dt_us).astype(np.int32)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            xs = np.clip(xs, 0, self.width - 1)
            ys = np.clip(ys, 0, self.height - 1)

            # Accumulate into frames without a Python loop
            flat_idx = bin_idx * (2 * self.height * self.width) + \
                       ps * (self.height * self.width) + \
                       ys * self.width + xs
            np.add.at(frames.ravel(), flat_idx, 1.0)

        if self.mode == "binary":
            frames = (frames > 0).astype(np.float32)
        else:  # count
            frames = np.clip(frames, 0, self.clip_count) / self.clip_count

        return torch.from_numpy(frames)  # (T, 2, H, W)

    def encode_window(self, events: np.ndarray,
                      t_centre_us: float,
                      n_bins: int = 20) -> torch.Tensor:
        """
        Encode a fixed-length window of n_bins centred on t_centre_us.
        Useful for extracting training samples around a known collision time.

        Returns: (T, 2, H, W) where T = n_bins
        """
        half = (n_bins / 2) * self.dt_us
        return self.encode(events,
                           t_start_us=t_centre_us - half,
                           t_end_us=t_centre_us + half)


def angular_velocity_label(
    obstacle_pos: np.ndarray,
    drone_pos: np.ndarray,
    obstacle_radius: float,
    dt_s: float,
) -> np.ndarray:
    """
    Compute the analytical DCMD label: angular expansion rate dθ/dt.

    For a sphere of radius r approaching at speed v from distance d:
        θ(t)  = 2 * arctan(r / d(t))
        dθ/dt ≈ 2 * r * v / (d² + r²)

    Args:
        obstacle_pos:    (T, 3) obstacle positions over time
        drone_pos:       (3,) or (T, 3) drone position(s)
        obstacle_radius: approximate obstacle half-size in metres
        dt_s:            time step between positions in seconds

    Returns:
        (T,) dθ/dt in radians/second, clipped to [0, ∞)
    """
    if drone_pos.ndim == 1:
        drone_pos = np.tile(drone_pos, (len(obstacle_pos), 1))

    diff = obstacle_pos - drone_pos
    d = np.linalg.norm(diff, axis=1)  # (T,) distance
    d = np.maximum(d, obstacle_radius)  # avoid division by zero

    theta = 2.0 * np.arctan2(obstacle_radius, d)

    # Finite difference for dθ/dt
    dtheta_dt = np.gradient(theta, dt_s)
    return np.clip(dtheta_dt, 0.0, None).astype(np.float32)
