"""
Event stream encoder: raw HDF5 events -> spike frame tensors for LGMD input.

Converts (N, 4) uint32 [timestamp_us, x, y, polarity] events into
(T, 2, H, W) float tensors suitable for LGMDNet.

Two encoding modes:
  'binary'  -- each pixel fires 0 or 1 per time bin (ignores event count)
  'count'   -- raw event count per pixel per bin, normalised to [0, 1]

Binary is closer to biology and avoids magnitude scaling issues.

Optional spatial_downsample pools event coordinates before binning,
producing (T, 2, H//ds, W//ds) output directly.  This is 16x faster
at ds=4 and the model can skip its AvgPool2d (set pool_factor=1).
"""

import numpy as np
import torch


class EventEncoder:
    """
    Encodes a flat event array into a sequence of spike frames.

    Args:
        height:              sensor height in pixels
        width:               sensor width in pixels
        dt_us:               time bin width in microseconds
        mode:                'binary' or 'count'
        clip_count:          for 'count' mode -- clip counts at this value before normalising
        spatial_downsample:  pool event coordinates by this factor before binning
    """

    def __init__(
        self,
        height: int = 260,
        width: int = 346,
        dt_us: float = 5000.0,
        mode: str = "binary",
        clip_count: int = 5,
        spatial_downsample: int = 1,
    ):
        self.height = height
        self.width = width
        self.dt_us = dt_us
        self.mode = mode
        self.clip_count = clip_count
        self.spatial_downsample = spatial_downsample
        self.enc_height = height // spatial_downsample
        self.enc_width = width // spatial_downsample
        self._frame_stride = 2 * self.enc_height * self.enc_width

    def encode(self, events: np.ndarray,
               t_start_us: float | None = None,
               t_end_us: float | None = None) -> torch.Tensor:
        """
        Encode events into a spike frame sequence.

        Args:
            events:     (N, 4) uint32 -- [timestamp_us, x, y, polarity]
            t_start_us: start of encoding window (default: first event)
            t_end_us:   end of encoding window (default: last event)

        Returns:
            frames: (T, 2, enc_H, enc_W) float32 tensor
                    dim 1: channel 0 = ON, channel 1 = OFF
        """
        if len(events) == 0:
            n_bins = 1 if (t_start_us is None or t_end_us is None) else \
                     max(1, int((t_end_us - t_start_us) / self.dt_us))
            return torch.zeros(n_bins, 2, self.enc_height, self.enc_width)

        if t_start_us is None:
            t_start_us = float(events[0, 0])
        if t_end_us is None:
            t_end_us = float(events[-1, 0])

        n_bins = max(1, int((t_end_us - t_start_us) / self.dt_us))

        ts = events[:, 0].astype(np.float64)
        xs = events[:, 1].astype(np.int32)
        ys = events[:, 2].astype(np.int32)
        ps = events[:, 3].astype(np.int32)

        mask = (ts >= t_start_us) & (ts < t_end_us)
        ts, xs, ys, ps = ts[mask], xs[mask], ys[mask], ps[mask]

        return self._encode_columns(ts, xs, ys, ps, t_start_us, n_bins)

    def _encode_columns(
        self,
        ts: np.ndarray,    # float64, already filtered to window
        xs: np.ndarray,    # int32
        ys: np.ndarray,    # int32
        ps: np.ndarray,    # int32
        t_start_us: float,
        n_bins: int,
    ) -> torch.Tensor:
        """
        Inner encode from pre-typed, pre-filtered column arrays.
        Called by LoomingDataset.__getitem__ with pre-cast slices to avoid
        per-call type copies. Uses np.bincount (~5x faster than np.add.at).
        """
        H = self.enc_height
        W = self.enc_width
        total = n_bins * self._frame_stride

        if len(ts) > 0:
            bin_idx = np.clip(
                ((ts - t_start_us) / self.dt_us).astype(np.int32),
                0, n_bins - 1,
            )
            ds = self.spatial_downsample
            if ds > 1:
                xs_c = np.clip(xs // ds, 0, W - 1)
                ys_c = np.clip(ys // ds, 0, H - 1)
            else:
                xs_c = np.clip(xs, 0, W - 1)
                ys_c = np.clip(ys, 0, H - 1)

            flat_idx = (bin_idx * self._frame_stride
                        + ps * (H * W)
                        + ys_c * W + xs_c)

            counts = np.bincount(flat_idx, minlength=total)
        else:
            counts = np.zeros(total, dtype=np.intp)

        frames = counts.reshape(n_bins, 2, H, W).astype(np.float32)

        if self.mode == "binary":
            frames = (frames > 0).astype(np.float32)
        else:
            frames = np.clip(frames, 0, self.clip_count) / self.clip_count

        return torch.from_numpy(frames)   # (T, 2, enc_H, enc_W)

    def encode_window(self, events: np.ndarray,
                      t_centre_us: float,
                      n_bins: int = 20) -> torch.Tensor:
        """
        Encode a fixed-length window of n_bins centred on t_centre_us.

        Returns: (T, 2, enc_H, enc_W) where T = n_bins
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
    Compute the analytical DCMD label: angular expansion rate dtheta/dt.

    For a sphere of radius r approaching at speed v from distance d:
        theta(t)  = 2 * arctan(r / d(t))
        dtheta/dt ~= 2 * r * v / (d^2 + r^2)

    Args:
        obstacle_pos:    (T, 3) obstacle positions over time
        drone_pos:       (3,) or (T, 3) drone position(s)
        obstacle_radius: approximate obstacle half-size in metres
        dt_s:            time step between positions in seconds

    Returns:
        (T,) dtheta/dt in radians/second, clipped to [0, inf)
    """
    if drone_pos.ndim == 1:
        drone_pos = np.tile(drone_pos, (len(obstacle_pos), 1))

    diff = obstacle_pos - drone_pos
    d = np.linalg.norm(diff, axis=1)
    d = np.maximum(d, obstacle_radius)

    theta = 2.0 * np.arctan2(obstacle_radius, d)
    dtheta_dt = np.gradient(theta, dt_s)
    return np.clip(dtheta_dt, 0.0, None).astype(np.float32)
