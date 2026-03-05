"""
LGMD-inspired spiking neural network for looming detection.

Architecture mirrors the locust Lobula Giant Movement Detector circuit:
  P layer  → photoreceptor-like spatial pooling (no spikes)
  S layer  → LIF excitation + delayed lateral inhibition
  LGMD     → integration of (excitation - inhibition)
  DCMD     → global spatial sum → collision-imminence spike rate

Input:  (T, B, 2, H, W)  — T time bins, batch B, 2 polarities, H×W pixels
Output: (T, B)            — DCMD spike count per time bin per sample

References:
  Rind & Bramwell (1996) — original computational LGMD model
  Stafford et al. (2007) — LGMD robot implementation
  Meng et al. (2023)     — SNN LGMD for UAV avoidance
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Build a 2-D Gaussian kernel for lateral inhibition initialisation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    return kernel


class LGMDNet(nn.Module):
    """
    LGMD spiking network.

    Args:
        height:        input frame height (after any pre-resize)
        width:         input frame width
        pool_factor:   spatial pooling divisor applied before S layer
        tau_mem:       LIF membrane time constant (shared across S and LGMD layers)
        inh_kernel_size: size of lateral inhibition conv kernel (must be odd)
        inh_sigma:     Gaussian sigma for inhibition kernel initialisation
        inh_delay:     number of time steps to delay inhibition (1 = biological default)
    """

    def __init__(
        self,
        height: int = 260,
        width: int = 346,
        pool_factor: int = 4,
        tau_mem: float = 2.0,
        inh_kernel_size: int = 7,
        inh_sigma: float = 1.5,
        inh_delay: int = 1,
    ):
        super().__init__()
        self.pool_factor = pool_factor
        self.inh_delay = inh_delay
        self.h = height // pool_factor
        self.w = width // pool_factor

        # ── P layer: spatial pooling ──────────────────────────────────────────
        # Mimics retinal photoreceptor pooling. Not spiking.
        self.pool = nn.AvgPool2d(pool_factor)

        # ── S layer: excitation ───────────────────────────────────────────────
        # ON channel drives positive excitation; OFF channel drives weak negative
        # (1, 2, 3, 3) conv collapses polarity channels to single excitation map
        self.exc_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.exc_conv.weight[:, 0:1], 1.0)   # ON  → +excitation
        nn.init.constant_(self.exc_conv.weight[:, 1:2], -0.3)  # OFF → weak inhibition

        # ── S layer: lateral inhibition ───────────────────────────────────────
        # Gaussian spatial spread, applied to previous time step's S-layer activity.
        # Negative weights: inhibition suppresses LGMD excitation.
        self.inh_conv = nn.Conv2d(1, 1, kernel_size=inh_kernel_size,
                                  padding=inh_kernel_size // 2,
                                  bias=False, groups=1)
        with torch.no_grad():
            kernel = _gaussian_kernel(inh_kernel_size, inh_sigma)
            # Negative: inhibition reduces LGMD membrane potential
            self.inh_conv.weight.data = -kernel.view(1, 1, inh_kernel_size, inh_kernel_size)

        # ── LGMD LIF neurons ──────────────────────────────────────────────────
        # One LIF neuron per spatial location; integrates (excitation + inhibition)
        self.lgmd_lif = neuron.LIFNode(tau=tau_mem, detach_reset=True)

        # ── DCMD readout ──────────────────────────────────────────────────────
        # Learnable spatial weighting before global sum.
        # Initialised to uniform — learns to weight central field-of-view more.
        self.dcmd_weight = nn.Parameter(
            torch.ones(1, 1, self.h, self.w) / (self.h * self.w)
        )

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (T, B, 2, H, W) float tensor, values in [0, 1]
               channel 0 = ON events, channel 1 = OFF events

        Returns:
            dcmd_spikes: (T, B) DCMD output spike count per bin
            lgmd_spikes: (T, B, 1, h, w) spatial spike map (for visualisation)
        """
        functional.reset_net(self)

        T, B, C, H, W = x.shape
        dcmd_out = []
        lgmd_out = []

        # Ring buffer for inhibition delay
        inh_buffer = [torch.zeros(B, 1, self.h, self.w, device=x.device)
                      for _ in range(self.inh_delay)]

        for t in range(T):
            frame = x[t]  # (B, 2, H, W)

            # P layer: pool
            p = self.pool(frame)  # (B, 2, h, w)

            # S layer: excitation from current frame
            exc = self.exc_conv(p)          # (B, 1, h, w)
            exc = F.relu(exc)               # rectify: only positive excitation

            # S layer: inhibition from delayed activity
            inh = self.inh_conv(inh_buffer[0])  # (B, 1, h, w), already negative

            # LGMD input: excitation + inhibition (inhibition is negative)
            lgmd_in = exc + inh             # (B, 1, h, w)

            # LGMD LIF: integrate and fire
            spikes = self.lgmd_lif(lgmd_in) # (B, 1, h, w), binary {0, 1}
            lgmd_out.append(spikes)

            # Update inhibition delay buffer (shift, add current excitation)
            inh_buffer.pop(0)
            inh_buffer.append(exc.detach())

            # DCMD: weighted spatial sum
            weighted = spikes * torch.abs(self.dcmd_weight)
            dcmd_spikes = weighted.sum(dim=(-1, -2, -3))  # (B,)
            dcmd_out.append(dcmd_spikes)

        dcmd_spikes = torch.stack(dcmd_out, dim=0)   # (T, B)
        lgmd_spikes = torch.stack(lgmd_out, dim=0)   # (T, B, 1, h, w)
        return dcmd_spikes, lgmd_spikes

    def collision_imminence(self, dcmd_spikes: torch.Tensor,
                            window: int = 5) -> torch.Tensor:
        """
        Smooth DCMD output with a causal moving average.

        Args:
            dcmd_spikes: (T, B) raw DCMD spike counts
            window:      smoothing window in time bins

        Returns:
            (T, B) smoothed collision-imminence signal in [0, 1]
        """
        T, B = dcmd_spikes.shape
        kernel = torch.ones(1, 1, window, device=dcmd_spikes.device) / window
        # (B, 1, T) for conv1d
        x = dcmd_spikes.T.unsqueeze(1)
        smoothed = F.conv1d(x, kernel, padding=window - 1)[..., :T]
        return smoothed.squeeze(1).T  # (T, B)
