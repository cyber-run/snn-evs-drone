"""
LGMD-inspired spiking neural network for looming detection.

Architecture mirrors the locust Lobula Giant Movement Detector circuit:
  P layer  -> photoreceptor-like spatial pooling (no spikes)
  S layer  -> LIF excitation + delayed lateral inhibition
  LGMD     -> integration of (excitation - inhibition)
  DCMD     -> global spatial sum -> collision-imminence spike rate

Input:  (T, B, 2, H, W)  -- T time bins, batch B, 2 polarities, H x W pixels
Output: (T, B)            -- DCMD spike count per time bin per sample

References:
  Rind & Bramwell (1996) -- original computational LGMD model
  Stafford et al. (2007) -- LGMD robot implementation
  Meng et al. (2023)     -- SNN LGMD for UAV avoidance
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Build a 2-D Gaussian kernel for lateral inhibition initialisation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


class LGMDNet(nn.Module):
    """
    LGMD spiking network.

    Args:
        height:          input frame height (after any pre-resize)
        width:           input frame width
        pool_factor:     spatial pooling divisor applied before S layer
        tau_mem:         LIF membrane time constant
        inh_kernel_size: size of lateral inhibition conv kernel (must be odd)
        inh_sigma:       Gaussian sigma for inhibition kernel initialisation
        inh_delay:       time steps to delay inhibition (1 = biological default)
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

        # P layer: spatial pooling (not spiking)
        self.pool = nn.AvgPool2d(pool_factor)

        # S layer: excitation -- ON drives positive, OFF drives weak negative
        self.exc_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.exc_conv.weight[:, 0:1],  1.0)   # ON
        nn.init.constant_(self.exc_conv.weight[:, 1:2], -0.3)   # OFF

        # S layer: lateral inhibition -- Gaussian spread, one step delayed
        self.inh_conv = nn.Conv2d(1, 1, kernel_size=inh_kernel_size,
                                  padding=inh_kernel_size // 2, bias=False)
        with torch.no_grad():
            k = _gaussian_kernel(inh_kernel_size, inh_sigma)
            self.inh_conv.weight.data = -k.view(1, 1, inh_kernel_size, inh_kernel_size)

        # LGMD LIF neurons -- multi-step mode processes all T steps in one call
        self.lgmd_lif = neuron.LIFNode(tau=tau_mem, v_threshold=0.5, detach_reset=True,
                                       step_mode='m')

        # DCMD readout: fixed uniform spatial average (not learnable).
        # Learnable weights overfit to specific pixel locations; a uniform
        # average forces the network to learn spatially-invariant features.
        self.register_buffer(
            "dcmd_weight",
            torch.ones(1, 1, self.h, self.w) / (self.h * self.w),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (T, B, 2, H, W) float tensor, values in [0, 1]

        Returns:
            dcmd_spikes: (T, B)
            lgmd_spikes: (T, B, 1, h, w)
        """
        functional.reset_net(self)

        T, B, C, H, W = x.shape

        # Vectorize P + S-excitation across all T frames in a single GPU call
        x_2d = x.reshape(T * B, C, H, W)
        p    = self.pool(x_2d)                          # (T*B, 2, h, w)
        exc  = F.relu(self.exc_conv(p))                 # (T*B, 1, h, w)
        exc  = exc.reshape(T, B, 1, self.h, self.w)

        # Delayed inhibition: shift exc by inh_delay steps, pad start with zeros.
        # Detach to match original behaviour (no gradient through delay buffer).
        pad         = exc.new_zeros(self.inh_delay, B, 1, self.h, self.w)
        exc_delayed = torch.cat([pad, exc[:-self.inh_delay].detach()], dim=0)
        inh = self.inh_conv(
            exc_delayed.reshape(T * B, 1, self.h, self.w)
        ).reshape(T, B, 1, self.h, self.w)

        # LIF in multi-step mode: (T, B, 1, h, w) -> (T, B, 1, h, w)
        lgmd_in = exc + inh
        spikes  = self.lgmd_lif(lgmd_in)

        # DCMD: learned spatial weighting then global sum
        dcmd = (spikes * self.dcmd_weight).sum(dim=(-1, -2, -3))  # (T, B)

        # Return mean RECTIFIED post-inhibition excitation (T, B) for training.
        # Rectified lgmd_in = max(0, exc - background_inhibition).
        # This is positive only where looming edges survive the lateral suppression,
        # giving a sparse but correctly-signed training signal.
        net_exc = lgmd_in.clamp(min=0).mean(dim=(-1, -2, -3))   # (T, B)
        return dcmd, spikes, net_exc

    def collision_imminence(self, dcmd_spikes: torch.Tensor,
                            window: int = 5) -> torch.Tensor:
        """
        Smooth DCMD output with a causal moving average.

        Args:
            dcmd_spikes: (T, B)
            window:      smoothing window in time bins

        Returns:
            (T, B) smoothed collision-imminence signal
        """
        T, B = dcmd_spikes.shape
        kernel = torch.ones(1, 1, window, device=dcmd_spikes.device) / window
        x = dcmd_spikes.T.unsqueeze(1)
        smoothed = F.conv1d(x, kernel, padding=window - 1)[..., :T]
        return smoothed.squeeze(1).T
