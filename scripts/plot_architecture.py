"""
Publication-quality architecture figure for the LGMD+DCMD SNN.

Renders results/lgmd_architecture.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import gaussian_filter

# ── Colour palette ─────────────────────────────────────────────────────────────
C_ON    = "#EF5350"     # ON events — red
C_OFF   = "#42A5F5"     # OFF events — blue
C_EXC   = "#FF8F00"     # excitation — amber
C_INH   = "#5C6BC0"     # inhibition — indigo
C_LIF   = "#66BB6A"     # LIF/LGMD — green
C_DCMD  = "#AB47BC"     # DCMD — purple
C_BIO   = "#90A4AE"     # biological annotations — grey
C_BOX   = "#FAFAFA"
C_EDGE  = "#B0BEC5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

rng = np.random.default_rng(42)

# ── Synthetic illustrative data ────────────────────────────────────────────────
H, W = 40, 54   # display resolution for inset frames

def make_event_frame():
    """Simulate a looming event frame: expanding ring of ON events."""
    on  = np.zeros((H, W))
    off = np.zeros((H, W))
    # expanding ring (approaching object)
    cy, cx = H // 2, W // 2
    for r in [10, 11, 12]:   # thick expanding edge
        ys, xs = np.ogrid[:H, :W]
        mask = np.abs(np.sqrt((ys - cy)**2 + (xs - cx)**2) - r) < 1.5
        on[mask] = 1.0
    # some off-events from background grid lines
    for row in [8, 16, 24, 32]:
        off[row, 5:W-5] = rng.choice([0, 1], W-10, p=[0.85, 0.15]).astype(float)
    noise_on = rng.random((H, W)) < 0.015
    on = np.clip(on + noise_on * 0.5, 0, 1)
    return on, off

def pooled(frame, factor=4):
    """Simple block-average pooling."""
    H2, W2 = frame.shape[0] // factor, frame.shape[1] // factor
    return frame[:H2*factor, :W2*factor].reshape(H2, factor, W2, factor).mean(axis=(1, 3))

def excitation(on_p, off_p):
    """ON excites (+1), OFF weakly inhibits (-0.3), ReLU."""
    exc = on_p * 1.0 + off_p * (-0.3)
    return np.clip(exc, 0, None)

def inhibition_delayed(exc_prev):
    """Gaussian lateral spread of previous step's excitation."""
    return gaussian_filter(exc_prev, sigma=1.2) * 0.7

def lgmd_spikes(exc_cur, inh_prev):
    """Pixels where excitation > inhibition → fire."""
    net = exc_cur - inh_prev
    return (net > 0.08).astype(float)

# Build the illustrative data pipeline
on_frame, off_frame = make_event_frame()

# Use a slightly expanded ring as the "previous" frame for delay illustration
on_prev, _  = make_event_frame()

on_p  = pooled(on_frame,  factor=4)
off_p = pooled(off_frame, factor=4)
exc   = excitation(on_p, off_p)

on_p_prev = pooled(on_prev, factor=4)
off_p_prev = pooled(on_prev * 0, factor=4)
exc_prev   = excitation(on_p_prev * 0.8, off_p_prev)  # smaller ring

inh  = inhibition_delayed(exc_prev)
spks = lgmd_spikes(exc, inh)

# DCMD trace: flat background, rising during loom, threshold line
n_t = 80
t   = np.linspace(0, 8, n_t)
dcmd = np.zeros(n_t)
dcmd[30:] = 0.55 * (1 - np.exp(-(t[30:] - t[30]) / 1.2))
dcmd += rng.normal(0, 0.015, n_t)
dcmd = np.clip(dcmd, 0, None)
smooth_k = np.ones(7) / 7
dcmd_sm = np.convolve(dcmd, smooth_k, mode='same')

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 8.5))

# Main pipeline insets: 6 small axes across the top + 1 DCMD trace bottom-right
# We'll use manual axes placement
inset_y   = 0.50   # top of inset axes (normalised)
inset_h   = 0.30
inset_w   = 0.095
inset_xs  = [0.035, 0.175, 0.315, 0.455, 0.595, 0.735]
dcmd_ax_rect = [0.835, 0.42, 0.14, 0.40]

axes = [fig.add_axes([x, inset_y, inset_w, inset_h]) for x in inset_xs]
ax_dcmd = fig.add_axes(dcmd_ax_rect)

# ── Helper: draw a stylised event frame ───────────────────────────────────────
def draw_event_frame(ax, on, off, title="", bio=""):
    bg = np.zeros((on.shape[0], on.shape[1], 3))
    bg[..., 0] += on  * 0.9
    bg[..., 2] += on  * 0.2   # red-ish ON
    bg[..., 2] += off * 0.9
    bg[..., 0] += off * 0.1   # blue-ish OFF
    bg = np.clip(bg, 0, 1)
    ax.imshow(np.ones_like(bg) * 0.06, cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax.imshow(bg, aspect='auto', alpha=0.95)
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(C_EDGE); sp.set_linewidth(1.2)
    if bio:
        ax.set_xlabel(bio, fontsize=7.5, color=C_BIO, labelpad=3)

def draw_heatmap(ax, data, cmap, title="", bio="", vmin=None, vmax=None):
    ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
              interpolation='nearest')
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(C_EDGE); sp.set_linewidth(1.2)
    if bio:
        ax.set_xlabel(bio, fontsize=7.5, color=C_BIO, labelpad=3)

# Stage 0: Event frame (ON/OFF)
draw_event_frame(axes[0], on_frame, off_frame,
                 title="1 · Event Frame\n(T, 2, H, W)",
                 bio="Photoreceptors\nΔlog(I) → ON / OFF")

# Stage 1: P layer (pooled)
draw_heatmap(axes[1], on_p, cmap='hot',
             title="2 · P Layer\nAvgPool2d ÷4",
             bio="P cells (medulla)\nSpatial integration",
             vmin=0, vmax=1)

# Stage 2: Excitation
draw_heatmap(axes[2], exc, cmap='YlOrBr',
             title="3 · S-Excitation\nON×1 + OFF×(−0.3), ReLU",
             bio="S cells\nDirect excitation",
             vmin=0, vmax=1)

# Stage 3: Lateral inhibition (delayed)
draw_heatmap(axes[3], inh, cmap='Blues',
             title="4 · I-Inhibition\n7×7 Gauss, 1-step delay",
             bio="I cells (afferents)\nLateral suppression",
             vmin=0, vmax=inh.max() + 0.01)

# Stage 4: LGMD spikes
spk_rgb = np.zeros((*spks.shape, 3))
spk_rgb[spks > 0] = [0.3, 0.85, 0.4]  # green spikes
bg_arr = np.ones((*spks.shape, 3)) * 0.07
display = bg_arr.copy()
display[spks > 0] = spk_rgb[spks > 0]
axes[4].imshow(np.ones((*spks.shape, 3)) * 0.07, aspect='auto')
axes[4].imshow(display, aspect='auto')
axes[4].set_title("5 · LGMD LIF\nLIF v_thr=0.5, τ=2ms", fontsize=8.5,
                  fontweight='bold', pad=3)
axes[4].set_xticks([]); axes[4].set_yticks([])
axes[4].set_xlabel("LGMD neurons\nSpike map (H×W)", fontsize=7.5,
                   color=C_BIO, labelpad=3)
for sp in axes[4].spines.values():
    sp.set_edgecolor(C_EDGE); sp.set_linewidth(1.2)

# Stage 5: DCMD aggregation icon (uniform weights grid)
dcmd_icon = np.ones((on_p.shape[0], on_p.shape[1])) * 0.5
axes[5].imshow(dcmd_icon, cmap='Purples', aspect='auto', vmin=0, vmax=1)
axes[5].text(0.5, 0.5, "Σ / (H·W)", transform=axes[5].transAxes,
             ha='center', va='center', fontsize=10, color='white',
             fontweight='bold')
axes[5].set_title("6 · DCMD Readout\nUniform spatial avg", fontsize=8.5,
                  fontweight='bold', pad=3)
axes[5].set_xticks([]); axes[5].set_yticks([])
axes[5].set_xlabel("DCMD axon\nCollision imminence", fontsize=7.5,
                   color=C_BIO, labelpad=3)
for sp in axes[5].spines.values():
    sp.set_edgecolor(C_EDGE); sp.set_linewidth(1.2)

# ── DCMD time-series ───────────────────────────────────────────────────────────
ax_dcmd.plot(t, dcmd,    color=C_DCMD, alpha=0.3, linewidth=0.8)
ax_dcmd.plot(t, dcmd_sm, color=C_DCMD, linewidth=2.2, label="DCMD (smoothed)")
ax_dcmd.axhline(0.25, color="#E53935", linewidth=1.2, linestyle="--",
                label="Threshold (0.25)")
ax_dcmd.axvline(t[30], color="#37474F", linewidth=1.0, linestyle=":",
                alpha=0.7, label="Obstacle launch")
evasion_t = t[np.argmax(dcmd_sm > 0.25)]
ax_dcmd.axvline(evasion_t, color="#FF6F00", linewidth=1.5,
                label="Evasion trigger")
ax_dcmd.fill_between(t, dcmd_sm, 0.25,
                     where=dcmd_sm > 0.25, alpha=0.15, color="#FF6F00")
ax_dcmd.set_xlim(t[0], t[-1])
ax_dcmd.set_ylim(-0.02, 0.55)
ax_dcmd.set_xlabel("Time (s)", fontsize=8.5)
ax_dcmd.set_ylabel("DCMD spike rate", fontsize=8.5, color=C_DCMD)
ax_dcmd.set_title("DCMD Output Signal", fontsize=9, fontweight='bold')
ax_dcmd.legend(fontsize=7.5, loc='upper left')
ax_dcmd.spines['top'].set_visible(False)
ax_dcmd.spines['right'].set_visible(False)
ax_dcmd.grid(True, alpha=0.25)
ax_dcmd.tick_params(labelsize=7.5)

# ── Pipeline arrows (figure-level) ────────────────────────────────────────────
arrow_kw = dict(arrowstyle="-|>", color="#607D8B", linewidth=1.5,
                mutation_scale=12, transform=fig.transFigure, clip_on=False)
arrow_y = inset_y + inset_h * 0.5

for i in range(5):
    x0 = inset_xs[i] + inset_w + 0.003
    x1 = inset_xs[i+1] - 0.003
    arr = FancyArrowPatch((x0, arrow_y), (x1, arrow_y),
                          **arrow_kw)
    fig.add_artist(arr)

# Arrow from stage 6 to DCMD trace
x0 = inset_xs[5] + inset_w + 0.003
x1 = dcmd_ax_rect[0] - 0.003
arr = FancyArrowPatch((x0, arrow_y), (x1, arrow_y), **arrow_kw)
fig.add_artist(arr)

# ── Box labels and block descriptions ─────────────────────────────────────────
# Drawn as text boxes below the pipeline axes
block_desc = [
    ("Event Camera\n(DAVIS346 346×260)",
     "Δlog(I) > θ → ON spike\nΔlog(I) < −θ → OFF spike\ndt = 10 ms bins"),
    ("P Layer\n(Photoreceptor pool)",
     "AvgPool2d(4)\n87×65 output\n16× fewer elements"),
    ("S-Excitation\n(3×3 Conv + ReLU)",
     "ON channel: ×+1.0\nOFF channel: ×−0.3\nReLU → non-negative"),
    ("I-Inhibition\n(7×7 Gauss Conv)",
     "Gaussian σ=1.5, fixed\nShift: 1 time step\nSpreads, doesn't learn"),
    ("LGMD LIF\n(per-pixel neuron)",
     "Integrate exc + inh\nv_threshold = 0.5\nτ_mem = 2 ms"),
    ("DCMD Readout\n(uniform avg)",
     "Σ spikes / (H×W)\nFixed buffer (not trained)\nSpatially invariant"),
]

desc_y   = 0.38
desc_h   = 0.10
for i, (xs_i, (title, desc)) in enumerate(zip(inset_xs, block_desc)):
    ax_d = fig.add_axes([xs_i, desc_y, inset_w, desc_h])
    ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1)
    ax_d.axis('off')
    ax_d.text(0.5, 0.95, title, ha='center', va='top', fontsize=7.5,
              fontweight='bold', color="#263238")
    ax_d.text(0.5, 0.55, desc, ha='center', va='top', fontsize=6.8,
              color="#455A64", linespacing=1.4)

# ── Delay mechanism diagram ────────────────────────────────────────────────────
# Bottom-left: timing diagram showing why delay → looming selectivity
ax_delay = fig.add_axes([0.035, 0.04, 0.55, 0.26])
ax_delay.set_xlim(0, 10)
ax_delay.set_ylim(-0.5, 3.5)
ax_delay.axis('off')

ax_delay.set_title("Why 1-step inhibition delay → looming selectivity",
                   fontsize=9, fontweight='bold', loc='left', pad=4)

# Translating object: excitation shifts, inhibition follows exactly → cancels
def draw_timing_row(ax, y, exc_xs, inh_xs, label, color_e=C_EXC, color_i=C_INH):
    for x in exc_xs:
        ax.add_patch(mpatches.FancyBboxPatch((x, y + 0.05), 0.35, 0.55,
                     boxstyle="round,pad=0.05", fc=color_e, ec='none', alpha=0.8))
    for x in inh_xs:
        ax.add_patch(mpatches.FancyBboxPatch((x, y + 0.05), 0.35, 0.55,
                     boxstyle="round,pad=0.05", fc=color_i, ec='none', alpha=0.55))
    ax.text(-0.15, y + 0.35, label, ha='right', va='center', fontsize=7.5,
            color='#263238')

# Time axis labels
for t_i, label in [(1, "t−2"), (2.5, "t−1"), (4, "t"), (5.5, "t+1"), (7, "t+2")]:
    ax_delay.text(t_i + 0.17, 3.25, label, ha='center', va='bottom',
                  fontsize=7.5, color='#546E7A')
    ax_delay.axvline(t_i, ymin=0.05, ymax=0.92, color='#B0BEC5',
                     linewidth=0.6, linestyle=':', alpha=0.7)

# Translating object rows
ax_delay.text(4.5, 3.0, "Translating stimulus", ha='center', fontsize=8,
              color='#B71C1C', fontweight='bold')
draw_timing_row(ax_delay, 2.3,
    exc_xs=[1, 2.5, 4, 5.5],    # excitation moves right each step
    inh_xs=[1, 2.5, 4, 5.5],    # inhibition follows exactly (same position, 1 step later)
    label="Excitation\n(px = motion)")
draw_timing_row(ax_delay, 1.55,
    exc_xs=[1, 2.5, 4, 5.5],
    inh_xs=[1, 2.5, 4, 5.5],
    label="Inhibition\n(px = same)")
ax_delay.text(8.2, 2.0, "→ LGMD silent\n(exc ≈ inh)", ha='left',
              va='center', fontsize=7.5, color=C_INH, fontweight='bold')

# Looming object rows
ax_delay.text(4.5, 1.15, "Looming stimulus", ha='center', fontsize=8,
              color='#1B5E20', fontweight='bold')
draw_timing_row(ax_delay, 0.6,
    exc_xs=[1, 2.2, 3.5, 4.7],         # expanding: each step edge moves outward
    inh_xs=[1.6, 2.8, 4.1, 5.3],       # inhibition = delayed smaller ring
    label="Excitation\n(expanding edge)")
draw_timing_row(ax_delay, -0.2,
    exc_xs=[1, 2.2, 3.5, 4.7],
    inh_xs=[0.4, 1.6, 2.8, 4.1],       # inh lags behind the expanding edge
    label="Inhibition\n(1 step earlier,\nsmaller ring)")
ax_delay.text(8.2, 0.3, "→ LGMD fires!\n(exc > inh at edge)", ha='left',
              va='center', fontsize=7.5, color=C_LIF, fontweight='bold')

# Legend
leg_x = 9.0; leg_y = 3.2
ax_delay.add_patch(mpatches.FancyBboxPatch((leg_x, leg_y - 0.1), 0.35, 0.4,
                   boxstyle="round,pad=0.05", fc=C_EXC, ec='none', alpha=0.8))
ax_delay.text(leg_x + 0.5, leg_y + 0.1, "Excitation (S)", va='center',
              fontsize=7.5, color=C_EXC)
ax_delay.add_patch(mpatches.FancyBboxPatch((leg_x, leg_y - 0.65), 0.35, 0.4,
                   boxstyle="round,pad=0.05", fc=C_INH, ec='none', alpha=0.55))
ax_delay.text(leg_x + 0.5, leg_y - 0.45, "Inhibition (I)", va='center',
              fontsize=7.5, color=C_INH)

# ── Training signal annotation ─────────────────────────────────────────────────
ax_train = fig.add_axes([0.60, 0.04, 0.37, 0.26])
ax_train.axis('off')
ax_train.set_title("Training supervision", fontsize=9, fontweight='bold',
                   loc='left', pad=4)
lines = [
    ("Label:",           "Analytical dθ/dt = 2r·v / (d²+r²)  from trajectory"),
    ("",                 "Thresholded → binary loom / no-loom"),
    ("Loss:",            "BCE on DCMD output + weighted net_exc auxiliary"),
    ("",                 "Looming windows oversampled 5× (WeightedRandomSampler)"),
    ("Augmentation:",    "Horiz. flip, vert. flip, polarity swap,"),
    ("",                 "event noise, dropout — all on GPU"),
    ("Optimiser:",       "Adam + cosine LR schedule + gradient clipping"),
    ("Surrogate grad:",  "Rectangular (SpikingJelly default) through LIF threshold"),
]
for row_i, (key, val) in enumerate(lines):
    y = 0.94 - row_i * 0.125
    ax_train.text(0.0, y, key, ha='left', va='top', fontsize=7.8,
                  fontweight='bold', color="#263238", transform=ax_train.transAxes)
    ax_train.text(0.22, y, val, ha='left', va='top', fontsize=7.5,
                  color="#455A64", transform=ax_train.transAxes)

# ── Figure title ───────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, "LGMD+DCMD Spiking Neural Network — Architecture Overview",
         ha='center', va='top', fontsize=13, fontweight='bold', color='#1A237E')
fig.text(0.5, 0.945,
         "Biological circuit: Locust Lobula Giant Movement Detector (LGMD) → Descending Contralateral Movement Detector (DCMD)",
         ha='center', va='top', fontsize=9, color='#546E7A')

from pathlib import Path
Path("results").mkdir(exist_ok=True)
fig.savefig("results/lgmd_architecture.png", dpi=300, bbox_inches='tight',
            facecolor='white')
print("Saved → results/lgmd_architecture.png")
