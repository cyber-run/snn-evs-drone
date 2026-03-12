"""
Publication-quality training visualisation for LGMD SNN.

Produces two figures:
  1. Training dashboard (3-panel): loss curves, correlation metrics,
     loom vs background discrimination over training
  2. DCMD temporal response: signal over a full recording with launch marker

Usage:
    # After training (requires results/lgmd_pearson_sw.csv):
    python scripts/plot_training.py \
        --csv  results/lgmd_pearson_sw.csv \
        --h5   data/evasion_head_on_events/events.h5 \
        --weights results/lgmd_pearson_sw.pt \
        --out  results/

    # Training curves only (no H5 required):
    python scripts/plot_training.py --csv results/lgmd_pearson_sw.csv

Industry context:
  - TensorBoard / W&B for live monitoring during training runs
  - Matplotlib + seaborn for publication figures (ICRA, CoRL, RA-L)
  - Key metrics: ExCorr (model learns label rank), discrimination ratio (Ex_loom/Ex_bg),
    and temporal DCMD alignment with dθ/dt for the contribution figure
"""

import argparse
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Style ─────────────────────────────────────────────────────────────────────

COLORS = {
    "tr_loss":  "#2196F3",   # blue
    "va_loss":  "#F44336",   # red
    "excorr":   "#4CAF50",   # green
    "dccorr":   "#FF9800",   # orange
    "loom":     "#E91E63",   # pink
    "bg":       "#607D8B",   # blue-grey
    "ratio":    "#9C27B0",   # purple
}

def _setup_style():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "legend.fontsize":    8.5,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linewidth":     0.6,
        "lines.linewidth":    1.8,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> dict:
    """Parse training metrics CSV into numpy arrays."""
    import csv
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    def _col(key, dtype=float):
        vals = []
        for r in rows:
            v = r.get(key, "").strip()
            try:
                vals.append(dtype(v))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        return np.array(vals, dtype=np.float64)

    return {
        "epoch":   _col("epoch", int),
        "tr_loss": _col("tr_loss"),
        "va_loss": _col("va_loss"),
        "ex_corr": _col("ex_corr"),
        "dc_corr": _col("dc_corr"),
        "ex_loom": _col("ex_loom"),
        "ex_bg":   _col("ex_bg"),
        "acc":     _col("acc"),
        "lr":      _col("lr"),
    }


# ── Figure 1: Training dashboard ──────────────────────────────────────────────

def plot_training_dashboard(metrics: dict, out_path: str,
                             loss_name: str = "Loss") -> None:
    """
    3-panel training dashboard:
      (a) Train + val loss
      (b) ExCorr + DcCorr over epochs
      (c) Ex_loom / Ex_bg ratio — the key selectivity metric
    """
    _setup_style()
    ep = metrics["epoch"]
    has_val = not np.all(np.isnan(metrics["va_loss"]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.subplots_adjust(wspace=0.35)

    # ── (a) Loss ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(ep, metrics["tr_loss"], color=COLORS["tr_loss"], label="Train loss")
    if has_val:
        ax.plot(ep, metrics["va_loss"], color=COLORS["va_loss"],
                label="Val loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_name)
    ax.set_title("(a) Loss")
    ax.legend()
    ax.set_xlim(left=0)

    # ── (b) Correlation ──────────────────────────────────────────────────────
    ax = axes[1]
    valid = ~np.isnan(metrics["ex_corr"])
    if valid.any():
        ax.plot(ep[valid], metrics["ex_corr"][valid],
                color=COLORS["excorr"], label="ExCorr (net_exc)")
        ax.plot(ep[valid], metrics["dc_corr"][valid],
                color=COLORS["dccorr"], label="DcCorr (DCMD)",
                linestyle="--")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        # Annotate best
        best_idx = np.nanargmax(metrics["ex_corr"][valid])
        best_ep  = ep[valid][best_idx]
        best_val = metrics["ex_corr"][valid][best_idx]
        ax.axvline(best_ep, color=COLORS["excorr"], linewidth=0.8,
                   linestyle=":", alpha=0.7)
        ax.annotate(f"peak {best_val:+.3f}\n@ ep {best_ep}",
                    xy=(best_ep, best_val),
                    xytext=(best_ep + max(ep) * 0.05, best_val - 0.05),
                    fontsize=7.5, color=COLORS["excorr"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["excorr"],
                                   lw=0.8))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pearson r")
    ax.set_title("(b) Correlation with dθ/dt")
    ax.legend()
    ax.set_xlim(left=0)

    # ── (c) Loom vs Background discrimination ────────────────────────────────
    ax = axes[2]
    valid = ~np.isnan(metrics["ex_loom"])
    if valid.any():
        ax.plot(ep[valid], metrics["ex_loom"][valid],
                color=COLORS["loom"], label="Loom windows")
        ax.plot(ep[valid], metrics["ex_bg"][valid],
                color=COLORS["bg"],  label="Background", linestyle="--")
        ax.set_ylabel("Mean net_exc")
        ax.set_title("(c) LGMD selectivity (loom vs bg)")

        # Add ratio on right axis
        ratio = (metrics["ex_loom"][valid]
                 / (metrics["ex_bg"][valid] + 1e-9))
        ax2 = ax.twinx()
        ax2.plot(ep[valid], ratio, color=COLORS["ratio"],
                 linewidth=1.2, alpha=0.6, linestyle=":")
        ax2.set_ylabel("Loom / Bg ratio", color=COLORS["ratio"])
        ax2.tick_params(axis="y", labelcolor=COLORS["ratio"])
        ax2.spines["top"].set_visible(False)

        # Mark final ratio
        final_ratio = ratio[-1]
        ax2.annotate(f"final {final_ratio:.1f}×",
                     xy=(ep[valid][-1], final_ratio),
                     xytext=(ep[valid][-1] - max(ep) * 0.15, final_ratio + 0.3),
                     fontsize=7.5, color=COLORS["ratio"])

    ax.set_xlabel("Epoch")
    ax.legend(loc="upper left")
    ax.set_xlim(left=0)

    fig.suptitle("LGMD SNN Training — Phase 1 Validation", fontsize=12, y=1.01)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Training dashboard saved → {out_path}")


# ── Figure 2: Improved DCMD temporal response ─────────────────────────────────

def plot_dcmd_response(h5_path: str, weights_path: str, out_path: str,
                       dt_us: float = 10_000.0, n_bins: int = 20,
                       pool: int = 4, dcmd_threshold: float = 0.3,
                       t_window: tuple | None = None) -> None:
    """
    Publication-quality DCMD signal plot cropped to the looming window.

    t_window: (t_start, t_end) in seconds to crop x-axis.  If None, auto-crops
    to ±2s around launch.
    """
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.eval_dcmd import eval_recording

    result = eval_recording(h5_path, weights_path,
                            dt_us=dt_us, n_bins=n_bins, pool=pool,
                            smooth_window=5)

    t   = result["t_s"]
    er  = result["event_rate"]
    lbl = result["label"]
    dc  = result["dcmd"]
    dcs = result["dcmd_smooth"]
    lt  = result["launch_t_s"]

    # Auto-crop window: 1.5s before launch to 3.5s after
    if t_window is None and lt is not None:
        t_window = (max(t[0], lt - 1.5), min(t[-1], lt + 3.5))
    if t_window is not None:
        mask = (t >= t_window[0]) & (t <= t_window[1])
        t, er, lbl, dc, dcs = t[mask], er[mask], lbl[mask], dc[mask], dcs[mask]

    _setup_style()
    fig, (ax_ev, ax_sig) = plt.subplots(2, 1, figsize=(8, 4.5),
                                        sharex=True,
                                        gridspec_kw={"height_ratios": [1, 2]})
    fig.subplots_adjust(hspace=0.08)

    # Top: event rate
    ax_ev.fill_between(t, er, color="silver", alpha=0.8, linewidth=0)
    ax_ev.set_ylabel("Event rate\n(norm.)", fontsize=9)
    ax_ev.set_ylim(0, 1.1)
    ax_ev.set_yticks([0, 0.5, 1.0])
    if lt is not None:
        ax_ev.axvline(lt, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax_ev.set_title("LGMD DCMD Response — head-on looming stimulus", fontsize=11)

    # Bottom: label + DCMD
    ax_sig.plot(t, lbl, color="#2196F3", linewidth=2.0,
                label="dθ/dt label (ground truth)", zorder=3)
    ax_sig.plot(t, dc,  color="#FFCCBC", linewidth=0.8, alpha=0.6, zorder=2)
    ax_sig.plot(t, dcs, color="#F44336", linewidth=2.2,
                label="DCMD output (smoothed)", zorder=4)
    ax_sig.axhline(dcmd_threshold, color="#F44336", linewidth=1.0,
                   linestyle=":", alpha=0.8,
                   label=f"Threshold ({dcmd_threshold:.1f})")
    if lt is not None:
        ax_sig.axvline(lt, color="black", linewidth=1.2, linestyle="--",
                       alpha=0.6, label=f"Launch (t={lt:.1f}s)")

    # Shade looming region
    loom_mask = lbl > 0.3
    if loom_mask.any():
        ax_sig.fill_between(t, 0, 1, where=loom_mask, alpha=0.08,
                            color="#4CAF50", label="Looming region (dθ/dt>0.3)")

    ax_sig.set_xlabel("Time (s)")
    ax_sig.set_ylabel("Normalised amplitude")
    ax_sig.set_ylim(-0.05, 1.1)
    ax_sig.legend(loc="upper left", fontsize=8, ncol=2)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"DCMD response saved → {out_path}")


# ── Figure 4: Evasion result annotated DCMD ───────────────────────────────────

def plot_evasion_result(h5_path: str, weights_path: str, out_path: str,
                        evasion_t: float, evasion_dcmd: float,
                        closest_miss: float, closest_hit: float,
                        dt_us: float = 10_000.0, n_bins: int = 20,
                        pool: int = 4, dcmd_threshold: float = 0.25) -> None:
    """
    Hero figure: DCMD trace with evasion trigger annotated.
    Shows the closed-loop result — when the SNN fires and what happens.
    """
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.eval_dcmd import eval_recording

    result = eval_recording(h5_path, weights_path,
                            dt_us=dt_us, n_bins=n_bins, pool=pool,
                            smooth_window=5)

    t   = result["t_s"]
    lbl = result["label"]
    dcs = result["dcmd_smooth"]
    lt  = result["launch_t_s"]

    # Crop to looming window
    if lt is not None:
        mask = (t >= lt - 0.5) & (t <= lt + 3.0)
        t, lbl, dcs = t[mask], lbl[mask], dcs[mask]

    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.plot(t, lbl, color="#2196F3", linewidth=2.0, label="dθ/dt (ground truth)", zorder=3)
    ax.plot(t, dcs, color="#F44336", linewidth=2.2, label="DCMD output", zorder=4)
    ax.axhline(dcmd_threshold, color="#F44336", linewidth=1.0, linestyle=":",
               alpha=0.7, label=f"Threshold ({dcmd_threshold})")

    if lt is not None:
        ax.axvline(lt, color="black", linewidth=1.2, linestyle="--",
                   alpha=0.5, label=f"Obstacle launch")

    # Evasion trigger annotation
    ax.axvline(evasion_t, color="#FF6F00", linewidth=2.0, linestyle="-",
               label=f"Evasion triggered (t={evasion_t:.2f}s)", zorder=5)
    ax.annotate(f"SNN fires\nDCMD={evasion_dcmd:.2f}",
                xy=(evasion_t, dcmd_threshold),
                xytext=(evasion_t + 0.15, dcmd_threshold + 0.15),
                fontsize=9, color="#FF6F00", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#FF6F00", lw=1.5))

    # Outcome annotation
    ax.text(0.98, 0.95,
            f"MISS  —  closest approach {closest_miss:.2f}m\n"
            f"(no evasion: {closest_hit:.2f}m)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#2E7D32",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                      edgecolor="#4CAF50", linewidth=1.2))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised amplitude")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("LGMD-SNN Reactive Evasion — head-on looming stimulus", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Evasion result figure saved → {out_path}")


# ── Figure 3: Loom vs Background bar chart ────────────────────────────────────

def plot_discrimination_bar(metrics: dict, out_path: str) -> None:
    """
    Final epoch loom vs background mean excitation — bar chart comparing
    multiple runs or epochs.  Shows the key selectivity result clearly.
    """
    _setup_style()

    # Use last 10% of epochs as "converged" estimate
    n = len(metrics["epoch"])
    last = max(1, n // 10)
    valid = ~np.isnan(metrics["ex_loom"])
    if not valid.any():
        print("No discrimination data — skipping bar chart")
        return

    ep_loom = metrics["ex_loom"][valid]
    ep_bg   = metrics["ex_bg"][valid]
    ep_num  = metrics["epoch"][valid]

    # Final converged values
    loom_mean = ep_loom[-last:].mean()
    loom_std  = ep_loom[-last:].std()
    bg_mean   = ep_bg[-last:].mean()
    bg_std    = ep_bg[-last:].std()

    ratio = loom_mean / (bg_mean + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.subplots_adjust(wspace=0.4)

    # Bar chart
    ax = axes[0]
    bars = ax.bar(["Looming", "Background"], [loom_mean, bg_mean],
                  yerr=[loom_std, bg_std],
                  color=[COLORS["loom"], COLORS["bg"]],
                  capsize=5, width=0.5, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Mean net_exc (LGMD input)")
    ax.set_title(f"Loom vs Background\n(converged, ratio = {ratio:.1f}×)")
    ax.set_ylim(bottom=0)
    for bar, val, err in zip(bars, [loom_mean, bg_mean], [loom_std, bg_std]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + err + loom_mean * 0.03,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8.5)

    # Ratio over training
    ax2 = axes[1]
    ratio_curve = ep_loom / (ep_bg + 1e-9)
    ax2.plot(ep_num, ratio_curve, color=COLORS["ratio"], linewidth=1.8)
    ax2.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
                label="Ratio = 1 (no selectivity)")
    ax2.fill_between(ep_num, 1.0, ratio_curve,
                     where=(ratio_curve > 1.0), alpha=0.15, color=COLORS["ratio"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loom / Background ratio")
    ax2.set_title("LGMD selectivity over training")
    ax2.legend(fontsize=8)
    ax2.set_xlim(left=0)

    fig.suptitle("LGMD Neuron Selectivity for Looming Stimuli", fontsize=11, y=1.01)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Discrimination chart saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     required=True,
                   help="Training metrics CSV (output of train_lgmd.py)")
    p.add_argument("--h5",      default=None,
                   help="H5 recording for DCMD temporal plot")
    p.add_argument("--weights", default=None,
                   help="Trained model weights for DCMD temporal plot")
    p.add_argument("--out",     default="results/",
                   help="Output directory (default: results/)")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="DCMD threshold line on signal plot")
    args = p.parse_args()

    out = Path(args.out)
    metrics = load_csv(args.csv)
    stem = Path(args.csv).stem

    plot_training_dashboard(metrics,
                            str(out / f"{stem}_training.png"))
    plot_discrimination_bar(metrics,
                            str(out / f"{stem}_discrimination.png"))

    if args.h5 and args.weights:
        plot_dcmd_response(args.h5, args.weights,
                           str(out / f"{stem}_dcmd.png"),
                           dcmd_threshold=args.threshold)
