"""
Side-by-side comparison video: baseline (no evasion) vs LGMD-SNN evasion.

For a given profile, stitches the two frame directories into a single
horizontally-stacked MP4 with text annotations and a closest-approach
overlay drawn from the saved meta.npz trajectories.

Usage:
    python scripts/make_comparison_video.py \
        --profile head_on \
        --baseline_name baseline_v2 \
        --evasion_name  burst_evasion \
        --out results/comparison_head_on.mp4

    # Batch — all profiles (assumes names follow convention):
    python scripts/make_comparison_video.py --all
"""

import argparse
import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path


FPS     = 120   # simulation capture framerate
OUT_FPS = 30    # output video framerate (4× slowdown relative to sim)


def load_meta(name: str) -> dict | None:
    meta_path = f"/tmp/evasion_{name}_meta/meta.npz"
    if not os.path.exists(meta_path):
        print(f"[WARN] meta not found: {meta_path}")
        return None
    m = np.load(meta_path)
    dp = m["drone_positions"]
    op = m["obstacle_positions"]
    dists = np.linalg.norm(dp - op, axis=1)
    return {
        "drone_pos":  dp,
        "obs_pos":    op,
        "dists":      dists,
        "closest":    float(dists.min()),
        "closest_step": int(dists.argmin()),
        "drone_max_z": float(dp[:, 2].max()),
    }


def annotate_frames(frame_dir: str, out_dir: str, label: str,
                    meta: dict | None, evasion: bool,
                    evasion_step: int | None = None) -> int:
    """
    Copy frames to out_dir with cv2 text overlays.
    Returns number of frames written.
    """
    import cv2

    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".bmp"))
    if not frames:
        return 0

    os.makedirs(out_dir, exist_ok=True)

    closest = meta["closest"] if meta else None
    verdict = None
    if meta and evasion:
        verdict = "MISS" if closest > 0.6 else "HIT"

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    thickness  = 1
    pad        = 8

    for i, fname in enumerate(frames):
        img = cv2.imread(os.path.join(frame_dir, fname))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Top-left label
        cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(img, label, (pad, 20), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        # Evasion trigger marker
        if evasion_step is not None and i == evasion_step:
            cv2.rectangle(img, (0, 0), (w, h), (0, 165, 255), 4)
            cv2.putText(img, "EVASION", (w // 2 - 50, h // 2),
                        font, 1.2, (0, 165, 255), 2, cv2.LINE_AA)

        # Bottom: closest approach distance (live)
        if meta is not None:
            step = min(i, len(meta["dists"]) - 1)
            dist = meta["dists"][step]
            bar_color = (0, 200, 80) if dist > 0.6 else (0, 80, 220)
            cv2.rectangle(img, (0, h - 28), (w, h), (20, 20, 20), -1)
            dist_text = f"dist {dist:.2f}m"
            if verdict and i >= meta["closest_step"]:
                dist_text += f"  [{verdict}]"
            cv2.putText(img, dist_text, (pad, h - 8), font, font_scale,
                        bar_color, thickness, cv2.LINE_AA)

        out_path = os.path.join(out_dir, f"frame_{i:06d}.png")
        cv2.imwrite(out_path, img)

    return len(frames)


def make_side_by_side(left_dir: str, right_dir: str, out_path: str,
                      in_fps: int = FPS, out_fps: int = OUT_FPS) -> bool:
    """Use ffmpeg hstack to produce side-by-side video."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Pad the shorter sequence to match lengths
    left_n  = len([f for f in os.listdir(left_dir)  if f.endswith(".png")])
    right_n = len([f for f in os.listdir(right_dir) if f.endswith(".png")])
    if left_n == 0 or right_n == 0:
        print("[ERROR] One side has no frames")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(in_fps),
        "-i", os.path.join(left_dir,  "frame_%06d.png"),
        "-framerate", str(in_fps),
        "-i", os.path.join(right_dir, "frame_%06d.png"),
        "-filter_complex",
        f"[0:v]fps={out_fps}[l];[1:v]fps={out_fps}[r];[l][r]hstack=inputs=2[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg hstack failed:\n{result.stderr[-500:]}")
        return False

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Comparison video saved → {out_path}  ({size_mb:.1f} MB)")
    return True


def make_comparison(profile: str, baseline_name: str, evasion_name: str,
                    out_path: str, evasion_trigger_step: int | None = None):
    baseline_frames = f"/tmp/evasion_{baseline_name}_frames"
    evasion_frames  = f"/tmp/evasion_{evasion_name}_frames"

    if not os.path.isdir(baseline_frames):
        print(f"[ERROR] Baseline frames not found: {baseline_frames}")
        return
    if not os.path.isdir(evasion_frames):
        print(f"[ERROR] Evasion frames not found: {evasion_frames}")
        return

    meta_b = load_meta(baseline_name)
    meta_e = load_meta(evasion_name)

    with tempfile.TemporaryDirectory() as tmp:
        left_dir  = os.path.join(tmp, "left")
        right_dir = os.path.join(tmp, "right")

        print(f"Annotating baseline frames...")
        annotate_frames(baseline_frames, left_dir,
                        label=f"{profile} — No Evasion  |  baseline",
                        meta=meta_b, evasion=False)

        print(f"Annotating evasion frames...")
        annotate_frames(evasion_frames, right_dir,
                        label=f"{profile} — LGMD-SNN Evasion",
                        meta=meta_e, evasion=True,
                        evasion_step=evasion_trigger_step)

        make_side_by_side(left_dir, right_dir, out_path)

    if meta_b and meta_e:
        print(f"\nSummary — {profile}:")
        print(f"  Baseline  closest approach: {meta_b['closest']:.3f}m")
        verdict = "MISS" if meta_e["closest"] > 0.6 else "HIT"
        print(f"  Evasion   closest approach: {meta_e['closest']:.3f}m  [{verdict}]")
        print(f"  Drone max altitude (evasion): {meta_e['drone_max_z']:.2f}m")


PROFILE_PAIRS = {
    "head_on":  ("head_on_baseline",  "head_on_evasion"),
    "lateral":  ("lateral_baseline",  "lateral_evasion"),
    "high":     ("high_baseline",     "high_evasion"),
    "low":      ("low_baseline",      "low_evasion"),
    "diagonal": ("diagonal_baseline", "diagonal_evasion"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile",  default=None, choices=list(PROFILE_PAIRS.keys()),
                   help="Single profile to process")
    p.add_argument("--baseline_name", default=None,
                   help="Baseline run name (e.g. head_on_baseline)")
    p.add_argument("--evasion_name",  default=None,
                   help="Evasion run name (e.g. head_on_evasion)")
    p.add_argument("--out", default=None,
                   help="Output MP4 path")
    p.add_argument("--all", action="store_true",
                   help="Process all profiles using default naming convention")
    p.add_argument("--evasion_step", type=int, default=None,
                   help="Render step at which evasion triggered (for annotation)")
    args = p.parse_args()

    Path("results/videos").mkdir(parents=True, exist_ok=True)

    if args.all:
        for profile, (b_name, e_name) in PROFILE_PAIRS.items():
            out = f"results/videos/comparison_{profile}.mp4"
            print(f"\n{'='*50}")
            print(f"  {profile}")
            print(f"{'='*50}")
            make_comparison(profile, b_name, e_name, out)
    elif args.profile and args.baseline_name and args.evasion_name:
        out = args.out or f"results/videos/comparison_{args.profile}.mp4"
        make_comparison(args.profile, args.baseline_name, args.evasion_name, out,
                        evasion_trigger_step=args.evasion_step)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
