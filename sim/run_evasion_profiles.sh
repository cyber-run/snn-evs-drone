#!/usr/bin/env bash
# Run baseline + SNN evasion for all 5 profiles and produce comparison videos.
#
# For each profile:
#   1. Baseline run  (no evasion, --video)
#   2. Evasion run   (--evasion --weights ..., --video)
#   3. Comparison video (side-by-side, via make_comparison_video.py)
#
# Results:
#   /tmp/evasion_{profile}_baseline_video.mp4
#   /tmp/evasion_{profile}_evasion_video.mp4
#   results/videos/comparison_{profile}.mp4
#   results/evasion_results.txt  — summary table

set -euo pipefail

REPO="$HOME/snn-evs-drone"
VENV="$HOME/isaaclab-env/bin/activate"
WEIGHTS="${WEIGHTS:-$REPO/results/lgmd_sw.pt}"
THRESHOLD="${THRESHOLD:-0.25}"

export OMNI_KIT_ACCEPT_EULA=Y
export ISAACSIM_PATH="$HOME/isaaclab-env/lib/python3.10/site-packages/isaacsim"

source "$VENV"
cd "$REPO"

if [ ! -f "$WEIGHTS" ]; then
    echo "[ERROR] Weights not found: $WEIGHTS"
    echo "        Set WEIGHTS env var or ensure results/lgmd_sw.pt exists"
    exit 1
fi

PROFILES=(head_on lateral high low diagonal)
RESULTS_FILE="results/evasion_results.txt"
mkdir -p results/videos

echo "LGMD-SNN Evasion Evaluation — $(date)" > "$RESULTS_FILE"
echo "Weights: $WEIGHTS  |  Threshold: $THRESHOLD" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
printf "%-12s  %-20s  %-20s  %-8s\n" \
    "Profile" "Baseline closest (m)" "Evasion closest (m)" "Verdict" >> "$RESULTS_FILE"
echo "-------------------------------------------------------------------" >> "$RESULTS_FILE"

for PROFILE in "${PROFILES[@]}"; do
    BASELINE_NAME="${PROFILE}_baseline"
    EVASION_NAME="${PROFILE}_evasion"

    echo ""
    echo "========================================="
    echo "  Profile: $PROFILE  ($(date +'%H:%M:%S'))"
    echo "========================================="

    # ── 1. Baseline run ───────────────────────────────────────────────────────
    echo "  [1/3] Baseline simulation..."
    python sim/hover_evasion_capture.py \
        --sim-only --profile "$PROFILE" --name "$BASELINE_NAME" --video \
        > "/tmp/sim_${BASELINE_NAME}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "  [ERROR] Baseline failed — check /tmp/sim_${BASELINE_NAME}.log"
        tail -10 "/tmp/sim_${BASELINE_NAME}.log"
        continue
    fi
    echo "  [1/3] Baseline done"

    # ── 2. Evasion run ────────────────────────────────────────────────────────
    echo "  [2/3] SNN evasion simulation..."
    python sim/hover_evasion_capture.py \
        --sim-only --profile "$PROFILE" --name "$EVASION_NAME" \
        --evasion --weights "$WEIGHTS" --dcmd_threshold "$THRESHOLD" --video \
        > "/tmp/sim_${EVASION_NAME}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "  [ERROR] Evasion failed — check /tmp/sim_${EVASION_NAME}.log"
        tail -10 "/tmp/sim_${EVASION_NAME}.log"
        continue
    fi
    echo "  [2/3] Evasion done"

    # ── 3. Comparison video ───────────────────────────────────────────────────
    echo "  [3/3] Generating comparison video..."
    python scripts/make_comparison_video.py \
        --profile "$PROFILE" \
        --baseline_name "$BASELINE_NAME" \
        --evasion_name  "$EVASION_NAME" \
        --out "results/videos/comparison_${PROFILE}.mp4" \
        2>&1 | tail -6

    # ── Extract results from logs ─────────────────────────────────────────────
    BASELINE_CLOSEST=$(grep -oP "(?<=closest approach: )[0-9.]+" \
        "/tmp/sim_${BASELINE_NAME}.log" 2>/dev/null | tail -1 || echo "N/A")
    EVASION_CLOSEST=$(grep -oP "(?<=Closest approach: )[0-9.]+" \
        "/tmp/sim_${EVASION_NAME}.log" 2>/dev/null | tail -1 || echo "N/A")
    VERDICT=$(grep -oP "(?<=EVASION RESULT: )\w+" \
        "/tmp/sim_${EVASION_NAME}.log" 2>/dev/null | tail -1 || echo "N/A")

    printf "%-12s  %-20s  %-20s  %-8s\n" \
        "$PROFILE" "$BASELINE_CLOSEST" "$EVASION_CLOSEST" "$VERDICT" \
        >> "$RESULTS_FILE"

    echo "  Result: baseline=${BASELINE_CLOSEST}m  evasion=${EVASION_CLOSEST}m  [${VERDICT}]"
done

echo ""
echo "========================================="
echo "  All profiles complete  ($(date +'%H:%M:%S'))"
echo "========================================="
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Videos: results/videos/"
ls -lh results/videos/ 2>/dev/null || echo "(none generated)"
