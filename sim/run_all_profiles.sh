#!/usr/bin/env bash
# Run full sim+v2e pipeline for all 11 approach profiles sequentially.
# Each profile: Isaac Sim renders frames at 1000 FPS → v2e converts to events.h5
# Logs: data/sim_<profile>.log and data/v2e_<profile>.log
#
# Usage:
#   bash sim/run_all_profiles.sh                  # Black Gridroom for all
#   bash sim/run_all_profiles.sh --randomize_env  # random environment per profile

set -euo pipefail

REPO="$HOME/snn-evs-drone"
VENV="$HOME/isaaclab-env/bin/activate"
DATA_DIR="$REPO/data"

export OMNI_KIT_ACCEPT_EULA=Y
export ISAACSIM_PATH="$HOME/isaaclab-env/lib/python3.10/site-packages/isaacsim"

source "$VENV"

cd "$REPO"
mkdir -p "$DATA_DIR"

# Pass through extra flags (e.g. --randomize_env)
EXTRA_FLAGS="${*}"

PROFILES=(
    head_on lateral high low diagonal
    head_on_slow lateral_slow diagonal_slow
    head_on_fast lateral_fast diagonal_fast
)

TOTAL=${#PROFILES[@]}
CURRENT=0

for PROFILE in "${PROFILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "========================================="
    echo "  [$CURRENT/$TOTAL] Profile: $PROFILE  ($(date +'%H:%M:%S'))"
    echo "========================================="

    SIM_LOG="$DATA_DIR/sim_${PROFILE}.log"
    V2E_LOG="$DATA_DIR/v2e_${PROFILE}.log"

    echo "  [1/2] Simulation → $SIM_LOG"
    python sim/hover_evasion_capture.py \
        --sim-only --profile "$PROFILE" --name "$PROFILE" \
        $EXTRA_FLAGS \
        > "$SIM_LOG" 2>&1
    SIM_EXIT=$?
    if [ $SIM_EXIT -ne 0 ]; then
        echo "  [ERROR] Simulation failed (exit $SIM_EXIT). Check $SIM_LOG"
        tail -20 "$SIM_LOG"
        exit 1
    fi
    FRAME_COUNT=$(ls "$DATA_DIR"/evasion_${PROFILE}_frames/frame_*.jpg "$DATA_DIR"/evasion_${PROFILE}_frames/frame_*.bmp 2>/dev/null | wc -l)
    echo "  [1/2] Done — $FRAME_COUNT frames captured"

    echo "  [2/2] v2e → $V2E_LOG"
    python sim/hover_evasion_capture.py \
        --v2e-only --name "$PROFILE" \
        > "$V2E_LOG" 2>&1
    V2E_EXIT=$?
    if [ $V2E_EXIT -ne 0 ]; then
        echo "  [ERROR] v2e failed (exit $V2E_EXIT). Check $V2E_LOG"
        tail -20 "$V2E_LOG"
        exit 1
    fi

    H5="$DATA_DIR/evasion_${PROFILE}_events/events.h5"
    if [ -f "$H5" ]; then
        H5_SIZE=$(du -sh "$H5" | cut -f1)
        echo "  [2/2] Done — events.h5 ($H5_SIZE)"
    else
        echo "  [ERROR] events.h5 not found at $H5"
        exit 1
    fi
done

echo ""
echo "========================================="
echo "  All $TOTAL profiles complete!  ($(date +'%H:%M:%S'))"
echo "========================================="
echo ""
echo "H5 files:"
for PROFILE in "${PROFILES[@]}"; do
    H5="$DATA_DIR/evasion_${PROFILE}_events/events.h5"
    [ -f "$H5" ] && echo "  $H5  ($(du -sh $H5 | cut -f1))" || echo "  MISSING: $H5"
done
