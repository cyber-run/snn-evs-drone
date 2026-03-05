#!/usr/bin/env bash
# Run full sim+v2e pipeline for all 5 approach profiles sequentially.
# Each profile: Isaac Sim renders frames → v2e converts to events.h5
# Logs: /tmp/sim_<profile>.log and /tmp/v2e_<profile>.log

set -euo pipefail

REPO="$HOME/snn-evs-drone"
VENV="$HOME/isaaclab-env/bin/activate"

export OMNI_KIT_ACCEPT_EULA=Y
export ISAACSIM_PATH="$HOME/isaaclab-env/lib/python3.10/site-packages/isaacsim"

source "$VENV"

cd "$REPO"

PROFILES=(head_on lateral high low diagonal)

for PROFILE in "${PROFILES[@]}"; do
    echo ""
    echo "========================================="
    echo "  Profile: $PROFILE  ($(date +'%H:%M:%S'))"
    echo "========================================="

    SIM_LOG="/tmp/sim_${PROFILE}.log"
    V2E_LOG="/tmp/v2e_${PROFILE}.log"

    echo "  [1/2] Simulation → $SIM_LOG"
    python sim/hover_evasion_capture.py \
        --sim-only --profile "$PROFILE" --name "$PROFILE" \
        > "$SIM_LOG" 2>&1
    SIM_EXIT=$?
    if [ $SIM_EXIT -ne 0 ]; then
        echo "  [ERROR] Simulation failed (exit $SIM_EXIT). Check $SIM_LOG"
        tail -20 "$SIM_LOG"
        exit 1
    fi
    FRAME_COUNT=$(ls /tmp/evasion_${PROFILE}_frames/*.bmp 2>/dev/null | wc -l)
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

    H5="/tmp/evasion_${PROFILE}_events/events.h5"
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
echo "  All 5 profiles complete!  ($(date +'%H:%M:%S'))"
echo "========================================="
echo ""
echo "H5 files:"
for PROFILE in "${PROFILES[@]}"; do
    H5="/tmp/evasion_${PROFILE}_events/events.h5"
    [ -f "$H5" ] && echo "  $H5  ($(du -sh $H5 | cut -f1))" || echo "  MISSING: $H5"
done
