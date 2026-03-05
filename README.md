# Neuromorphic Quadcopter: Insect-Inspired Event-Driven Obstacle Avoidance

Sim-to-real quadcopter collision avoidance using event cameras and spiking neural networks, with architecture inspired by the locust LGMD (Lobula Giant Movement Detector) neuron.

## Overview

A hovering quadcopter detects and evades incoming obstacles using only event camera data processed by a biologically-plausible SNN. The system is trained entirely in simulation and deployed on real hardware, demonstrating efficient, low-latency reactive flight without frame-based vision.

**Target venues:** ICRA / CoRL / RA-L

## Stack

| Component | Tool | Notes |
|---|---|---|
| Quadcopter simulator | Isaac Sim 4.5 + Pegasus Simulator 5.1.0 | Iris quadrotor, physics-accurate |
| Event camera simulator | v2e 1.5.1 + SuperSloMo | DAVIS346 resolution, ~926µs timestamp resolution at 120 FPS |
| SNN framework | SpikingJelly | PyTorch-based surrogate gradient training |
| Cloud GPU | Brev + Crusoe (L40s 48GB) | RTX-class GPU required for Isaac Sim renderer |
| Real camera interface | Lucid Vision Arena SDK | Hardware at Imperial |

## Architecture: LGMD-Inspired SNN

Biological inspiration: the locust *Lobula Giant Movement Detector* (LGMD) neuron, one of the best-documented collision-avoidance circuits in biology. It responds selectively to looming stimuli — objects expanding in the visual field — exactly the signature an approaching obstacle produces in an event camera.

```
Event Camera (DAVIS346)
        │
        ▼
  Preprocessing layer
  ├── ON/OFF channel split
  ├── Spatial pooling: AvgPool2d(4) → 86×65
  └── Temporal binning into spike frames (10ms bins)
        │
        ▼
  LGMD SNN (SpikingJelly)
  ├── Excitation conv (2→1, 3×3): ON events excite, OFF weakly inhibit
  ├── Lateral inhibition (7×7 Gaussian, 1-step delay): suppresses background motion
  ├── LIF neurons per spatial location (v_threshold=0.5)
  └── DCMD readout: fixed uniform spatial weighting → global sum → collision-imminence spike rate
        │
        ▼
  Evasion controller
  └── DCMD rate > threshold → lateral thrust override
        │
        ▼
  Pegasus Multirotor (Isaac Sim)
```

Key properties of LGMD:
- **Looming selectivity**: responds to angular expansion rate `dθ/dt`, not absolute size or speed
- **Lateral inhibition**: 1-step delayed inhibition makes the cell selective to expanding (not translating) edges
- **DCMD output**: single collision-imminence signal — simple to threshold for reactive evasion
- **Analytical training label**: `dθ/dt` computed from known obstacle trajectory — no manual annotation

## Project Structure

```
sim/
  headless_hover_test.py     Physics pipeline validation (drone falls under gravity)
  hover_evasion_capture.py   Hovering drone + dynamic obstacle launched at it → frames + trajectory

events/
  capture_and_convert.py     Static obstacle scene (legacy, for pipeline testing)
  visualise_events.py        Event stream visualisation and per-bin rate stats

snn/
  models/
    lgmd_net.py              LGMD SNN architecture (SpikingJelly)
    event_encoder.py         Raw events → (T, 2, H, W) spike frames; dθ/dt label generation
  training/
    train_lgmd.py            Training on looming event sequences; supports multiple --h5 inputs

scripts/      Utility and experiment scripts
results/      Experiment outputs (gitignored)
docs/         Notes, architecture diagrams
```

## Simulation Scenario

The training scenario is designed to produce clean looming events matching what LGMD is tuned for:

1. Iris quadrotor hovers at fixed point (altitude P+D controller, base throttle 568 rad/s)
2. Environment: **Black Gridroom** — static grid background, no animated lighting or sky
3. `DynamicCuboid` obstacle (1.0 m cube, **checkerboard texture**) holds at launch point during 0.5 s warmup (gravity disabled)
4. At step 60, obstacle launched toward drone at 8–10 m/s — fills FOV rapidly, generating a dense looming event burst
5. Camera captures expanding face at 120 FPS → ~960 frames per 8 s run (lossless BMP)
6. v2e converts frames to synthetic events with SuperSloMo ~9× upsampling (~926 µs resolution)
7. Trajectory metadata (obstacle positions, drone position, sim_dt, launch_step) embedded into `events.h5`
8. LGMD SNN trained on analytical `dθ/dt` label with **physically-aligned time axis** (µs timestamps matched to sim time)

**Why checkerboard texture?**
A uniform solid-colour cube generates events only at its expanding perimeter edges — ~18 px at 5 m distance.
A checkerboard creates a grid of high-contrast boundaries across the entire face; every tile edge fires as the
cube translates even a fraction of a pixel, providing orders-of-magnitude more looming signal.

### Approach profiles

| Profile | Launch pos (m) | Velocity (m/s) | Approx. time-to-contact |
|---|---|---|---|
| `head_on` | (6, 0, 1.5) | (-10, 0, 0) | ~0.6 s |
| `lateral` | (6, 3, 1.5) | (-8, -4, 0) | ~0.8 s |
| `high` | (6, 0, 3.5) | (-8, 0, -2) | ~0.8 s |
| `low` | (6, 0, -0.5) | (-8, 0, 2) | ~0.8 s |
| `diagonal` | (5, 5, 1.5) | (-6, -6, 0) | ~0.6 s |

## Setup

### Cloud (Brev + Crusoe L40s)

Deploy from a fresh Ubuntu 22.04 + RTX GPU instance:

```bash
git clone https://github.com/cyber-run/snn-evs-drone.git
cd snn-evs-drone
bash setup.sh
source ~/.bashrc
```

This installs: Isaac Sim 4.5, Isaac Lab, Pegasus Simulator 5.1.0, v2e, SpikingJelly, and all project dependencies.

Download SuperSloMo checkpoint (required for high-resolution event timestamps):
```bash
pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/19YDLygMkXey4ePj8_W54BVlkKxTxWiEk" \
      -O ~/v2e/input/SuperSloMo39.ckpt
```

Required environment variables for all Isaac Sim scripts:
```bash
export OMNI_KIT_ACCEPT_EULA=Y
export ISAACSIM_PATH=~/isaaclab-env/lib/python3.10/site-packages/isaacsim
```

### Running the pipeline

```bash
# 1. Run simulation (hover drone + obstacle approach), one profile at a time
OMNI_KIT_ACCEPT_EULA=Y ISAACSIM_PATH=... \
  python sim/hover_evasion_capture.py --sim-only --profile head_on --name head_on
OMNI_KIT_ACCEPT_EULA=Y ISAACSIM_PATH=... \
  python sim/hover_evasion_capture.py --sim-only --profile lateral --name lateral

# 2. Convert frames to events via v2e (run after sim exits)
python sim/hover_evasion_capture.py --v2e-only --name head_on
python sim/hover_evasion_capture.py --v2e-only --name lateral

# Output: /tmp/evasion_{name}_events/events.h5 with trajectory metadata embedded

# 3. Train LGMD SNN on one or more recordings
#    --val_h5 holds out a full recording for validation (default: diagonal profile)
python snn/training/train_lgmd.py \
  --h5 /tmp/evasion_head_on_events/events.h5 \
      /tmp/evasion_lateral_events/events.h5 \
      /tmp/evasion_high_events/events.h5 \
      /tmp/evasion_low_events/events.h5 \
  --val_h5 /tmp/evasion_diagonal_events/events.h5 \
  --epochs 50 --batch 8 --dt_us 10000 --n_bins 20 \
  --save results/lgmd_weights.pt

# 4. Visualise event stream
python events/visualise_events.py --h5 /tmp/evasion_head_on_events/events.h5

# 5. Physics validation
OMNI_KIT_ACCEPT_EULA=Y ISAACSIM_PATH=... python sim/headless_hover_test.py
```

### Local (no GPU)

```bash
pip install -r requirements.txt
# Isaac Sim / Pegasus require the cloud GPU instance
# SpikingJelly, event encoding, and visualisation work locally
```

## Isaac Sim 4.5 / Pegasus 5.1.0 Compatibility

Three incompatibilities fixed in this repo:

1. `MonocularCamera.start()` calls `set_lens_distortion_model` (removed in 4.5) — fixed via `MonocularCameraIsaacSim45` subclass
2. `set_resolution(maintain_square_pixels=True)` kwarg removed — fixed in same subclass
3. `PythonBackend` removed in Pegasus 5.1 — backends inherit `Backend` base class directly

## v2e Pipeline Notes

- v2e reads **all files** in the input directory as frames — trajectory metadata (`meta.npz`) must be saved to a **separate directory**, not alongside frames
- If v2e crashes mid-write, the HDF5 file becomes locked — `rm -f events.h5` before retrying
- `simulation_app.close()` calls `sys.exit()` — v2e must run after the sim process has exited (use `--sim-only` then `--v2e-only`)
- SuperSloMo checkpoint must be present at `~/v2e/input/SuperSloMo39.ckpt`

## Key References

- Rind & Bramwell (1996) — Original LGMD computational model
- Stafford et al. (2007) — LGMD-based robot collision avoidance
- Meng et al. (2023) — SNN implementation of LGMD for UAV avoidance
- Hu et al. (2021) — v2e: Video to Events synthetic event generation
- Fang et al. (2023) — SpikingJelly SNN training framework
