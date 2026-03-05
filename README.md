# Neuromorphic Quadcopter: Insect-Inspired Event-Driven Obstacle Avoidance

Sim-to-real quadcopter collision avoidance using event cameras and spiking neural networks, with architecture inspired by the locust LGMD (Lobula Giant Movement Detector) neuron.

## Overview

A hovering quadcopter detects and evades incoming obstacles using only event camera data processed by a biologically-plausible SNN. The system is trained entirely in simulation and deployed on real hardware, demonstrating efficient, low-latency reactive flight without frame-based vision.

**Target venues:** ICRA / CoRL / RA-L

## Stack

| Component | Tool | Notes |
|---|---|---|
| Quadcopter simulator | Isaac Sim 4.5 + Pegasus Simulator 5.1.0 | Iris quadrotor, physics-accurate |
| Event camera simulator | v2e 1.5.1 + SuperSloMo | DAVIS346 resolution, 980µs timestamp resolution |
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
  ├── Spatial pooling (reduce resolution)
  └── Temporal binning into spike frames
        │
        ▼
  LGMD SNN (SpikingJelly)
  ├── Excitation: ON events from expanding edge
  ├── Lateral inhibition: suppresses background motion
  └── DCMD readout: collision imminence spike rate
        │
        ▼
  Evasion controller
  └── DCMD rate → thrust / pitch / roll override
        │
        ▼
  Pegasus Multirotor (Isaac Sim)
```

Key properties of LGMD:
- **Looming selectivity**: responds to angular expansion rate, not absolute size or speed
- **Lateral inhibition**: suppresses wide-field background optic flow (e.g. from self-rotation)
- **DCMD output**: a single collision-imminence signal — simple to threshold for evasion
- **No training data required for basic version**: the circuit structure encodes the behaviour

## Project Structure

```
sim/          Isaac Sim + Pegasus scene scripts
  headless_hover_test.py    Physics pipeline validation
  hover_evasion_capture.py  Hovering drone + dynamic incoming obstacles (TODO)

events/       Event camera pipeline
  capture_and_convert.py    Frame capture → v2e synthetic events
  visualise_events.py       Event stream visualisation and stats

snn/          SNN models and training
  lgmd_net.py               LGMD SNN architecture (SpikingJelly) (TODO)
  train_lgmd.py             Training on looming event sequences (TODO)

scripts/      Utility and experiment scripts
results/      Experiment outputs (gitignored)
docs/         Notes, architecture diagrams
```

## Simulation Scenario

The training scenario is designed to produce clean looming events:

1. Drone hovers at a fixed point with a PID hover controller
2. Dynamic obstacles (cuboids) are launched toward the drone from various angles
3. The event camera captures the expanding silhouette as a burst of ON events
4. The LGMD SNN processes events and outputs a collision-imminence signal
5. Above a threshold, the evasion controller overrides hover with a lateral thrust

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

SuperSloMo checkpoint (required for v2e high-resolution events):
```bash
pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/19YDLygMkXey4ePj8_W54BVlkKxTxWiEk" \
      -O ~/v2e/input/SuperSloMo39.ckpt
```

### Running the pipeline

```bash
# 1. Capture frames and generate events
OMNI_KIT_ACCEPT_EULA=Y ISAACSIM_PATH=~/isaaclab-env/lib/python3.10/site-packages/isaacsim \
  python events/capture_and_convert.py --sim-only

python events/capture_and_convert.py --v2e-only

# 2. Visualise event stream
python events/visualise_events.py --h5 /tmp/sim_events_obstacles/events.h5

# 3. Physics validation only
OMNI_KIT_ACCEPT_EULA=Y ISAACSIM_PATH=~/isaaclab-env/lib/python3.10/site-packages/isaacsim \
  python sim/headless_hover_test.py
```

### Local (no GPU)

```bash
pip install -r requirements.txt
# Isaac Sim / Pegasus require the cloud instance
# SpikingJelly and event visualisation work locally
```

## Isaac Sim 4.5 Compatibility Notes

Pegasus 5.1.0 has two incompatibilities with Isaac Sim 4.5, both fixed in this repo:

1. `MonocularCamera.start()` calls `set_lens_distortion_model` (removed in 4.5) — fixed via `MonocularCameraIsaacSim45` subclass in `capture_and_convert.py`
2. `set_resolution(maintain_square_pixels=True)` kwarg removed — fixed in the same subclass
3. `PythonBackend` removed — all backends now inherit from `Backend` base class directly

## Key References

- Rind & Bramwell (1996) — Original LGMD computational model
- Stafford et al. (2007) — LGMD-based robot collision avoidance
- Meng et al. (2023) — SNN implementation of LGMD for UAV avoidance
- v2e: Hu et al. (2021) — Synthetic event generation from video
- SpikingJelly: Fang et al. (2023) — SNN training framework
