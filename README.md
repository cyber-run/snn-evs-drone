# Neuromorphic Quadcopter: Event-Driven SNN Obstacle Avoidance

Sim-to-real quadcopter obstacle avoidance using event cameras and spiking neural networks.

## Stack
- **Simulator**: Isaac Lab + Aerial Gym
- **Event camera sim**: v2e
- **SNN framework**: SpikingJelly
- **Cloud GPU**: Brev (RTX A6000)

## Structure
```
sim/          Isaac Lab env configs and wrappers
events/       Event camera pipeline (v2e integration)
snn/          SNN models and training code
scripts/      Utility and experiment scripts
results/      Experiment outputs (gitignored)
docs/         Notes, figures
```

## Setup

### Local
```bash
pip install -r requirements.txt
```

### Brev (cloud)
Use the official NVIDIA Isaac Lab Launchable, then:
```bash
# inside Brev instance
git clone <this-repo>
pip install -r requirements.txt
# Isaac Lab and Aerial Gym are pre-installed via Launchable
```
