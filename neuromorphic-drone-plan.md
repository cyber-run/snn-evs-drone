# Neuromorphic Quadcopter: Event-Driven SNN Obstacle Avoidance
## Project Plan

**Goal:** A sim-to-real quadcopter obstacle avoidance system using event cameras and spiking neural networks — demonstrating that the temporal sparsity of event data and SNNs are a natural, efficient pairing for high-speed reactive flight.

**Time budget:** ~40 hrs/week
**Hardware available:** EVS cameras, quadcopters (Imperial)
**Target output:** Working real-world demo + paper submission (ICRA / CoRL / RA-L)

---

## Stack

| Component | Tool | Reason |
|---|---|---|
| Quadcopter simulation | Flightmare | Fast, Python-friendly, RL-ready |
| Event camera simulation | v2e | No ROS dependency, simpler integration |
| SNN framework | SpikingJelly | PyTorch-based, best documented, active community |
| RL training | Stable-Baselines3 / custom | Standard, well-tested |
| Real camera interface | Lucid Vision Arena SDK | Hardware available |
| Version control | Git + GitHub | Public repo from day one |

---

## Phases

### Phase 1 — Foundations (Weeks 1–3) ~120 hrs

**Goal:** Everything running, talking to each other, a quadcopter flying in simulation with synthetic event data being generated.

**Week 1 — Environment setup**
- Install and configure Flightmare, v2e, SpikingJelly
- Get a quadcopter flying in Flightmare with basic PID control
- Understand Flightmare's observation/action space
- Set up project repo with clear structure from day one

**Week 2 — Event camera integration**
- Pipe Flightmare rendered frames through v2e to generate synthetic events
- Visualise event streams — understand what obstacle scenarios look like in event space
- Characterise the noise and quality of v2e output vs real EVS data
- Start building the data ingestion pipeline

**Week 3 — SNN foundations**
- Implement a basic SNN in SpikingJelly — get comfortable with surrogate gradient training
- Define event data encoding strategy (rate coding vs temporal coding vs population coding)
- Build the bridge: event stream → SNN input representation
- Simple classification task to validate the pipeline end-to-end

**Milestone:** Synthetic event data flowing into an SNN, quadcopter flying in sim.

---

### Phase 2 — SNN Policy Development (Weeks 4–8) ~200 hrs

**Goal:** A trained SNN policy that achieves reliable obstacle avoidance in simulation.

**Week 4 — Environment design**
- Design obstacle avoidance task in Flightmare — start simple (single static obstacle, open space)
- Define reward function: forward progress + collision penalty + energy efficiency term
- Define observation space: event frame representation fed to SNN
- Define action space: velocity commands or rotor thrusts

**Week 5 — Baseline**
- Implement a frame-based CNN baseline for the same task — this is your comparison point for the paper
- Train baseline to a working level
- Document performance metrics clearly

**Weeks 6–7 — SNN policy training**
- Implement SNN policy architecture — start shallow, add depth if needed
- Train using surrogate gradient backpropagation through time (BPTT)
- Curriculum learning: begin with easy scenarios (single obstacle, slow speed), gradually increase difficulty
- Expect this to be the hardest part — SNN training is finicky, budget time for debugging

**Week 8 — Evaluation & tuning**
- Systematic evaluation across obstacle densities, speeds, and environment layouts
- Compare SNN vs CNN baseline: accuracy, latency, spike sparsity (this is your efficiency argument)
- Tune reward function and architecture based on results

**Milestone:** SNN policy achieving reliable obstacle avoidance in simulation, outperforming or matching baseline with demonstrably lower computational cost.

---

### Phase 3 — Sim Validation & Real Camera Bridging (Weeks 9–11) ~120 hrs

**Goal:** Validate sim results and begin closing the sim-to-real gap using real EVS hardware.

**Week 9 — Stress testing simulation**
- Test policy robustness: different lighting conditions, obstacle types, speeds
- Identify failure modes — document them, this goes in the paper
- Ablation studies: what happens without temporal coding? with rate coding only? — good paper content

**Week 10 — Real EVS data collection**
- Mount EVS camera, record event streams of obstacles in various scenarios (static setup, no drone yet)
- Compare real event data characteristics vs v2e synthetic data — document the gap
- Fine-tune v2e parameters to better match real camera output

**Week 11 — Domain adaptation**
- Apply domain randomisation in simulation to improve real-world transfer
- If gap is large: explore fine-tuning the policy on a small amount of real event data
- Validate SNN input pipeline works with real camera data format

**Milestone:** Policy trained in sim successfully processing real EVS camera data in a static test.

---

### Phase 4 — Real Hardware Deployment (Weeks 12–15) ~160 hrs

**Goal:** SNN policy running onboard or tethered, avoiding real obstacles on a real quadcopter.

**Week 12 — Hardware integration**
- Mount EVS camera on quadcopter
- Set up onboard compute pipeline: EVS → event encoding → SNN inference → motor commands
- Low-level safety setup: kill switch, net enclosure, tether if needed
- Ground tests: validate latency of full pipeline

**Week 13 — Tethered flight tests**
- Begin with tethered/constrained flight, single obstacle
- Identify and fix integration issues
- Validate that sim-trained policy generalises — expect it to partially work, partially fail

**Week 14 — Iterative real-world refinement**
- Fine-tune policy based on real flight data if needed
- Gradually increase obstacle complexity
- Collect flight data for paper

**Week 15 — Full demo**
- Untethered obstacle avoidance in a structured environment
- Record high quality video — essential for paper submission and GitHub
- Collect quantitative metrics: success rate, collision rate, speed

**Milestone:** Quadcopter avoiding obstacles autonomously using SNN + event camera.

---

### Phase 5 — Paper & Release (Weeks 16–18) ~120 hrs

**Goal:** Submit to a top venue and release clean open-source code.

**Week 16 — Paper writing**
- Structure: Introduction → Related Work → Method → Experiments → Results → Conclusion
- Key contributions to argue:
  1. Event cameras and SNNs are a natural match for high-speed reactive flight
  2. Demonstrated sim-to-real transfer
  3. Efficiency comparison vs frame-based CNN baseline
- Target venue: **CoRL** (Conference on Robot Learning) or **ICRA** — check deadlines now and work backwards

**Week 17 — Paper refinement + figures**
- Clear figures are half the battle in robotics papers
- Must-have: architecture diagram, sim vs real event data comparison, quantitative results table, trajectory plots
- Video submission — most robotics venues accept/encourage supplementary video

**Week 18 — Open source release**
- Clean, documented codebase
- README with clear setup instructions
- Example data so people can run it without hardware
- Pretrained model weights
- Submit paper

---

## Key Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| SNN training instability | High | Start with shallow networks, validate surrogate gradient implementation on simple tasks first |
| Large sim-to-real gap in event data | High | Characterise gap early (Week 10), apply domain randomisation, budget time for fine-tuning |
| Quadcopter integration issues | Medium | Start tethered, have a frame-based fallback controller for safety |
| Compute bottleneck for training | Medium | Use Brev/cloud GPU for training runs, develop locally |
| Paper deadline pressure | Medium | Check CoRL/ICRA deadlines now and work backwards to set internal milestones |

---

## Paper Argument (Core Thesis)

> Event cameras produce sparse, asynchronous data with microsecond timing. SNNs process information as sparse, asynchronous spikes. This representational alignment means the combination is uniquely suited to high-speed reactive robotics — delivering lower latency, lower power consumption, and better temporal resolution than conventional frame-based CNN approaches.

This is what makes it more than an engineering project — it's a scientific argument about why this pairing is principled, not just novel.

---

## Timeline Summary

| Phase | Weeks | Hours |
|---|---|---|
| 1. Foundations | 1–3 | ~120 |
| 2. SNN Policy Development | 4–8 | ~200 |
| 3. Sim Validation & Real Bridging | 9–11 | ~120 |
| 4. Real Hardware Deployment | 12–15 | ~160 |
| 5. Paper & Release | 16–18 | ~120 |
| **Total** | **18 weeks** | **~720 hrs** |
