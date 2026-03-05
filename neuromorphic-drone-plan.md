# Neuromorphic Quadcopter: Project Plan
## Insect-Inspired Event-Driven Obstacle Avoidance

**Goal:** A sim-to-real quadcopter collision avoidance system using event cameras and a biologically-plausible SNN, demonstrating that the temporal sparsity of event data and SNNs are a natural, efficient pairing for high-speed reactive flight.

**Time budget:** ~40 hrs/week
**Hardware available:** EVS cameras, quadcopters (Imperial)
**Target output:** Working real-world demo + paper submission (ICRA / CoRL / RA-L)

---

## Stack (confirmed)

| Component | Tool | Reason |
|---|---|---|
| Quadcopter simulator | Isaac Sim 4.5 + Pegasus 5.1.0 | Physics-accurate, Iris quadrotor, NVIDIA-supported |
| Event camera simulator | v2e 1.5.1 + SuperSloMo | 980µs timestamp resolution, DAVIS346, no ROS required |
| SNN framework | SpikingJelly | PyTorch-based surrogate gradient, active community |
| Cloud GPU | Brev + Crusoe (L40s 48GB) | RTX GPU required for Isaac Sim renderer |
| Real camera interface | Lucid Vision Arena SDK | Hardware available at Imperial |

Dropped:
- **Flightmare** — unmaintained, no event camera support
- **Aerial Gym** — requires legacy Isaac Gym, incompatible with Isaac Lab

---

## Biological Inspiration: LGMD

The core SNN architecture is modelled on the **Locust LGMD (Lobula Giant Movement Detector)**, one of the best-characterised collision-avoidance neurons in biology.

**Why LGMD:**
- Responds selectively to looming stimuli (objects expanding in the visual field)
- Lateral inhibition suppresses background optic flow — robust to ego-motion
- Output (DCMD) is a single spike-rate signal: collision imminence
- Well-documented computational models (Rind & Bramwell 1996; Stafford 2007)
- Prior SNN implementations exist as reference (Meng et al. 2023)
- Maps naturally onto event camera output: expanding edge = burst of radial ON events

**Scenario design driven by biology:**
- Drone hovers in place (PID hold) → minimises background optic flow
- Obstacles launched toward drone → maximises looming signal
- This is the exact stimulus LGMD is tuned for

---

## Simulation Scenario

1. Iris quadrotor hovers at fixed point with PID controller
2. `DynamicCuboid` obstacles launched at drone from various angles/speeds
3. Event camera (DAVIS346, 346×260) captures expanding silhouette
4. v2e converts frames to synthetic events (SuperSloMo 17× upsampling, 980µs resolution)
5. LGMD SNN processes event stream → DCMD collision-imminence output
6. Evasion controller: DCMD spike rate > threshold → lateral thrust override

---

## Phases

### Phase 1 — Foundations (Weeks 1–3) ✓ Complete

**Week 1 — Environment setup** ✓
- Isaac Sim 4.5 + Pegasus 5.1.0 installed on Brev/Crusoe L40s
- Physics validated: `sim/headless_hover_test.py` — Iris falls under gravity, ~700 steps/sec
- Repo structure, GitHub, setup.sh deployment script

**Week 2 — Event camera pipeline** ✓
- `events/capture_and_convert.py`: Isaac Sim → 502 frames → v2e → 1.31M events
- SuperSloMo upsampling, DAVIS346 (346×260); event format: HDF5 `(N, 4)` uint32 `[timestamp_us, x, y, polarity]`
- Isaac Sim 4.5 compatibility fixes: `MonocularCameraIsaacSim45`, `Backend` base class
- `events/visualise_events.py`: accumulation-window video renderer

**Week 3 — SNN foundations** ✓
- `snn/models/lgmd_net.py`: LGMD SNN in SpikingJelly (excitation + delayed lateral inhibition + DCMD readout)
- `snn/models/event_encoder.py`: events → `(T, 2, H, W)` spike frames; analytical `dθ/dt` label from trajectory
- `snn/training/train_lgmd.py`: training on looming sequences, multi-H5 support, 50 epochs in ~40s on L40s
- `sim/hover_evasion_capture.py`: hover drone + dynamic obstacle, 120 FPS, trajectory metadata in H5
- Initial training run: 2 profiles (head_on, lateral), 164 windows, loss 0.054 → 0.046
- **Known issue**: gravity on obstacle not disabled until end of session — all data before fix needs regeneration
- FPS raised to 120 (SuperSloMo ~9× vs 17×, v2e ~2× faster, more accurate events)

**Milestone:** LGMD SNN architecture complete and training pipeline working end-to-end. ✓

---

### Phase 2 — Dynamic Obstacle Scene + LGMD Training (Weeks 4–8) ← current

**Week 4 — Data quality diagnosis + simulation fixes** ← in progress
- [x] Diagnose root cause of training failure: textureless obstacle + dynamic environment artifacts + label misalignment
  - "Curved Gridroom" produced up to 581 K spurious events/bin from animated lighting while obstacle was stationary
  - Solid-colour obstacle generated only ~84 events/bin looming signal (< background noise at 91 events/bin)
  - `make_label_from_trajectory` used uniform fraction resampling instead of physical time axis
- [x] Fix simulation: Black Gridroom + checkerboard texture + 1.0 m obstacle + 8–10 m/s approach + 0.5 s warmup
- [x] Fix label alignment: `dθ/dt` mapped via physical time axis (µs timestamps ↔ sim step × sim_dt)
- [x] SNN model fixes: LIF v_threshold 0.5, `dcmd_weight` fixed buffer, `net_exc` auxiliary signal
- [x] Training improvements: recording-level val split, combined Pearson loss, per-epoch validation metrics
- [ ] **Regenerate all 5 profiles** with updated `hover_evasion_capture.py` and retrain
- [ ] Validate: DCMD peaks during looming phase; Pearson val_corr > 0.5

**Week 5 — LGMD SNN training refinement**
- [ ] Supervised training on full 5-profile dataset
- [ ] Frame-based CNN baseline on same task (efficiency comparison for paper)
- [ ] Ablation: effect of lateral inhibition delay, pooling factor, time bin size

**Weeks 6–7 — Evasion controller**
- [ ] DCMD output → lateral thrust/pitch/roll override
- [ ] Closed-loop sim: hover → detect → evade → re-hover
- [ ] Curriculum: slow obstacles → fast, single → multiple

**Week 8 — Evaluation**
- Metrics: evasion success rate, reaction latency, spike sparsity
- SNN vs CNN baseline: accuracy, power, latency
- Ablation: effect of lateral inhibition, temporal coding, accumulation window

**Milestone:** Closed-loop evasion in simulation with quantified metrics.

---

### Phase 3 — Sim Validation & Real Camera Bridging (Weeks 9–11)

**Week 9 — Stress testing**
- Varied lighting, obstacle materials, approach profiles
- Failure mode documentation (essential for paper)
- Domain randomisation for sim-to-real

**Week 10 — Real EVS data**
- Record real DAVIS346 data of approaching objects (bench test, no drone)
- Characterise sim-to-real gap in event statistics
- Fine-tune v2e parameters to match real camera

**Week 11 — Domain adaptation**
- If gap is large: fine-tune on small real dataset
- Validate SNN input pipeline with real camera format

**Milestone:** Policy trained in sim processing real EVS data in static test.

---

### Phase 4 — Real Hardware Deployment (Weeks 12–15)

**Week 12 — Hardware integration**
- Mount DAVIS346 on quadcopter
- Pipeline: EVS → encoding → SNN → motor commands
- Safety setup: kill switch, net enclosure

**Week 13–14 — Tethered and iterative flight**
- Tethered tests with single incoming obstacle
- Real-world fine-tuning if needed

**Week 15 — Full demo**
- Untethered obstacle avoidance
- Record video for paper + GitHub

**Milestone:** Quadcopter autonomously evading incoming obstacles with event camera + SNN.

---

### Phase 5 — Paper & Release (Weeks 16–18)

**Core paper argument:**
> Event cameras produce sparse, asynchronous spikes when edges move — exactly the stimulus the locust LGMD is tuned to detect. We show that an SNN modelled on the LGMD circuit detects incoming collisions from event camera data with lower latency and power than frame-based CNN approaches, and transfers from simulation to real hardware.

**Key contributions:**
1. LGMD-inspired SNN architecture for event-based looming detection
2. Sim-to-real transfer via v2e + domain randomisation
3. Efficiency comparison: SNN vs CNN (sparsity, latency, energy)
4. Real quadcopter demo

**Target:** CoRL / ICRA — check deadlines and work backwards.

---

## Key Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| LGMD SNN insufficient for noisy real events | Medium | Domain randomisation + fine-tuning on small real dataset |
| Sim-to-real event statistics gap | High | Characterise early (Week 10), tune v2e noise params |
| SNN training instability | Medium | Start shallow, validate on synthetic looming sequences first |
| Quadcopter integration issues | Medium | Tethered tests, frame-based fallback controller |
| Isaac Sim API changes | Low | Compatibility fixes already in place; version-pinned |

---

## Timeline Summary

| Phase | Weeks | Status |
|---|---|---|
| 1. Foundations | 1–3 | Complete ✓ |
| 2. Dynamic Scene + LGMD Training | 4–8 | Week 4 in progress |
| 3. Sim Validation & Real Bridging | 9–11 | Upcoming |
| 4. Real Hardware Deployment | 12–15 | Upcoming |
| 5. Paper & Release | 16–18 | Upcoming |
