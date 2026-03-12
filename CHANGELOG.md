# Changelog

## [Unreleased] — Session 6

### Feature: external (third-person) camera for evasion visualisation

Added a fixed world-space camera that captures both drone and obstacle during evasion runs,
enabling side-by-side comparison videos showing the full scene rather than just the onboard
drone-eye view.

- **`--ext_camera` flag** on `hover_evasion_capture.py`: activates a fixed external camera
  at position `(8, -10, 5)` looking at the drone hover point `(0, 0, 1.5)`, outputting
  1280×720 frames to `/tmp/evasion_{name}_extframes/`.
- **Replicator API** (`omni.replicator.core`): uses `rep.create.camera(look_at=...)` +
  `rep.AnnotatorRegistry.get_annotator("rgb")` for world-space camera capture. Earlier
  attempts with `omni.isaac.sensor.Camera` and `get_rgba()` produced near-black frames
  (mean≈2, max≈38) because annotators were invalidated by `world.reset()` and the
  `get_rgba()` method returned an uninitialised buffer. Replicator handles this correctly.
- **`_look_at_quat()` helper**: builds a USD-convention rotation matrix (`local X=right,
  local Y=up, local Z=-forward`) from eye/target positions. Return value corrected to
  Isaac Sim's scalar-first `[w, x, y, z]` quaternion format (was scipy `[x, y, z, w]`).
- **`make_comparison_video.py`** already preferred `_extframes` over onboard frames — no
  changes needed; comparison videos now automatically use the external view.
- **`run_evasion_profiles.sh`** already passed `--ext_camera --video` — all 5 profiles
  produce external-camera comparison videos in a single batch run.

### Evaluation: all 5 profiles run with external camera

Full baseline vs SNN-evasion comparison run completed on Brev L40s. Results:

| Profile | Evasion closest approach | Verdict |
|---------|--------------------------|---------|
| head_on | 1.21 m | MISS |
| lateral | 0.96 m | MISS |
| high | 1.43 m | MISS |
| low | 3.63 m | N/A (no trigger) |
| diagonal | 1.42 m | MISS |

4/5 profiles triggered evasion. `low` profile (obstacle from below) did not trigger —
the upward approach produces insufficient looming signal at the current threshold (0.25).
Comparison videos saved to `results/videos/comparison_{profile}.mp4`.

### Infrastructure: migrated to local workstation (nina)

Brev L40s instance restarted, wiping all `/tmp/` training data (BMP frames + events.h5).
Results and trained weights in `results/` were preserved (already pulled to local).

- **New machine**: `nina` (Ubuntu 24.04, NVIDIA RTX 3080 10 GB, local network)
- **Setup**: Isaac Sim 4.5 + Pegasus 5.1.0 + v2e installed via `setup.sh` (adapted for Python 3.10 on Ubuntu 24.04)
- **SuperSloMo checkpoint**: copied from Brev → `~/v2e/input/SuperSloMo39.ckpt`
- **Data regeneration**: all 5 profiles re-running via `run_all_profiles.sh` on nina (estimated 1–2 hrs vs ~30 min on L40s due to lower GPU throughput)
- **Lesson**: training data should be saved to a persistent directory (not `/tmp/`) or pushed to object storage between sessions

---

## [Unreleased] — Session 5

### Feature: closed-loop LGMD-SNN evasion demo

First complete closed-loop validation: trained LGMD-SNN detects a head-on looming obstacle
and triggers a reactive evasion manoeuvre in Pegasus Simulator.

**Result**: MISS at **0.82 m** closest approach (obstacle radius 0.5 m → 0.32 m clearance).
Baseline (no evasion): **0.661 m** — drone center inside the obstacle → HIT.

- Evasion triggered at **t = 3.76 s** (0.76 s after obstacle launch), DCMD = 0.28
- Drone climbed from 1.5 m → **5.80 m** peak; new hover target set to 5.5 m
- Figures: `results/evasion_burst_v3.png` (DCMD trace + annotation) and
  `results/evasion_trajectory_v3.png` (altitude + distance comparison)

### Feature: full position+attitude (SO(3)) hover controller

Replaced the altitude-only P+D controller (`BASE_THROTTLE=568`) with a proper cascaded
position+attitude controller matching the Pegasus nonlinear controller design:

- **Outer loop** (position PID): `F_des = -KP·ep - KD·ev - KI·∫ep` + gravity feedforward.
  Gains `KP=10, KD=8.5, KI=1.5` on all axes; integral clamped to ±5 m·s to prevent windup.
- **Inner loop** (SO(3) attitude PD): desired body-z aligned to `F_des`; SO(3) error
  `e_R = vee(R_des^T R - R^T R_des) / 2`; gains `KR=3.5, KW=0.5`.
- `force_and_torques_to_velocities(u1, tau)` converts thrust + torques to rotor speeds via
  Pegasus mixing matrix.

Root cause of the previous drone crash: `update_state()` in Pegasus 5.1 + Isaac Sim 4.5
is not reliably called as a physics callback. State is now read directly from
`self.vehicle.state` inside `Backend.update()`.

### Changed — `sim/hover_evasion_capture.py`

- **`HoverController`** class replaces `AltitudePID`: cascaded position PID + SO(3)
  attitude PD; correctly reads vehicle state; no longer drifts laterally.
- **Evasion state machine**:
  - DCMD threshold check gated on `_sim_time > warmup_s` (accumulated via `dt`) instead
    of `step_count > WARMUP_STEPS` — fixes 2× timing error (physics runs at 2× render rate).
  - On trigger: set `ctrl.target_pos[2] = 5.5 m`, reset `ctrl._int[2] = 0`, start
    `_evasion_burst_remain = 72` physics steps (~0.3 s) of full thrust (`THRUST_MAX=60 N`).
  - During burst: attitude PD still active to stay level; vertical thrust set to `THRUST_MAX`.
  - After burst: normal position+attitude PID to new altitude target.
  - Evasion direction change uses `target_pos` (not direct force injection), preventing the
    KP term from cancelling the evasion force.
- **`--evasion` / `--weights` / `--dcmd_threshold` flags** for closed-loop demo mode.
- **Log-diff inline event camera**: consecutive rendered frames converted to ON/OFF spikes
  via log-luminance difference with configurable threshold. No v2e required at inference time.
- Trajectory result summary printed at end: evasion time, DCMD value, closest approach,
  HIT/MISS verdict.

### Added — `scripts/eval_dcmd.py`

Sliding-window DCMD visualisation over a full event recording:
- Encodes all bins upfront, slides causal `n_bins` window, batches forward passes.
- Plots: event rate (grey), `dθ/dt` label (blue), DCMD raw + smoothed (red), launch marker.
- Used by `plot_training.py::plot_evasion_result` for the publication figure.

### Added — `scripts/plot_training.py::plot_evasion_result`

Hero figure function: DCMD trace with evasion trigger annotated, outcome box (MISS/HIT
closest-approach distance vs baseline), looming region shading.

---

## [Unreleased] — Session 4

### Performance: encoder spatial pre-pooling + pre-encoding

- **`EventEncoder` spatial pre-pooling** (`spatial_downsample` parameter): event coordinates
  are integer-divided by downsample factor *before* bincount, producing `(T, 2, H//ds, W//ds)`
  output directly. At `ds=4` this is 16× fewer output elements (65×87 vs 260×346) and ~16×
  faster encoding. The model receives pre-pooled frames and runs with `pool_factor=1`.
- **Pre-encoding at init**: `LoomingDataset` now encodes all windows upfront at pooled
  resolution (~2.3 MB/window, ~5 GB total for 2270 windows). `__getitem__` is a pure tensor
  lookup — no encoding, no augmentation, no CPU bottleneck.

### Performance: GPU batch augmentation

- **`EventAugmentor`** operates on `(T, B, 2, H, W)` GPU tensors in the training loop, not
  per-sample in CPU DataLoader workers. Augmentations (horizontal flip, vertical flip,
  polarity swap, noise injection, event dropout) use per-sample random masks via broadcasting.
  Eliminates `torch.rand_like` overhead in CPU workers that was consuming 92% CPU.
- **Result**: epoch time reduced from **118 s → 1.65 s** (71× speedup) through the combined
  spatial pre-pooling, pre-encoding, and GPU augmentation changes.

### Changed — `snn/training/train_lgmd.py`

- **Updated defaults**: `dt_us=2000` (2 ms bins), `n_bins=50` (100 ms window),
  `stride_bins=10` (80% overlap), `tau=5.0`, `batch=32`, `epochs=200`.
- **`--pool` flag** (default 4): controls `EventEncoder.spatial_downsample` and matching
  model `pool_factor=1`.
- **`WeightedRandomSampler`**: oversamples looming windows (label > threshold) by
  `--loom_weight` factor (default 3.0) to counter the ~7–10% looming class imbalance.
- **`--augment` flag**: enables `EventAugmentor` on GPU (hflip 50%, vflip 30%,
  polarity swap 20%, noise 0.5%, dropout 5%).
- Training windows increased from 164 → **~2270** (11×) via finer dt, more bins, stride
  overlap, and all 5 approach profiles.

### Changed — `snn/models/event_encoder.py`

- Added `spatial_downsample` parameter; `enc_height`, `enc_width`, `_frame_stride` derived
  from pooled dimensions.
- `_encode_columns()`: coordinates clipped to pooled grid (`xs // ds`, `ys // ds`) before
  flat index computation.

### Changed — `snn/models/lgmd_net.py`

- Removed redundant `torch.abs()` on the always-positive `dcmd_weight` buffer.

### Changed — `sim/hover_evasion_capture.py`

- Fixed `globals()["FRAME_DIR"]` hack → direct assignment.
- Fixed trajectory print off-by-one: subsampling now applied before print statement.

### Data regeneration

- **Root cause of ExCorr regression identified**: checkerboard texture material loading
  in Isaac Sim triggers a massive event burst (~100K events/bin) at ~1.45 s post-sim-start,
  regardless of obstacle distance. With 1.5 s warmup this burst occurred 50 ms before launch,
  creating an anti-correlation between event rate and dθ/dt (Pearson = −0.28 in approach window).
- Warmup increased to **3.0 s** (was 1.5 s): 1.55 s gap between texture burst and launch.
- Launch distance increased to **15 m** (was 6 m): texture subtends <1 px at start,
  minimising the burst magnitude.
- Total sim duration extended to **12 s** (was 10 s) to preserve approach + post recording.
- All 5 profiles regenerated with updated config.
- Batch script `sim/run_all_profiles.sh` runs all profiles (sim + v2e) sequentially.

---

## [Unreleased] — Session 3

### Root cause: simulation data quality

Diagnostic analysis of `events.h5` files revealed the training data was fundamentally
flawed in three ways, preventing any positive Pearson correlation from being learned:

1. **Textureless obstacle** — a solid-colour cube only generates events at its perimeter
   edges (~18 px at 5 m). Interior of the approaching face was silent, making the looming
   signal orders of magnitude weaker than expected.
2. **Dynamic environment artifacts** — the "Curved Gridroom" environment has animated
   sky/lighting that produced event spikes up to **581 K events/bin** while the obstacle was
   stationary, dwarfing the 84 events/bin looming signal (which was actually *less* than
   background noise at 91 events/bin, due to the obstacle occluding background texture).
3. **Label misalignment** — `make_label_from_trajectory` used uniform fraction resampling
   (`np.linspace(0,1)`) rather than physical time axes, causing the `dθ/dt` peak to map
   to the wrong bins — especially bad with a 2 s warmup (20% of the recording).

### Changed — `sim/hover_evasion_capture.py`

- **Environment**: switched from `"Curved Gridroom"` to **`"Black Gridroom"`** — static
  grid, no animated sky or lighting, zero background event noise.
- **Checkerboard texture** applied to the obstacle via OmniPBR material:
  - 256×256 image, 8×8 tile grid (32 px tiles), written to `/tmp/obstacle_checker.png`
  - Every contrast boundary across the cube face fires events on approach — dense looming
    signal across the entire obstacle face, not just its edges.
- **Obstacle scale**: `0.5 m → 1.0 m` cube. Larger angular size at launch; more edge
  pixels even before the texture contribution; `OBSTACLE_HALF_SIZE` updated to `0.5 m`.
- **Approach speed**: 5 m/s → **8–10 m/s** across all profiles; launch distance reduced
  from 8 m → 6 m. Obstacle reaches the drone within ~0.6 s of launch, producing a sharp,
  rapid event burst the LGMD can lock on to.
- **Warmup**: `2.0 s → 1.5 s` (initially reduced to 0.5 s, then raised to 1.5 s).
  Later raised to **3.0 s** in Session 4 after diagnosing the texture-activation
  event burst as a fixed-time renderer artifact (~1.45 s post-sim-start).
- **`launch_step` metadata**: `np.int32(WARMUP_STEPS)` now saved to `meta.npz`/`events.h5`
  so the training script can compute physically-aligned labels.
- **Subprocess pass-through**: `--name` flag now forwarded to the v2e subprocess call so
  output directories are consistent end-to-end.
- **Frame extension**: camera frame files now saved as `.bmp` (lossless); `--v2e-only`
  file count corrected to match `.bmp` extension.

### Updated approach profiles

| Profile | Launch pos (m) | Velocity (m/s) |
|---|---|---|
| `head_on` | (6, 0, 1.5) | (-10, 0, 0) |
| `lateral` | (6, 3, 1.5) | (-8, -4, 0) |
| `high` | (6, 0, 3.5) | (-8, 0, -2) |
| `low` | (6, 0, -0.5) | (-8, 0, 2) |
| `diagonal` | (5, 5, 1.5) | (-6, -6, 0) |

### Changed — `snn/training/train_lgmd.py`

- **`make_label_from_trajectory` — physical time alignment**:
  - Now builds a real-time axis `traj_t_s = step × sim_dt` for the trajectory samples.
  - Converts event bin timestamps from µs to seconds: `event_t_s = ts_µs × 1e-6`.
  - Uses `np.interp` on matched physical axes so each event bin receives the `dθ/dt` value
    for the *correct simulation time* — regardless of warmup duration or recording length.
  - Reads `launch_step` from H5 metadata if present (graceful fallback for older files).
- **Recording-level train/val split** via `--val_h5` argument: hold out one full recording
  (default: `diagonal` profile) to catch per-profile overfitting.
- **DCMD normalisation**: divided by fixed pixel count `h×w` instead of per-batch max,
  giving stable gradients across batches.
- **Loss function** (`combined_loss`): window-level Pearson correlation weighted sum of
  `net_exc` (rectified post-inhibition excitation, 80 %) and `dcmd` (20 %), plus a small
  background-suppression L1 penalty on `net_exc`.
- **Validation loop** reports `val_loss`, Pearson `val_corr`, `DCMD_loom`, and `DCMD_bg`
  each epoch for early stopping and diagnosis.

### Changed — `snn/models/lgmd_net.py`

- **LIF threshold**: `v_threshold 1.0 → 0.5` so the neuron fires at physiologically
  plausible input levels (SpikingJelly integrates `input/tau`, not raw `input`).
- **`dcmd_weight`**: converted from learnable `nn.Parameter` → fixed `register_buffer`
  with uniform weights `1/(h×w)`. Prevents the model from overfitting to pixel locations
  in the training set while leaving convolutional layers free to learn features.
- **`forward` return signature** now `(dcmd, spikes, net_exc)`:
  - `net_exc = lgmd_in.clamp(min=0).mean(dim=(-1,-2,-3))` — rectified post-inhibition
    excitation, positive only where looming edges survive lateral suppression; used as the
    primary auxiliary training signal in `combined_loss`.

---

## [Unreleased] — Session 2

### Added
- `sim/hover_evasion_capture.py` — hover-evasion simulation scene
  - Drone holds altitude with P+D controller; `DynamicCuboid` obstacle launched at it after warmup
  - Five named approach profiles: `head_on`, `lateral`, `high`, `low`, `diagonal`
  - Custom approach via `--launch_x/y/z --speed`
  - `--name` flag sets per-profile output directories (prevents frame overwriting between runs)
  - Trajectory metadata (obstacle positions, drone positions, sim_dt) saved to separate `META_DIR`
  - After v2e, trajectory embedded directly into `events.h5` for self-contained training files
- `snn/models/lgmd_net.py` — LGMD-inspired spiking neural network
  - P layer: spatial pooling (AvgPool2d)
  - S layer: excitation conv + delayed lateral inhibition (Gaussian kernel, fixed 1-step delay)
  - LGMD LIF neurons per spatial location
  - DCMD readout: learnable spatial weighting + global sum → collision-imminence spike rate
  - `collision_imminence()` causal smoothing for deployment
- `snn/models/event_encoder.py` — event stream encoder
  - Converts `[timestamp_us, x, y, polarity]` HDF5 events → `(T, 2, H, W)` spike frames
  - Binary and count modes; sliding window extraction
  - `angular_velocity_label()`: analytical `dθ/dt` from known obstacle trajectory
- `snn/training/train_lgmd.py` — LGMD training script
  - Supports multiple `--h5` inputs (datasets concatenated with `ConcatDataset`)
  - Uses analytical `dθ/dt` label when trajectory metadata present, falls back to event-rate heuristic
  - Adam optimiser + cosine LR schedule + gradient clipping
  - 50 epochs on 2 profiles (~164 windows) completes in ~40s on L40s GPU
- `events/visualise_events.py` — event stream visualisation and per-bin stats

### Changed
- Simulation capture FPS raised from 60 → **120 Hz**
  - `rendering_dt = 1/120` passed to Pegasus World settings
  - SuperSloMo upsampling factor reduced from 17× to ~9× — v2e ~2× faster, events more accurate
  - `WARMUP_STEPS` and `TOTAL_STEPS` now derived from `FPS` (`int(2.0 * FPS)`, `int(10.0 * FPS)`)
- `snn/training/train_lgmd.py`: `--h5` now accepts `nargs="+"` for multiple files
- Dataset build: `LoomingDataset` pre-sorts events and uses `np.searchsorted` for O(log N) window lookup (was O(N) Python loop per window — unusable at 16M events)
- `EventEncoder.encode()`: Python for-loop over events replaced with vectorised `np.add.at` on flat index

### Fixed
- **Critical**: `DynamicCuboid` obstacle falls to the floor during warmup under gravity — fixed by applying `PhysxSchema.PhysxRigidBodyAPI` to disable per-body gravity after scene setup
  - All training data generated before this fix should be discarded and regenerated
- `meta.npz` saved inside `FRAME_DIR` caused v2e to read it as a video frame — moved to separate `META_DIR`
- Stale HDF5 file lock from crashed v2e run blocked subsequent runs — must `rm -f events.h5` before retrying
- Trajectory H5 keys mismatched between `hover_evasion_capture.py` (`obstacle_positions`, `drone_hover_position`) and `train_lgmd.py` (was looking for `obstacle_trajectory`, `drone_position`)
- `simulation_app.close()` calls `sys.exit()` — v2e and downstream processing must run as separate process

---

## [0.2.0] — Session 1 (Week 1–2)

### Added
- `sim/headless_hover_test.py` — physics pipeline validation; confirmed Iris quadrotor falls under gravity (~700 steps/sec on L40s)
- `events/capture_and_convert.py` — full sim-to-events pipeline
  - Isaac Sim + Pegasus captures RGB frames at DAVIS346 resolution (346×260, 60 FPS)
  - v2e converts frames to synthetic events with SuperSloMo 17× upsampling (980µs resolution)
  - Static obstacle scene (3× `FixedCuboid` at x=2, 3.5, 5m)
  - `--sim-only` / `--v2e-only` flags to work around `sys.exit()` issue
- `setup.sh` — deployment script for fresh Ubuntu 22.04 + RTX GPU instances (Brev/Crusoe)
- `requirements.txt` — pinned project dependencies

### Fixed
- `MonocularCamera.start()` calls `set_lens_distortion_model` (removed in Isaac Sim 4.5) — fixed via `MonocularCameraIsaacSim45` subclass
- `set_resolution(maintain_square_pixels=True)` kwarg removed in Isaac Sim 4.5 — fixed in same subclass
- `PythonBackend` removed in Pegasus 5.1 — backends now inherit `Backend` base class directly, implementing `update_graphical_sensor`, `start`, `stop`, `reset`
- EULA non-interactive block — fixed via `OMNI_KIT_ACCEPT_EULA=Y` env var + pre-writing `EULA_ACCEPTED` file
- v2e CLI: `--input_dir` not valid — use `-i`; `--dvs_h5 events.h5` for HDF5 output
- v2e requires `tkinter` on headless server — `sudo apt-get install python3-tk`
- SuperSloMo checkpoint download via `gdown --fuzzy` (direct wget returned empty file from Google Drive)

### Infrastructure
- Repo initialised: https://github.com/cyber-run/snn-evs-drone
- Cloud GPU: Brev + Crusoe L40s 48GB, Ubuntu 22.04
  - SSH: `ubuntu@160.211.46.135`
  - Venv: `~/isaaclab-env`
  - Required env: `OMNI_KIT_ACCEPT_EULA=Y ISAACSIM_PATH=~/isaaclab-env/lib/python3.10/site-packages/isaacsim`
- Stack finalised: Isaac Sim 4.5 + Pegasus 5.1.0 (Aerial Gym and Flightmare dropped)

---

### Training results (Session 3 run)

All 5 profiles regenerated and training completed in ~5 min (300 epochs on L40S GPU).

| Metric | Before (bad data) | After (fixed data) |
|---|---|---|
| Val ExCorr (held-out diagonal) | −0.105 | **+0.271** |
| Val DcCorr (held-out diagonal) | −0.105 | **+0.293** |
| Mean ExCorr (5 profiles) | −0.104 | **+0.277** |
| Ex_loom / Ex_bg (head_on) | 0.01× | **2.8×** |
| Ex_loom / Ex_bg (diagonal, val) | 0.01× | **23.3×** |
| Best training ExCorr | stuck | **0.717 @ ep 105** |
| Converged? | No | **Yes** |

The model correctly responds more strongly to the looming phase than background
across all five held-in and the held-out (diagonal) approach profile.

## Known Issues / Next Steps

- Evasion is altitude-only (climb to 5.5 m); lateral evasion direction from DCMD spatial
  activity not yet implemented (left/right split on spike map).
- Single-trial demo — no systematic evaluation across profiles, speeds, or thresholds.
  Next: `scripts/eval_avoidance.py` for N-episode statistics and baseline comparisons.
- Only window-level Pearson loss trained; per-timestep Pearson (used in evaluation) differs
  slightly in scale.
- Sim-to-real gap for event cameras (~50% perf drop reported in literature) — needs early
  characterisation with real Lucid Vision EVS hardware at Imperial.
- Phase 2 (Isaac Lab parallel env): N=256 drones with TiledCamera inline event camera
  for rigorous evaluation and RL training — see `neuromorphic-drone-plan.md`.
