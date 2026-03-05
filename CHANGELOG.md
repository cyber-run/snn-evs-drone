# Changelog

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

## Known Issues / Next Session

- **Training data invalid**: all H5 files generated before the gravity fix contain obstacle-on-ground data; regenerate all profiles after pulling latest
- `sim_dt` stored in metadata is `1/FPS` (1/120) but actual trajectory sampling is ~1/240 (physics runs 2 substeps per render step); absolute dθ/dt scale is off by ~2×. Labels are normalised so training is unaffected, but worth fixing for future quantitative analysis
- Only 2 profiles (head_on, lateral) trained on so far — need diagonal, high, low for better generalisation
- No closed-loop evasion controller yet — LGMD output not wired to rotor commands
