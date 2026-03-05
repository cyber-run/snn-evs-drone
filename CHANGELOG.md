# Changelog

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
- **Warmup**: `2.0 s → 0.5 s` (just enough for physics to settle). Reduces the proportion
  of stationary-obstacle frames from 20 % to ~6 %, so the looming phase dominates.
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

## Known Issues / Next Session

- **Trajectory double-sampling**: The training script now auto-detects and corrects this
  (1922 raw physics points → correct dt=1/240s). The sim script will be updated to
  subsample to render rate before saving in the next regeneration run.
- **Pre-launch texture spike (head_on, t≈0.4s)**: Checkerboard texture activates visually
  ~0.1s before the warmup ends, causing 416K-event background spike. Increasing warmup to
  1.5s (already done in sim script, takes effect on next regeneration) will push this spike
  safely into early warmup frames.
- `head_on` profile shows weaker correlation (ExCorr=0.074) than other profiles — likely
  due to the texture activation spike and the head-on approach producing a more symmetric
  (hence harder to separate) event pattern.
- No closed-loop evasion controller yet — LGMD output not wired to rotor commands.
- Only window-level Pearson loss trained; per-timestep Pearson (used in evaluation) differs
  slightly in scale, explaining the gap between training-log ExCorr (0.717) and eval ExCorr (0.277).
