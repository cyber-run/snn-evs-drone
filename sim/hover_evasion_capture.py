"""
Hover-evasion scene: stationary quadrotor + dynamic obstacle launched at it.

Generates training data for the LGMD SNN:
  - Drone hovers at a fixed point (altitude PID)
  - One DynamicCuboid is launched toward the drone after a warmup period
  - Camera captures the approaching obstacle (looming stimulus)
  - Obstacle trajectory + drone position saved as metadata

Output:
  /tmp/sim_frames_evasion/          BMP frames (lossless, fast write)
  /tmp/sim_meta_evasion/meta.npz    Trajectory metadata (kept separate so v2e ignores it)
  /tmp/sim_events_evasion/events.h5 Synthetic events with trajectory embedded

Usage:
  # Sim only (choose one of several preset approach profiles):
  python sim/hover_evasion_capture.py --sim-only --profile head_on
  python sim/hover_evasion_capture.py --sim-only --profile lateral
  python sim/hover_evasion_capture.py --sim-only --speed 4.0 --launch_x 8.0 --launch_y 1.5

  # v2e only (after frames exist):
  python sim/hover_evasion_capture.py --v2e-only

  # Full pipeline:
  python sim/hover_evasion_capture.py --profile head_on
"""

import time
import os
import argparse

# ── Constants ─────────────────────────────────────────────────────────────────
# Overridden at runtime by --name argument to support multiple profiles.

_BASE = "/tmp/evasion"
FRAME_DIR  = f"{_BASE}_frames"
META_DIR   = f"{_BASE}_meta"   # separate from frames so v2e ignores it
EVENT_DIR  = f"{_BASE}_events"
RESOLUTION = (346, 260)   # DAVIS346
FPS        = 120          # camera + renderer Hz (physics runs at 250 Hz internally)
                          # 60→17× SloMo, 120→9× SloMo, 240→4× SloMo
SIM_DT     = 1.0 / FPS   # simulated time per step

DRONE_SPAWN   = [0.0, 0.0, 1.5]   # metres — hover target
WARMUP_STEPS  = int(3.0 * FPS)    # 3.0 s warmup: physics settle + USD material propagation
                                  # Checkerboard texture activates at ~1.45s — 3s gives 1.5s gap
TOTAL_STEPS   = int(12.0 * FPS)   # 12 s total: 3s warmup + ~1.5s approach + 7.5s post

# Obstacle half-extents used for angular velocity label computation
# Obstacle scale below is 1.0 m (full side), so half-extent = 0.5 m
OBSTACLE_HALF_SIZE = 0.5           # metres (1.0 m cube)

# Approach profiles: (launch_x, launch_y, launch_z, speed_x, speed_y, speed_z)
# Launch distance 15 m — far enough that the checkerboard texture is sub-pixel
# at start, so the texture-activation event transient falls well within the
# warmup period (before any approach events).  Speeds 8-12 m/s give ~1-2 s
# approach time, producing a strong looming event burst.
PROFILES = {
    "head_on":  (15.0,  0.0,  1.5, -12.0,  0.0,  0.0),
    "lateral":  (15.0,  5.0,  1.5, -10.0, -4.0,  0.0),
    "high":     (15.0,  0.0,  4.0, -10.0,  0.0, -2.0),
    "low":      (15.0,  0.0, -1.0, -10.0,  0.0,  2.0),
    "diagonal": (12.0, 10.0,  1.5,  -8.0, -8.0,  0.0),
}


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] >>> {msg}...", flush=True)


def done(msg="done"):
    print(f"[{time.strftime('%H:%M:%S')}]     {msg}", flush=True)


# ── Stage 1: Simulation ───────────────────────────────────────────────────────

stage("Starting Isaac Sim")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
done("Isaac Sim started")

stage("Importing modules")
import omni.timeline
import numpy as np
from omni.isaac.core.world import World
from tqdm import tqdm

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.backend import Backend, BackendConfig
from omni.isaac.core.objects import DynamicCuboid
from scipy.spatial.transform import Rotation
import cv2
done("Imports complete")


# ── Isaac Sim 4.5 camera compatibility fix ────────────────────────────────────

class MonocularCameraIsaacSim45(MonocularCamera):
    """Pegasus 5.1 / Isaac Sim 4.5 compatibility — removes removed API calls."""
    def start(self):
        self._camera.initialize()
        self._camera.set_resolution(self._resolution)
        self._camera.set_clipping_range(*self._clipping_range)
        self._camera.set_frequency(self._frequency)
        if self._depth:
            self._camera.add_distance_to_image_plane_to_frame()
        self._camera_full_set = True


# ── Altitude PID ──────────────────────────────────────────────────────────────

class HoverController:
    """
    Full position + attitude controller for hover.

    Uses Pegasus's vehicle.force_and_torques_to_velocities() mixer, which
    converts a desired total thrust (N) + torques (Nm) to per-rotor rad/s.
    This avoids manual motor mixing and handles any Iris rotor layout.

    Controller structure (same as Pegasus nonlinear_controller.py):
      - Outer loop:  position/velocity PID → desired total thrust u1
      - Inner loop:  SO(3) attitude PD    → desired torques tau
    """

    # Iris physical parameters (Pegasus default)
    MASS = 1.50     # kg
    G    = 9.81     # m/s²

    # Outer-loop gains (position PID in z only; x/y held by attitude loop)
    KP_POS = np.array([10.0, 10.0, 10.0])
    KD_POS = np.array([ 8.5,  8.5,  8.5])
    KI_POS = np.array([ 1.5,  1.5,  1.5])

    # Inner-loop gains (attitude PD on SO3 error)
    KR     = np.array([3.5, 3.5, 3.5])
    KW     = np.array([0.5, 0.5, 0.5])

    # Clamp total thrust to sane range (N)
    THRUST_MIN = 0.0
    THRUST_MAX = 60.0   # ~4× Iris hover thrust

    def __init__(self, target_pos: np.ndarray):
        from scipy.spatial.transform import Rotation as _R
        self._Rotation = _R
        self.target_pos = np.array(target_pos, dtype=float)
        self._int       = np.zeros(3)

        # State (updated by update_state callback)
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.R   = _R.identity()
        self.w   = np.zeros(3)

    def update_state(self, state):
        """Called by HoverEvasionBackend.update_state()."""
        self.pos = np.array(state.position,        dtype=float)
        self.vel = np.array(state.linear_velocity, dtype=float)
        self.w   = np.array(state.angular_velocity, dtype=float)
        self.R   = self._Rotation.from_quat(state.attitude)

    def compute(self, dt: float, evasion_force: float = 0.0):
        """
        Return (u1, tau) for vehicle.force_and_torques_to_velocities().
          u1           – desired total thrust [N]
          tau (3,)     – desired torques [roll, pitch, yaw] [Nm]
          evasion_force – extra upward thrust [N] during evasion
        """
        # ── Outer loop: position PID ──────────────────────────────────────────
        ep = self.pos - self.target_pos          # position error (want to zero)
        ev = self.vel                             # velocity error (target vel=0)

        if dt > 0:
            self._int += ep * dt
        self._int = np.clip(self._int, -5.0, 5.0)

        F_des = (- self.KP_POS * ep
                 - self.KD_POS * ev
                 - self.KI_POS * self._int
                 + np.array([0.0, 0.0, self.MASS * self.G]))
        F_des[2] += evasion_force

        # ── Desired thrust = projection onto current body Z ───────────────────
        Z_B = self.R.as_matrix()[:, 2]
        u1  = float(np.dot(F_des, Z_B))
        u1  = float(np.clip(u1, self.THRUST_MIN, self.THRUST_MAX))

        # ── Inner loop: SO(3) attitude control ───────────────────────────────
        # Desired attitude: align Z_B with F_des, yaw = 0
        F_norm = np.linalg.norm(F_des)
        if F_norm < 1e-3:
            Z_b_des = np.array([0.0, 0.0, 1.0])
        else:
            Z_b_des = F_des / F_norm

        X_c_des = np.array([1.0, 0.0, 0.0])       # desired heading (yaw=0)
        Z_cross  = np.cross(Z_b_des, X_c_des)
        Z_cross_n = np.linalg.norm(Z_cross)
        if Z_cross_n < 1e-3:
            Y_b_des = np.array([0.0, 1.0, 0.0])
        else:
            Y_b_des = Z_cross / Z_cross_n
        X_b_des = np.cross(Y_b_des, Z_b_des)

        R_des = np.column_stack([X_b_des, Y_b_des, Z_b_des])
        R_cur = self.R.as_matrix()

        # Rotation error (Mellinger/Lee vee-map)
        eR_mat = 0.5 * (R_des.T @ R_cur - R_cur.T @ R_des)
        e_R    = np.array([eR_mat[2, 1], eR_mat[0, 2], eR_mat[1, 0]])
        e_w    = self.w                              # desired angular velocity = 0

        tau = - self.KR * e_R - self.KW * e_w

        return u1, tau


# ── Backend ───────────────────────────────────────────────────────────────────

class HoverEvasionBackend(Backend):
    """
    Backend that:
      - Holds position with a full position+attitude controller
      - Captures RGB frames every sim step
      - Records obstacle and drone positions every step
      - (optional) runs live SNN evasion via log-diff event camera
    """

    # Evasion: max-thrust burst on trigger, then PID to new altitude
    # At ~240Hz physics, 72 steps ≈ 0.3s. With 41N max thrust and 1.5kg:
    # net upward accel ≈ (41-14.7)/1.5 = 17.5 m/s² → ~0.79m climb in 0.3s
    EVASION_BURST_STEPS = 72   # physics steps of max-thrust burst

    def __init__(self, frame_dir: str, max_frames: int, target_z: float,
                 evasion_model=None, n_bins: int = 20,
                 dcmd_threshold: float = 0.3, pool_factor: int = 4,
                 log_diff_threshold: float = 0.15):
        super().__init__(config=BackendConfig())
        os.makedirs(frame_dir, exist_ok=True)
        self.frame_dir = frame_dir
        self.max_frames = max_frames

        self.ctrl = HoverController(target_pos=np.array(DRONE_SPAWN))
        self._rotor_cmd = [0.0] * 4

        self.frame_count = 0
        self.step_count = 0

        # Trajectory recording — filled during simulation
        self.drone_positions    = []   # (T, 3)
        self.obstacle_positions = []   # (T, 3) — None until obstacle assigned
        self._obstacle = None          # DynamicCuboid reference, set externally

        # ── Live SNN evasion ─────────────────────────────────────────────────
        self._evasion_model = evasion_model
        self._n_bins = n_bins
        self._pool_factor = pool_factor
        self._log_diff_thresh = log_diff_threshold
        self._dcmd_threshold  = dcmd_threshold

        self._prev_log_lum    = None   # (H, W) float32
        self._event_buffer    = []     # rolling list of (2, Hp, Wp) tensors
        self._dcmd_history    = []     # (step, dcmd_value) log
        self._evading              = False
        self._evasion_step         = None
        self._evasion_time         = None
        self._evasion_dcmd         = None
        self._evasion_burst_remain = 0   # physics steps left in max-thrust burst
        self._closest_dist         = float("inf")
        self._sim_time             = 0.0

    def update_state(self, state):
        pass  # Not called reliably in Isaac Sim 4.5 — state read directly in update()

    # Called every physics step
    def update(self, dt: float):
        self.step_count += 1
        self._sim_time += dt

        # Read state directly from vehicle (reliable in Isaac Sim 4.5)
        state = self.vehicle.state
        pos = np.array(state.position, dtype=float)
        self.ctrl.pos = pos
        self.ctrl.vel = np.array(state.linear_velocity, dtype=float)
        self.ctrl.w   = np.array(state.angular_velocity, dtype=float)
        try:
            from scipy.spatial.transform import Rotation as _R
            self.ctrl.R = _R.from_quat(state.attitude)
        except Exception:
            pass

        # Record drone position
        self.drone_positions.append(pos.copy())

        # Record obstacle position (zeros before launch)
        if self._obstacle is not None:
            try:
                obs_pos, _ = self._obstacle.get_world_pose()
                self.obstacle_positions.append(np.array(obs_pos))
                # Track closest approach for evasion evaluation
                dist = float(np.linalg.norm(np.array(obs_pos) - pos))
                self._closest_dist = min(self._closest_dist, dist)
            except Exception:
                self.obstacle_positions.append(np.zeros(3))
        else:
            self.obstacle_positions.append(np.zeros(3))

        # ── Run SNN evasion check (post-warmup only, keyed on elapsed time) ──
        warmup_s = WARMUP_STEPS * SIM_DT           # 3.0 s
        if (self._evasion_model is not None
                and self._sim_time > warmup_s
                and len(self._event_buffer) == self._n_bins):
            import torch
            with torch.no_grad():
                x = torch.stack(self._event_buffer).unsqueeze(1)  # (T, 1, 2, Hp, Wp)
                dcmd, _, _ = self._evasion_model(x)
                imminence = float(dcmd[-1, 0])
            self._dcmd_history.append((self._sim_time, imminence))

            if imminence > self._dcmd_threshold and not self._evading:
                self._evading              = True
                self._evasion_step         = self.step_count
                self._evasion_time         = self._sim_time
                self._evasion_dcmd         = imminence
                self._evasion_burst_remain = self.EVASION_BURST_STEPS
                # Set new hover target above the obstacle path (PID takes over after burst)
                self.ctrl.target_pos[2] = DRONE_SPAWN[2] + 4.0
                self.ctrl._int[2] = 0.0   # reset altitude integral for clean transition
                tqdm.write(f"  [t={self._sim_time:.2f}s] "
                           f"EVASION triggered! DCMD={imminence:.4f} → max-thrust burst + target z={self.ctrl.target_pos[2]:.1f}m")

        # Compute desired thrust + torques, convert to rotor speeds
        if self._evasion_burst_remain > 0:
            # Reflex burst: directly command all rotors to max speed
            self._evasion_burst_remain -= 1
            max_speed = 1100.0   # Iris max rotor velocity (rad/s)
            self._rotor_cmd = [max_speed, max_speed, max_speed, max_speed]
        else:
            u1, tau = self.ctrl.compute(dt)          # normal position+attitude PID
            if self.vehicle is not None:
                self._rotor_cmd = self.vehicle.force_and_torques_to_velocities(u1, tau)


    def update_sensor(self, sensor_type: str, data):
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        if sensor_type != "MonocularCamera" or data is None:
            return
        if self.frame_count >= self.max_frames:
            return
        try:
            cam = data.get("camera")
            if cam is None:
                return
            rgb = cam.get_rgb()
            if rgb is not None and rgb.size > 0:
                # Save frame for v2e (skip in evasion-only mode)
                if self.frame_dir and not getattr(self, "_evasion_only", False):
                    bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
                    path = os.path.join(self.frame_dir,
                                        f"frame_{self.frame_count:06d}.bmp")
                    cv2.imwrite(path, bgr)
                self.frame_count += 1
                if self.frame_count % 60 == 0:
                    tqdm.write(f"  captured frame {self.frame_count}")

                # ── Live log-diff event camera for evasion SNN ───────────────
                if self._evasion_model is not None:
                    import torch, torch.nn.functional as TF
                    frame_f = rgb[..., :3].astype(np.float32) / 255.0
                    lum = (0.299 * frame_f[..., 0]
                           + 0.587 * frame_f[..., 1]
                           + 0.114 * frame_f[..., 2])
                    log_lum = np.log(lum + 1e-6)          # (H, W)
                    if self._prev_log_lum is not None:
                        diff = log_lum - self._prev_log_lum
                        on  = (diff >  self._log_diff_thresh).astype(np.float32)
                        off = (diff < -self._log_diff_thresh).astype(np.float32)
                        ev_frame = torch.from_numpy(
                            np.stack([on, off], axis=0))   # (2, H, W)
                        # Spatial pool to match model input (H//4, W//4)
                        if self._pool_factor > 1:
                            ev_frame = TF.avg_pool2d(
                                ev_frame.unsqueeze(0),
                                self._pool_factor).squeeze(0)
                        self._event_buffer.append(ev_frame)
                        if len(self._event_buffer) > self._n_bins:
                            self._event_buffer.pop(0)
                    self._prev_log_lum = log_lum
        except Exception as e:
            tqdm.write(f"  [warn] frame capture: {e}")

    def update_state(self, state):
        pass

    def input_reference(self):
        return self._rotor_cmd

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        self.step_count = 0
        self.frame_count = 0
        self._sim_time  = 0.0
        self.drone_positions.clear()
        self.obstacle_positions.clear()
        self._rotor_cmd              = [0.0] * 4
        self._evasion_burst_remain   = 0
        self.ctrl._int               = np.zeros(3)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(args):
    os.makedirs(FRAME_DIR, exist_ok=True)

    # ── Load SNN evasion model if --evasion flag given ────────────────────────
    evasion_model = None
    if getattr(args, "evasion", False) and args.weights:
        import torch, sys as _sys
        _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
        from snn.models.lgmd_net import LGMDNet
        H_enc = RESOLUTION[1] // 4   # 260 // 4 = 65
        W_enc = RESOLUTION[0] // 4   # 346 // 4 = 87
        evasion_model = LGMDNet(height=H_enc, width=W_enc, pool_factor=1)
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        evasion_model.load_state_dict(state)
        evasion_model.eval()
        tqdm.write(f"  Evasion model loaded from {args.weights} "
                   f"(threshold={args.dcmd_threshold:.2f})")

    # Resolve approach profile
    if args.profile and args.profile in PROFILES:
        lx, ly, lz, vx, vy, vz = PROFILES[args.profile]
        tqdm.write(f"  Using profile '{args.profile}': "
                   f"launch=({lx},{ly},{lz})  vel=({vx},{vy},{vz})")
    else:
        lx, ly, lz = args.launch_x, args.launch_y, args.launch_z
        # Compute velocity aimed at drone spawn point
        direction = np.array(DRONE_SPAWN) - np.array([lx, ly, lz])
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        vx, vy, vz = (direction * args.speed).tolist()
        tqdm.write(f"  Custom launch=({lx:.1f},{ly:.1f},{lz:.1f})  "
                   f"vel=({vx:.2f},{vy:.2f},{vz:.2f})")

    launch_pos   = np.array([lx, ly, lz])
    launch_vel   = np.array([vx, vy, vz])

    timeline = omni.timeline.get_timeline_interface()
    pg = PegasusInterface()
    # Match rendering rate to camera FPS (physics stays at 250 Hz internally)
    pg._world_settings["rendering_dt"] = 1.0 / FPS
    pg._world = World(**pg._world_settings)
    world = pg.world

    stage("Loading environment")
    # Black Gridroom: static grid pattern, no dynamic/animated elements.
    # Curved Gridroom had animated sky/lighting that caused massive background
    # event spikes (~580 K events/bin) unrelated to the looming obstacle.
    pg.load_environment(SIMULATION_ENVIRONMENTS["Black Gridroom"])
    done()

    # ── Generate a high-contrast checkerboard texture for the obstacle ──────
    # A uniform solid-color obstacle generates events ONLY at its expanding
    # edges (~18 pixels at 5 m distance).  A checkerboard creates sharp
    # contrast boundaries that fire events across the ENTIRE face of the cube
    # as it moves, giving orders-of-magnitude more looming signal.
    import cv2 as _cv2
    _checker_path = "/tmp/obstacle_checker.png"
    if not os.path.exists(_checker_path):
        _img = np.zeros((256, 256, 3), dtype=np.uint8)
        _tile = 32  # 8×8 grid of tiles
        for _i in range(256 // _tile):
            for _j in range(256 // _tile):
                if (_i + _j) % 2 == 0:
                    _img[_i*_tile:(_i+1)*_tile, _j*_tile:(_j+1)*_tile] = 230
        _cv2.imwrite(_checker_path, _img)
    done(f"Checkerboard texture: {_checker_path}")

    stage("Adding dynamic obstacle (gravity disabled — floats until launched)")
    obstacle = world.scene.add(DynamicCuboid(
        prim_path="/World/obstacle",
        name="obstacle",
        position=launch_pos,
        scale=np.array([1.0, 1.0, 1.0]),   # 1 m cube — larger = more edge pixels
        color=np.array([0.9, 0.9, 0.9]),   # will be overridden by texture below
        mass=1.0,
    ))

    # Apply checkerboard texture via OmniPBR material
    try:
        from pxr import UsdShade, Sdf
        _stage_usd = world.stage
        _mat = UsdShade.Material.Define(_stage_usd, "/World/ObstacleMat")
        _sh  = UsdShade.Shader.Define(_stage_usd, "/World/ObstacleMat/Shader")
        _sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        _sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        _sh.CreateInput("diffuse_texture",
                        Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(_checker_path))
        _mat.CreateSurfaceOutput().ConnectToSource(_sh.ConnectableAPI(), "surface")
        from pxr import UsdShade as _US
        _US.MaterialBindingAPI.Apply(
            _stage_usd.GetPrimAtPath("/World/obstacle")
        ).Bind(_mat)
        done("Checkerboard material applied to obstacle")
    except Exception as _e:
        done(f"[warn] Could not apply texture ({_e}); using solid color")
    # Disable gravity so the obstacle stays at launch height during warmup.
    # PhysxSchema is the correct Isaac Sim 4.5 API for per-body gravity control.
    from pxr import PhysxSchema
    obstacle_prim = world.stage.GetPrimAtPath("/World/obstacle")
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(obstacle_prim)
    physx_rb.GetDisableGravityAttr().Set(True)
    done("Obstacle added (gravity disabled)")

    stage("Spawning quadrotor with camera")
    backend = HoverEvasionBackend(FRAME_DIR, max_frames=TOTAL_STEPS,
                                  target_z=DRONE_SPAWN[2],
                                  evasion_model=evasion_model,
                                  dcmd_threshold=getattr(args, "dcmd_threshold", 0.3))
    backend._obstacle = obstacle

    config = MultirotorConfig()
    config.backends = [backend]
    config.graphical_sensors = [MonocularCameraIsaacSim45("front_camera", config={
        "frequency": FPS,
        "resolution": RESOLUTION,
        "position": np.array([0.15, 0.0, 0.0]),    # nose mount, forward-facing
        "orientation": np.array([0.0, 0.0, 180.0]),
        "depth": False,
    })]

    Multirotor(
        "/World/quadrotor",
        ROBOTS["Iris"],
        0,
        DRONE_SPAWN,
        Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
        config=config,
    )
    done("Quadrotor + camera spawned")

    stage("Resetting world and starting physics")
    world.reset()
    timeline.play()
    done()

    obstacle_launched = False

    stage(f"Running {TOTAL_STEPS} steps ({TOTAL_STEPS / FPS:.0f}s) — "
          f"obstacle launches at step {WARMUP_STEPS}")

    for step in tqdm(range(TOTAL_STEPS), desc="  Simulating", unit="step", ncols=70):

        # Launch obstacle after warmup — apply velocity once
        if step == WARMUP_STEPS and not obstacle_launched:
            try:
                obstacle.set_linear_velocity(launch_vel)
                obstacle_launched = True
                tqdm.write(f"  [step {step}] Obstacle launched — "
                           f"vel=({launch_vel[0]:.1f}, {launch_vel[1]:.1f}, {launch_vel[2]:.1f}) m/s")
            except Exception as e:
                tqdm.write(f"  [warn] could not set obstacle velocity: {e}")

        world.step(render=True)

    raw_traj_len = len(backend.drone_positions)
    done(f"Captured {backend.frame_count} frames  |  "
         f"{raw_traj_len} trajectory points (raw physics rate)")

    # ── Evasion summary ───────────────────────────────────────────────────────
    if backend._evasion_model is not None:
        if backend._evading:
            result = ("MISS" if backend._closest_dist > 0.6 else "HIT")
            tqdm.write(f"\n  === EVASION RESULT: {result} ===")
            tqdm.write(f"  Evasion triggered at t={backend._evasion_time:.2f}s  "
                       f"DCMD={backend._evasion_dcmd:.4f}")
            tqdm.write(f"  Closest approach: {backend._closest_dist:.2f}m")
        else:
            tqdm.write(f"\n  === NO EVASION triggered (DCMD never exceeded {backend._dcmd_threshold:.2f}) ===")
            tqdm.write(f"  Closest approach: {backend._closest_dist:.2f}m")

    # Save trajectory metadata to a separate directory so v2e doesn't read it as a frame
    os.makedirs(META_DIR, exist_ok=True)
    meta_path = os.path.join(META_DIR, "meta.npz")
    drone_pos_arr = np.array(backend.drone_positions, dtype=np.float32)
    obs_pos_arr   = np.array(backend.obstacle_positions, dtype=np.float32)

    # Pegasus calls backend.update() at the physics substep rate, which is a multiple
    # of the render rate.  Subsample the trajectory down to the RENDER rate (TOTAL_STEPS
    # points) so that traj[i] corresponds to frame i and sim_dt (= 1/FPS) is correct.
    physics_factor = max(1, round(raw_traj_len / TOTAL_STEPS))
    if physics_factor > 1:
        obs_pos_arr   = obs_pos_arr[::physics_factor]
        drone_pos_arr = drone_pos_arr[::physics_factor]
        tqdm.write(f"  Subsampled trajectory {physics_factor}× "
                   f"({raw_traj_len} → {len(obs_pos_arr)} points)")

    # Hover position: mean of first WARMUP_STEPS render frames (before obstacle launches)
    stable_pos = drone_pos_arr[:WARMUP_STEPS].mean(axis=0)

    np.savez(meta_path,
             drone_positions=drone_pos_arr,
             obstacle_positions=obs_pos_arr,
             drone_hover_position=stable_pos,
             obstacle_radius=np.float32(OBSTACLE_HALF_SIZE),
             sim_dt=np.float32(SIM_DT),
             warmup_steps=np.int32(WARMUP_STEPS),
             launch_step=np.int32(WARMUP_STEPS),
             launch_velocity=launch_vel.astype(np.float32))

    done(f"Trajectory metadata saved to {meta_path}")
    tqdm.write(f"  Drone hover z: {stable_pos[2]:.3f}m  "
               f"(target {DRONE_SPAWN[2]}m)")

    # Print a few trajectory samples for debugging
    check_steps = [0, WARMUP_STEPS//4, WARMUP_STEPS//2, WARMUP_STEPS - 1, WARMUP_STEPS]
    for s in check_steps:
        if s < len(drone_pos_arr):
            dp = drone_pos_arr[s]
            op = obs_pos_arr[s] if s < len(obs_pos_arr) else np.zeros(3)
            tqdm.write(f"  step {s:4d} t={s*SIM_DT:.2f}s  "
                       f"drone=({dp[0]:.2f},{dp[1]:.2f},{dp[2]:.2f})  "
                       f"obs=({op[0]:.1f},{op[1]:.1f},{op[2]:.1f})")

    timeline.stop()

    # ── Optional video export ─────────────────────────────────────────────────
    if getattr(args, "video", False) and backend.frame_count > 0:
        annotation = args.profile or run_name
        if backend._evasion_model is not None:
            if backend._evading:
                verdict = "MISS" if backend._closest_dist > 0.6 else "HIT"
                annotation += f" | SNN evasion | {verdict} {backend._closest_dist:.2f}m"
            else:
                annotation += f" | SNN evasion | no trigger"
        else:
            annotation += f" | baseline | closest {backend._closest_dist:.2f}m"
        video_path = f"/tmp/evasion_{run_name}_video.mp4"
        make_video(FRAME_DIR, video_path, annotation=annotation)

    simulation_app.close()
    return backend.frame_count


# ── Video export ──────────────────────────────────────────────────────────────

def make_video(frame_dir: str, out_path: str,
               in_fps: int = FPS, out_fps: int = 30,
               annotation: str = "") -> bool:
    """
    Stitch BMP frames to MP4 via ffmpeg.

    Args:
        frame_dir:   directory containing frame_XXXXXX.bmp files
        out_path:    output .mp4 path
        in_fps:      input (simulation) framerate
        out_fps:     output video framerate (30 gives smooth playback)
        annotation:  text overlay in top-left corner (profile, result, etc.)
    """
    import subprocess

    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".bmp"))
    if not frames:
        print(f"[WARN] No frames found in {frame_dir} — skipping video")
        return False

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    vf_parts = [f"fps={out_fps}"]
    if annotation:
        safe = annotation.replace("'", "\\'").replace(":", "\\:")
        vf_parts.append(
            f"drawtext=text='{safe}':fontsize=16:fontcolor=white"
            f":x=10:y=10:box=1:boxcolor=black@0.6:boxborderw=4"
        )

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(in_fps),
        "-i", os.path.join(frame_dir, "frame_%06d.bmp"),
        "-vf", ",".join(vf_parts),
        "-c:v", "libx264", "-crf", "22", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] ffmpeg failed:\n{result.stderr[-500:]}")
        return False
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Video saved → {out_path}  ({size_mb:.1f} MB)")
    return True


# ── Stage 2: v2e conversion ───────────────────────────────────────────────────

def run_v2e(num_frames):
    stage(f"Converting {num_frames} frames to events via v2e (SuperSloMo)")
    os.makedirs(EVENT_DIR, exist_ok=True)

    slomo_ckpt = os.path.expanduser("~/v2e/input/SuperSloMo39.ckpt")
    if not os.path.exists(slomo_ckpt):
        print(f"[WARN] SuperSloMo checkpoint not found at {slomo_ckpt}")
        print("       Run with --disable_slomo or download with:")
        print("       gdown --fuzzy 'https://drive.google.com/file/d/"
              "19YDLygMkXey4ePj8_W54BVlkKxTxWiEk' -O ~/v2e/input/SuperSloMo39.ckpt")
        slomo_flag = "--disable_slomo"
    else:
        slomo_flag = f"--slomo_model {slomo_ckpt}"

    cmd = (
        f"v2e "
        f"-i {FRAME_DIR} "
        f"-o {EVENT_DIR} "
        f"--overwrite "
        f"{slomo_flag} "
        f"--timestamp_resolution 0.001 "
        f"--auto_timestamp_resolution false "
        f"--dvs_exposure duration 0.005 "
        f"--input_frame_rate {FPS} "
        f"--no_preview "
        f"--dvs346 "
        f"--output_width {RESOLUTION[0]} "
        f"--output_height {RESOLUTION[1]} "
        f"--dvs_h5 events.h5 "
        f"--batch_size 64 "
        f"2>&1"
    )
    print(f"\n  Running: {cmd}\n")
    ret = os.system(cmd)
    if ret != 0:
        print(f"[WARN] v2e exited with code {ret}")
        return

    done(f"Events saved to {EVENT_DIR}/events.h5")
    _embed_trajectory(os.path.join(EVENT_DIR, "events.h5"))


def _embed_trajectory(h5_path: str):
    """Merge trajectory metadata from meta.npz into the events.h5 file."""
    import h5py

    meta_path = os.path.join(META_DIR, "meta.npz")
    if not os.path.exists(meta_path):
        print("[WARN] meta.npz not found — trajectory data will not be in events.h5")
        return

    meta = np.load(meta_path)
    with h5py.File(h5_path, "a") as f:
        for key in meta.files:
            if key in f:
                del f[key]
            f.create_dataset(key, data=meta[key])

    done(f"Trajectory metadata embedded into {h5_path}")
    print(f"  Keys: {list(meta.files)}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hover-evasion scene: drone holds position, obstacle launched at it."
    )
    parser.add_argument("--sim-only", action="store_true",
                        help="Run simulation only (skip v2e)")
    parser.add_argument("--v2e-only", action="store_true",
                        help="Run v2e only (frames must already exist)")

    # Per-run name: sets output dirs to /tmp/evasion_<name>_{frames,meta,events}
    parser.add_argument("--name", default=None,
                        help="Run name (e.g. 'head_on', 'lateral') — sets output dirs")

    # Approach profile (overrides manual settings if given)
    parser.add_argument("--profile", choices=list(PROFILES.keys()),
                        help="Named approach profile")

    # Manual approach settings (used if no --profile)
    parser.add_argument("--launch_x", type=float, default=8.0,
                        help="Obstacle spawn X (metres, default 8m ahead)")
    parser.add_argument("--launch_y", type=float, default=0.0)
    parser.add_argument("--launch_z", type=float, default=1.5)
    parser.add_argument("--speed",    type=float, default=5.0,
                        help="Approach speed in m/s (velocity aimed at drone)")

    # Evasion mode
    parser.add_argument("--evasion", action="store_true",
                        help="Enable closed-loop SNN evasion (requires --weights)")
    parser.add_argument("--weights", default=None,
                        help="Path to trained LGMDNet weights (.pt)")
    parser.add_argument("--dcmd_threshold", type=float, default=0.3,
                        help="DCMD threshold to trigger evasion (default 0.3)")
    parser.add_argument("--video", action="store_true",
                        help="Export MP4 video from captured frames after simulation")

    args = parser.parse_args()

    # Override output dirs based on --name (or auto-derive from --profile)
    run_name = args.name or args.profile or "default"
    FRAME_DIR = f"/tmp/evasion_{run_name}_frames"
    META_DIR  = f"/tmp/evasion_{run_name}_meta"
    EVENT_DIR = f"/tmp/evasion_{run_name}_events"
    print(f"Output dirs: frames={FRAME_DIR}  meta={META_DIR}  events={EVENT_DIR}")

    if args.v2e_only:
        run_v2e(len([f for f in os.listdir(FRAME_DIR)
                     if f.endswith(".bmp")]) if os.path.exists(FRAME_DIR) else 0)
    elif args.sim_only:
        run_simulation(args)
    else:
        import subprocess, sys
        num_frames = run_simulation(args)
        if num_frames > 0:
            print(f"\nLaunching v2e subprocess on {num_frames} frames...")
            subprocess.run([sys.executable, __file__,
                            "--v2e-only", "--name", run_name])
