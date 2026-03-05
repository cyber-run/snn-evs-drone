"""
Stage 1: Capture RGB frames from Isaac Sim / Pegasus
Stage 2: Convert frames to synthetic events via v2e

Output:
  /tmp/sim_frames/   - PNG frames from sim
  /tmp/sim_events/   - v2e output (events.npz, video previews)
"""

import time
import os

FRAME_DIR = "/tmp/sim_frames"
EVENT_DIR = "/tmp/sim_events"
NUM_STEPS = 600       # ~10 seconds at 60 fps
RESOLUTION = (346, 260)  # DAVIS346 — standard neuromorphic camera resolution
FPS = 60


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
from scipy.spatial.transform import Rotation
import cv2
done("Imports complete")


class MonocularCameraIsaacSim45(MonocularCamera):
    """MonocularCamera with Isaac Sim 4.5 compatibility fix.
    Pegasus 5.1 calls set_lens_distortion_model which was removed in Isaac Sim 4.5.
    """
    def start(self):
        self._camera.initialize()
        self._camera.set_resolution(self._resolution)
        self._camera.set_clipping_range(*self._clipping_range)
        self._camera.set_frequency(self._frequency)
        if self._depth:
            self._camera.add_distance_to_image_plane_to_frame()
        self._camera_full_set = True


class FrameCaptureBackend(Backend):
    """Backend that captures RGB frames from the onboard camera each step."""

    def __init__(self, frame_dir, max_frames):
        super().__init__(config=BackendConfig())
        os.makedirs(frame_dir, exist_ok=True)
        self.frame_dir = frame_dir
        self.max_frames = max_frames
        self.frame_count = 0
        self.step_count = 0

    def update(self, dt: float):
        self.step_count += 1

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
                bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
                path = os.path.join(self.frame_dir, f"frame_{self.frame_count:06d}.png")
                cv2.imwrite(path, bgr)
                self.frame_count += 1
                if self.frame_count % 60 == 0:
                    tqdm.write(f"  captured frame {self.frame_count}")
        except Exception as e:
            tqdm.write(f"  [warn] frame capture error: {e}")

    def update_state(self, state):
        pass

    def input_reference(self):
        return [0.0, 0.0, 0.0, 0.0]

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        self.step_count = 0
        self.frame_count = 0


def run_simulation():
    os.makedirs(FRAME_DIR, exist_ok=True)

    timeline = omni.timeline.get_timeline_interface()
    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world

    stage("Loading environment")
    pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
    done()

    stage("Spawning quadrotor with camera")
    backend = FrameCaptureBackend(FRAME_DIR, max_frames=NUM_STEPS)
    config = MultirotorConfig()
    config.backends = [backend]
    config.graphical_sensors = [MonocularCameraIsaacSim45("front_camera", config={
        "frequency": FPS,
        "resolution": RESOLUTION,
        "position": np.array([0.15, 0.0, 0.0]),   # forward-facing, nose mount
        "orientation": np.array([0.0, 0.0, 180.0]),
        "depth": False,
    })]

    Multirotor(
        "/World/quadrotor",
        ROBOTS["Iris"],
        0,
        [0.0, 0.0, 1.5],   # spawn 1.5m up so we see the room
        Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
        config=config,
    )
    done("Quadrotor + camera spawned")

    stage("Resetting world and starting physics")
    world.reset()
    timeline.play()
    done()

    stage(f"Running {NUM_STEPS} steps — capturing frames to {FRAME_DIR}")
    for _ in tqdm(range(NUM_STEPS), desc="  Simulating", unit="step", ncols=70):
        world.step(render=True)  # render=True required for camera sensors
    done(f"Captured {backend.frame_count} frames")

    timeline.stop()
    simulation_app.close()
    return backend.frame_count


# ── Stage 2: v2e conversion ───────────────────────────────────────────────────

def run_v2e(num_frames):
    stage(f"Converting {num_frames} frames to events via v2e")
    os.makedirs(EVENT_DIR, exist_ok=True)

    cmd = (
        f"v2e "
        f"-i {FRAME_DIR} "
        f"-o {EVENT_DIR} "
        f"--overwrite "
        f"--timestamp_resolution 0.001 "
        f"--auto_timestamp_resolution false "
        f"--dvs_exposure duration 0.005 "
        f"--input_frame_rate {FPS} "
        f"--no_preview "
        f"--dvs346 "
        f"--output_width {RESOLUTION[0]} "
        f"--output_height {RESOLUTION[1]} "
        f"--dvs_h5 events.h5 "       # raw event stream for SNN training
        f"--batch_size 4 "
        f"2>&1"
    )
    print(f"\n  Running: {cmd}\n")
    ret = os.system(cmd)
    if ret == 0:
        done(f"Events saved to {EVENT_DIR}")
    else:
        print(f"[WARN] v2e exited with code {ret} — check output above")


# ── Main ──────────────────────────────────────────────────────────────────────
# NOTE: simulation_app.close() calls sys.exit(), so v2e must run in a child
# process launched before close(), or as a separate script after this exits.

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-only", action="store_true", help="Only capture frames, skip v2e")
    parser.add_argument("--v2e-only", action="store_true", help="Only run v2e on existing frames")
    args = parser.parse_args()

    if args.v2e_only:
        run_v2e(len(os.listdir(FRAME_DIR)))
    elif args.sim_only:
        run_simulation()
    else:
        # Run sim, then launch v2e in a subprocess before close() exits
        import subprocess, sys
        num_frames = run_simulation()
        if num_frames > 0:
            print(f"\nLaunching v2e as subprocess on {num_frames} frames...")
            subprocess.run([sys.executable, __file__, "--v2e-only"])
