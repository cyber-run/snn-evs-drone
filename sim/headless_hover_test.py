"""
Headless hover test — verifies Pegasus + Isaac Lab physics pipeline.
Spawns a single quadrotor, runs 500 steps, prints position at each 100 steps.
No display required.
"""

import time

def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] >>> {msg}...", flush=True)

def done(msg="done"):
    print(f"[{time.strftime('%H:%M:%S')}]     {msg}", flush=True)


stage("Starting Isaac Sim (slowest step, ~60-120s)")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
done("Isaac Sim started")

stage("Importing Isaac Lab / Pegasus modules")
import omni.timeline
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.python_backend import PythonBackend
from scipy.spatial.transform import Rotation
import numpy as np
from tqdm import tqdm
done("Imports complete")

NUM_STEPS = 500
PRINT_EVERY = 100


class HoverBackend(PythonBackend):
    """Minimal backend — zero rotor commands so drone falls under gravity."""

    def __init__(self):
        super().__init__()
        self.step_count = 0

    def update(self, dt: float):
        self.step_count += 1
        if self.step_count % PRINT_EVERY == 0:
            state = self.vehicle.state
            pos = state.position
            tqdm.write(f"  step {self.step_count:4d} | pos: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}")

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_state(self, state):
        pass

    def input_reference(self):
        return [0.0, 0.0, 0.0, 0.0]


def main():
    stage("Initialising timeline and Pegasus interface")
    timeline = omni.timeline.get_timeline_interface()
    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world
    done()

    stage("Loading environment (Curved Gridroom)")
    pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
    done("Environment loaded")

    stage("Spawning quadrotor")
    backend = HoverBackend()
    config = MultirotorConfig()
    config.backends = [backend]
    Multirotor(
        "/World/quadrotor",
        ROBOTS["Iris"],
        0,
        [0.0, 0.0, 0.5],
        Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
        config=config,
    )
    done("Quadrotor spawned")

    stage("Resetting world and starting physics")
    world.reset()
    timeline.play()
    done("Physics running")

    print(f"\n[{time.strftime('%H:%M:%S')}] Running {NUM_STEPS} simulation steps...")
    for i in tqdm(range(NUM_STEPS), desc="  Simulating", unit="step", ncols=70):
        world.step(render=False)

    stage("Shutting down")
    timeline.stop()
    simulation_app.close()
    done("Complete — if z decreased over steps, gravity and physics are confirmed working")


if __name__ == "__main__":
    main()
