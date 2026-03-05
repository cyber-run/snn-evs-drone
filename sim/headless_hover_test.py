"""
Headless hover test — verifies Pegasus + Isaac Lab physics pipeline.
Spawns a single quadrotor, runs 500 steps, prints position at each 100 steps.
No display required.
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.timeline
from omni.isaac.core.world import World

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.python_backend import PythonBackend

from scipy.spatial.transform import Rotation
import numpy as np
from tqdm import tqdm

NUM_STEPS = 500
PRINT_EVERY = 100


class HoverBackend(PythonBackend):
    """Minimal backend that commands zero velocity — drone should fall under gravity."""

    def __init__(self):
        super().__init__()
        self.step_count = 0

    def update(self, dt: float):
        self.step_count += 1
        if self.step_count % PRINT_EVERY == 0:
            state = self.vehicle.state
            pos = state.position
            print(f"  step {self.step_count:4d} | pos: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}")

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_state(self, state):
        pass

    def input_reference(self):
        # Zero rotor speed reference — drone will fall, confirming gravity is working
        return [0.0, 0.0, 0.0, 0.0]


def main():
    timeline = omni.timeline.get_timeline_interface()

    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world

    pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

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

    world.reset()
    timeline.play()

    print(f"\nRunning {NUM_STEPS} steps (headless)...")
    for i in tqdm(range(NUM_STEPS), desc="Simulating", unit="step"):
        world.step(render=False)

    timeline.stop()
    simulation_app.close()
    print("\nDone. If z decreased over steps, gravity and physics are working.")


if __name__ == "__main__":
    main()
