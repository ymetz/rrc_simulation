import time
import numpy as np

import rrc_simulation

if __name__ == "__main__":

    platform = rrc_simulation.TriFingerPlatform(visualization=True)

    while True:
        position = rrc_simulation.robot_position.gym.sample()
        finger_action = platform.Action(position=position)

        for _ in range(100):
            t = platform.append_desired_action(finger_action)
            time.sleep(platform.get_time_step())

        # show the latest observations
        robot_observation = platform.get_robot_observation(t)
        print("Finger0 Joint Positions: %s" % robot_observation.position[:3])

        cube_pose = platform.get_object_pose(t)
        print("Cube Position (x, y, z): %s" % cube_pose.position)

