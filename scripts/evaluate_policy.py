#!/usr/bin/env python3
"""Example evaluation script to evaluate a policy.

This is an example evaluation script for evaluating a "RandomPolicy".  Use this
as a base for your own script to evaluate your policy.  All you need to do is
to replace the `RandomPolicy` and potentially the Gym environment with your own
ones (see the TODOs in the code below).

This script will be executed in an automated procedure.  For this to work, make
sure you do not change the overall structure of the script!

This script expects the following arguments in the given order:
 - Difficulty level (needed for reward computation)
 - initial pose of the cube (as JSON string)
 - goal pose of the cube (as JSON string)
 - file to which the action log is written

It is then expected to initialize the environment with the given initial pose
and execute exactly one episode with the policy that is to be evaluated.

When finished, the action log, which is created by the TriFingerPlatform class,
is written to the specified file.  This log file is crucial as it is used to
evaluate the actual performance of the policy.
"""
import sys

import gym
import numpy as np

from rrc_simulation.gym_wrapper.envs.cube_env_modified import CubeEnv, ActionType, RandomInitializer, ObservationType, \
    FlatObservationWrapper, GoalObservationWrapper, FixedInitializer
from rrc_simulation.tasks import move_cube
from rrc_simulation.gym_wrapper.wrappers import TimeFeatureWrapper, FrameStackWrapper

from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines import SAC
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main():
    try:
        difficulty = int(sys.argv[1])
        initial_pose_json = sys.argv[2]
        goal_pose_json = sys.argv[3]
        output_file = sys.argv[4]
    except IndexError:
        print("Incorrect number of arguments.")
        print(
            "Usage:\n"
            "\tevaluate_policy.py <difficulty_level> <initial_pose>"
            " <goal_pose> <output_file>"
        )
        sys.exit(1)

    # the poses are passes as JSON strings, so they need to be converted first
    initial_pose = move_cube.Pose.from_json(initial_pose_json)
    goal_pose = move_cube.Pose.from_json(goal_pose_json)

    # create a FixedInitializer with the given values
    initializer = FixedInitializer(
        difficulty, initial_pose, goal_pose
    )

    # TODO: Replace with your environment if you used a custom one.
    env = CubeEnv(frameskip=3,
                      visualization=False,
                      initializer=initializer,
                      action_type=ActionType.POSITION,
                      observation_type=ObservationType.WITHOUT_GOALS)
    env = TimeFeatureWrapper(FlatObservationWrapper(env))

    if difficulty == 2:
        norm_env = VecNormalize.load("models/normalized_env_09_18_2020_01_08_25_", DummyVecEnv([lambda: env]))
    else:
        print("load push model")
        norm_env = VecNormalize.load("models/normalized_env_09_17_2020_22_04_30_", DummyVecEnv([lambda: env]))

    # TODO: Replace this with your model
    # Note: You may also use a different policy for each difficulty level (difficulty)
    if difficulty == 2:
        policy = SAC.load("models/checkpoint_saves/SAC_09_18_2020_01_08_25__5000000_steps.zip")
    else:
        policy = SAC.load("models/checkpoint_saves/SAC_09_17_2020_22_04_30__2000000_steps.zip")

    # Execute one episode.  Make sure that the number of simulation steps
    # matches with the episode length of the task.  When using the default Gym
    # environment, this is the case when looping until is_done == True.  Make
    # sure to adjust this in case your custom environment behaves differently!
    is_done = False
    observation = env.reset()
    accumulated_reward = 0
    while not is_done:
        action, _ = policy.predict(np.expand_dims(norm_env.normalize_obs(observation),  axis=0), deterministic=True)
        observation, reward, is_done, info = env .step(action[0])
        accumulated_reward += reward

    print("Accumulated reward: {}".format(accumulated_reward))

    # store the log for evaluation
    env.platform.store_action_log(output_file)


if __name__ == "__main__":
    main()
