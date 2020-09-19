"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import enum

import numpy as np
import gym
import random
import math

from rrc_simulation import TriFingerPlatform
from rrc_simulation import visual_objects
from rrc_simulation.tasks import move_cube


class RandomInitializer:
    """Initializer that samples random initial states and goals."""

    def __init__(self, difficulty):
        """Initialize.

        Args:
            difficulty (int):  Difficulty level for sampling goals.
        """
        self.difficulty = difficulty

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        return move_cube.sample_goal(difficulty=-1)

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        return move_cube.sample_goal(difficulty=self.difficulty)


class CompletelyRandomInitializer:
    """Initializer that samples random initial states and goals."""

    def __init__(self):
        self.difficulty = 1

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        return move_cube.sample_goal(difficulty=-1)

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        self.difficulty = random.randrange(1, 5)
        return move_cube.sample_goal(difficulty=self.difficulty )


class FixedInitializer:
    """Initializer that uses fixed values for initial pose and goal."""

    def __init__(self, difficulty, initial_state, goal):
        """Initialize.

        Args:
            difficulty (int):  Difficulty level of the goal.  This is still
                needed even for a fixed goal, as it is also used for computing
                the reward (the cost function is different for the different
                levels).
            initial_state (move_cube.Pose):  Initial pose of the object.
            goal (move_cube.Pose):  Goal pose of the object.

        Raises:
            Exception:  If initial_state or goal are not valid.  See
            :meth:`move_cube.validate_goal` for more information.
        """
        move_cube.validate_goal(initial_state)
        move_cube.validate_goal(goal)
        self.difficulty = difficulty
        self.initial_state = initial_state
        self.goal = goal

    def get_initial_state(self):
        """Get the initial state that was set in the constructor."""
        return self.initial_state

    def get_goal(self):
        """Get the goal that was set in the constructor."""
        return self.goal


class ActionType(enum.Enum):
    """Different action types that can be used to control the robot."""

    #: Use pure torque commands.  The action is a list of torques (one per
    #: joint) in this case.
    TORQUE = enum.auto()
    #: Use joint position commands.  The action is a list of angular joint
    #: positions (one per joint) in this case.  Internally a PD controller is
    #: executed for each action to determine the torques that are applied to
    #: the robot.
    POSITION = enum.auto()
    #: Use both torque and position commands.  In this case the action is a
    #: dictionary with keys "torque" and "position" which contain the
    #: corresponding lists of values (see above).  The torques resulting from
    #: the position controller are added to the torques in the action before
    #: applying them to the robot.
    TORQUE_AND_POSITION = enum.auto()


class ObservationType(enum.Enum):
    # Return Observation as a dict
    WITH_GOALS = enum.auto()
    # Return Observation as a flat box space
    WITHOUT_GOALS = enum.auto()


class FlatObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, amplitude_scaling=False):
        super().__init__(env)
        self.amplitude_scaling = amplitude_scaling

        low = [
            self.observation_space["observation"][name].low.flatten()
            for name in self.observation_names
        ]

        high = [
            self.observation_space["observation"][name].high.flatten()
            for name in self.observation_names
        ]

        self.observation_space = gym.spaces.Box(
            low=np.concatenate(low), high=np.concatenate(high)
        )

    def observation(self, obs):
        observation = [obs["observation"][name].flatten() for name in self.observation_names]

        observation = np.concatenate(observation)
        if self.amplitude_scaling:
            observation *= np.random.uniform(0.8, 1.2, 1)

        return observation


class GoalObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, amplitude_scaling=False):
        super().__init__(env)
        self.amplitude_scaling = amplitude_scaling

        low = [
            self.observation_space["observation"][name].low.flatten()
            for name in self.observation_names
        ]

        high = [
            self.observation_space["observation"][name].high.flatten()
            for name in self.observation_names
        ]

        goal_low = [
            self.observation_space["desired_goal"][name].low.flatten()
            for name in self.goal_names
        ]

        goal_high = [
            self.observation_space["desired_goal"][name].high.flatten()
            for name in self.goal_names
        ]

        self.observation_space = gym.spaces.Dict(
            {"observation": gym.spaces.Box(
                low=np.concatenate(low), high=np.concatenate(high)),
                "desired_goal": gym.spaces.Box(
                low=np.concatenate(goal_low), high=np.concatenate(goal_high)),
                "achieved_goal": gym.spaces.Box(
                low=np.concatenate(goal_low), high=np.concatenate(goal_high)),
            }
        )

    def observation(self, obs):
        observation = [obs["observation"][name].flatten() for name in self.observation_names]

        if self.amplitude_scaling:
            observation = np.concatenate(observation) * np.random.uniform(0.8, 1.2, 1)
        else:
            observation = np.concatenate(observation)

        desired_goal = [obs["desired_goal"][name].flatten() for name in self.goal_names]
        achieved_goal = [obs["achieved_goal"][name].flatten() for name in self.goal_names]

        observation = {"observation": observation,
                       "desired_goal": np.concatenate(desired_goal),
                       "achieved_goal": np.concatenate(achieved_goal)
                       }
        return observation


class CubeEnv(gym.GoalEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        initializer,
        action_type=ActionType.POSITION,
        observation_type=None,
        frameskip=1,
        visualization=False,
        testing=False
    ):
        """Initialize.

        Args:
            initializer: Initializer class for providing initial cube pose and
                goal pose.  See :class:`RandomInitializer` and
                :class:`FixedInitializer`.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        # Basic initialization
        # ====================

        self.initializer = initializer
        self.action_type = action_type
        self.visualization = visualization

        # TODO: The name "frameskip" makes sense for an atari environment but
        # not really for our scenario.  The name is also misleading as
        # "frameskip = 1" suggests that one frame is skipped while it actually
        # means "do one step per step" (i.e. no skip).
        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip

        # will be initialized in reset()
        self.platform = None

        # Create the action and observation spaces
        # ========================================

        spaces = TriFingerPlatform.spaces

        object_state_space = gym.spaces.Dict(
            {
                "position": spaces.object_position.gym,
                "orientation": spaces.object_orientation.gym,
            }
        )

        if self.action_type == ActionType.TORQUE:
            self.unscaled_action_space = spaces.robot_torque.gym
            self.action_space = gym.spaces.Box(low=-np.ones((9,)), high=np.ones((9,)))
        elif self.action_type == ActionType.POSITION:
            self.unscaled_action_space = spaces.robot_position.gym
            self.action_space = gym.spaces.Box(low=-np.ones((9,)), high=np.ones((9,)))
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.unscaled_action_space = gym.spaces.Dict(
                {
                    "torque": spaces.robot_torque.gym,
                    "position": spaces.robot_position.gym,
                }
            )
            self.action_space = gym.spaces.Dict(
                {
                    "torque": gym.spaces.Box(low=-np.ones((9,)), high=np.ones((9,))),
                    "position": gym.spaces.Box(low=-np.ones((9,)), high=np.ones((9,))),
                }
            )
        else:
            raise ValueError("Invalid action_type")

        self.observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(
                    {
                        "robot_position": spaces.robot_position.gym,
                        "robot_velocity": spaces.robot_velocity.gym,
                        "robot_tip_positions": gym.spaces.Box(
                            low=np.array([spaces.object_position.low] * 3),
                            high=np.array([spaces.object_position.high] * 3),
                        ),
                        "object_position": spaces.object_position.gym,
                        "object_orientation": spaces.object_orientation.gym,
                        "goal_object_position": spaces.object_position.gym,
                    }
                ),
                "desired_goal": object_state_space,
                "achieved_goal": object_state_space,
            }
        )

        smoothing_params = {
            "num_episodes": 1000,
            "start_after": 3.0 / 7.0,
            "final_alpha": 0.975,
            "stop_after": 5.0 / 7.0,
        }
        if testing:
            self.smoothing_start_episode = 0
            self.smoothing_alpha = smoothing_params["final_alpha"]
            self.smoothing_increase_step = 0
            self.smoothing_stop_episode = math.inf
        else:
            self.smoothing_stop_episode = int(
                smoothing_params["num_episodes"]
                * smoothing_params["stop_after"]
            )

            self.smoothing_start_episode = int(
                smoothing_params["num_episodes"]
                * smoothing_params["start_after"]
            )
            num_smoothing_increase_steps = (
                    self.smoothing_stop_episode - self.smoothing_start_episode
            )
            self.smoothing_alpha = 0
            self.smoothing_increase_step = (
                    smoothing_params["final_alpha"] / num_smoothing_increase_steps
            )

        self.smoothed_action = None
        self.episode_count = 0

    @staticmethod
    def compute_reward(previous_observation, observation):

        # calculate first reward term
        current_distance_from_block = np.linalg.norm(
            observation["robot_tip_positions"] - observation["object_position"]
        )
        previous_distance_from_block = np.linalg.norm(
            previous_observation["robot_tip_positions"]
            - previous_observation["object_position"]
        )

        reward_term_1 = (
            previous_distance_from_block - current_distance_from_block
        )

        # calculate second reward term
        current_dist_to_goal = np.linalg.norm(
            observation["goal_object_position"]
            - observation["object_position"]
        )
        previous_dist_to_goal = np.linalg.norm(
            previous_observation["goal_object_position"]
            - previous_observation["object_position"]
        )
        reward_term_2 = previous_dist_to_goal - current_dist_to_goal

        reward = 500 * reward_term_1 + 500 * reward_term_2

        return reward

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float) : amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the difficulty level of
              the goal.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > move_cube.episode_length:
            excess = step_count_after - move_cube.episode_length
            num_steps = max(1, num_steps - excess)

        unscaled_action = self.unscale_action(action, self.unscaled_action_space)

        if self.smoothed_action is None:
            # start with current position
            # self.smoothed_action = self.finger.observation.position
            self.smoothed_action = unscaled_action

        self.smoothed_action = (
            self.smoothing_alpha * self.smoothed_action
            + (1 - self.smoothing_alpha) * action
        )

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > move_cube.episode_length:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(self.smoothed_action)
            t = self.platform.append_desired_action(robot_action)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            previous_observation = self._create_observation(t)
            observation = self._create_observation(t + 1)

            reward += self.compute_reward(
                previous_observation=previous_observation["observation"],
                observation=observation["observation"],
            )

        is_done = self.step_count == move_cube.episode_length

        return observation, reward, is_done, self.info

    def reset(self):
        # reset simulation
        del self.platform

        # initialize simulation
        initial_robot_position = (
            TriFingerPlatform.spaces.robot_position.default
        )
        initial_object_pose = self.initializer.get_initial_state()
        goal_object_pose = self.initializer.get_goal()

        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=initial_object_pose,
        )

        self.goal = {
            "position": goal_object_pose.position,
            "orientation": goal_object_pose.orientation,
        }

        # updates smoothing parameters
        self.update_smoothing()
        self.episode_count += 1
        self.smoothed_action = None

        # visualize the goal
        if self.visualization:
            self.goal_marker = visual_objects.CubeMarker(
                width=0.065,
                position=goal_object_pose.position,
                orientation=goal_object_pose.orientation,
                physicsClientId=self.platform.simfinger._pybullet_client_id,
            )

        self.info = {"difficulty": self.initializer.difficulty}

        self.step_count = 0

        return self._create_observation(0)

    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.

        .. note::

           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.

        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        move_cube.random = self.np_random
        return [seed]

    def _create_observation(self, t):
        robot_observation = self.platform.get_robot_observation(t)
        object_observation = self.platform.get_object_pose(t)
        robot_tip_positions = self.platform.forward_kinematics(
            robot_observation.position
        )
        robot_tip_positions = np.array(robot_tip_positions)

        observation = {
            "observation": {
                "robot_position": robot_observation.position,
                "robot_velocity": robot_observation.velocity,
                "robot_tip_positions": robot_tip_positions,
                "object_position": object_observation.position,
                "object_orientation": object_observation.orientation,
                "goal_object_position": self.goal["position"],
            },
            "desired_goal": self.goal,
            "achieved_goal": {
                "position": object_observation.position,
                "orientation": object_observation.orientation,
            },
        }
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def update_smoothing(self):
        """
        Update the smoothing coefficient with which the action to be
        applied is smoothed
        """
        if (
            self.smoothing_start_episode
            <= self.episode_count
            < self.smoothing_stop_episode
        ):
            self.smoothing_alpha += self.smoothing_increase_step
        print(
            "episode: {}, smoothing: {}".format(
                self.episode_count, self.smoothing_alpha
            )
        )

    @staticmethod
    def unscale_action(y, space):
        """
        Unscale some input from [-1;1] to the range of another space
        """
        return space.low + (y + 1.0) / 2.0 * (space.high - space.low)