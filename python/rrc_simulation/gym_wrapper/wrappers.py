import gym
from gym.wrappers import TimeLimit
import numpy as np

class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.
    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high= np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        env.reward_range = 1000

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.
        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))


class FrameStackWrapper(gym.Wrapper):
    """
    Frame stacking wrapper for vectorized environment

    :param env: (gym.env) the vectorized environment to wrap
    :param n_stack: (int) Number of frames to stack
    """
    def __init__(self, env, n_stack):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        self.env = env
        self.n_stack = n_stack
        wrapped_obs_space = env.observation_space
        low = np.repeat(wrapped_obs_space.low, self.n_stack, axis=-1)
        high = np.repeat(wrapped_obs_space.high, self.n_stack, axis=-1)
        self.stackedobs = np.zeros(low.shape, low.dtype)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

        super(FrameStackWrapper, self).__init__(env)


    def reset(self):
        obs = self.env.reset()
        self.stackedobs[...] = 0
        self.stackedobs[-obs.shape[-1]:] = obs
        return self.stackedobs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        last_ax_size = obs.shape[-1]
        self.stackedobs = np.roll(self.stackedobs, shift=-last_ax_size, axis=-1)
        self.stackedobs[-obs.shape[-1]:] = obs
        return self.stackedobs, reward, done, info

    def close(self):
        self.env.close()