import gym

from rrc_simulation.gym_wrapper.envs.cube_env_modified import CubeEnv, ActionType, RandomInitializer, ObservationType, \
    FlatObservationWrapper, GoalObservationWrapper
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper, HindsightExperienceReplayWrapper
from rrc_simulation.gym_wrapper.wrappers import TimeFeatureWrapper
from stable_baselines import HER, SAC, PPO2, TD3
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines.common import set_global_seeds
import os
import tensorflow as tf
import numpy as np
import math
from datetime import datetime


def train(method="SAC"):

    def get_multi_process_env(num_of_envs, subprocess=True, amplitude_scaling=False, frameskip=5, with_goals=False,
                              action_type=ActionType.POSITION, difficulty=1):

        initializer = RandomInitializer(difficulty=difficulty)

        def _make_env(rank):
            def _init():
                obs_type = ObservationType.WITH_GOALS if with_goals else ObservationType.WITHOUT_GOALS
                out_env = CubeEnv(frameskip=frameskip,
                                  visualization=False,
                                  initializer=initializer,
                                  action_type=action_type,
                                  observation_type=obs_type)
                out_env.seed(seed=12345)
                out_env.action_space.seed(seed=12345)
                if not with_goals:
                    out_env = FlatObservationWrapper(out_env, amplitude_scaling=amplitude_scaling)
                    out_env = TimeFeatureWrapper(out_env, max_steps=math.ceil(3750 / frameskip))
                else:
                    out_env = GoalObservationWrapper(out_env, amplitude_scaling=amplitude_scaling)
                return out_env

            return _init

        if subprocess:
            return SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
        else:
            return DummyVecEnv([_make_env(rank=i) for i in range(num_of_envs)])

    date_time_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_")
    print(method, date_time_str)
    set_global_seeds(0)

    if method == "HER":
        env = get_multi_process_env(1, subprocess=False, amplitude_scaling=True, frameskip=5, with_goals=True)
        env.set_attr("reward_range", 1000)
        policy_kwargs = dict(layers=[128, 128], act_fun=tf.tanh)

        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))

        model = HER("MlpPolicy", env, SAC, policy_kwargs=policy_kwargs, n_sampled_goal=4,
                    goal_selection_strategy='future', verbose=1,
                    tensorboard_log="tblogs", batch_size=512, buffer_size=100000, gamma=0.98, learning_starts=10000,
                    random_exploration=0.15)
        model.learn(int(2e6), log_interval=10, callback=CheckpointCallback(save_freq=int(1e5),
                                                                           save_path='models/checkpoint_saves',
                                                                           name_prefix=method + '_' + date_time_str),
                    tb_log_name=method + '_' + date_time_str
                    )
    if method == "SAC":
        env = VecNormalize(get_multi_process_env(1, subprocess=False, amplitude_scaling=True, frameskip=3,
                                                 action_type=ActionType.POSITION, difficulty=2),
                           norm_reward=False, clip_reward=1500, gamma=0.99)
        policy_kwargs = dict(layers=[256, 256])

        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))
        model = SAC("LnMlpPolicy", env, policy_kwargs=policy_kwargs, buffer_size=1000000, batch_size=1024, gamma=0.99,
                    train_freq=256, tau=0.005, learning_starts=10000, tensorboard_log="tblogs", verbose=1,
                    action_noise=action_noise)
        model.learn(int(5e6), log_interval=10, callback=CheckpointCallback(save_freq=int(1e5),
                                                                           save_path='models/checkpoint_saves',
                                                                           name_prefix=method + '_' + date_time_str),
                    tb_log_name=method + '_' + date_time_str
                    )
        env.save("normalized_env_"+date_time_str)
    if method == "CONTINUE_SAC_PUSH":
        env = VecNormalize.load("models/normalized_env_09_17_2020_15_43_53_",
                                get_multi_process_env(1, subprocess=False, amplitude_scaling=True, frameskip=3,
                                                      action_type=ActionType.POSITION, difficulty=1))

        model = SAC.load("models/checkpoint_saves/SAC_09_17_2020_15_43_53__2000000_steps.zip", env=env,
                         tensorboard_log="tblogs",)
        model.learn(int(1e6), log_interval=10, callback=CheckpointCallback(save_freq=int(1e5),
                                                                           save_path='models/checkpoint_saves',
                                                                           name_prefix=method + '_' + date_time_str),
                    tb_log_name=method + '_' + date_time_str
                    )
        env.save("normalized_env_"+date_time_str)
    if method == "CONTINUE_SAC_TAKE":
        env = VecNormalize.load("models/normalized_env_09_17_2020_17_07_04_",
                                get_multi_process_env(1, subprocess=False, amplitude_scaling=True, frameskip=3,
                                                      action_type=ActionType.POSITION, difficulty=2))

        model = SAC.load("models/checkpoint_saves/SAC_09_17_2020_17_07_04__2000000_steps.zip", env=env,
                         tensorboard_log="tblogs",)
        model.learn(int(1e6), log_interval=10, callback=CheckpointCallback(save_freq=int(1e5),
                                                                           save_path='models/checkpoint_saves',
                                                                           name_prefix=method + '_' + date_time_str),
                    tb_log_name=method + '_' + date_time_str
                    )
        env.save("normalized_env_"+date_time_str)
    if method == "save_vec_env":
        env = VecNormalize(get_multi_process_env(1, subprocess=True, amplitude_scaling=False, frameskip=5,
                                                 action_type=ActionType.POSITION),
                           norm_reward=False, clip_reward=1500, gamma=0.99)
        # env = get_multi_process_env(1, subprocess=False, amplitude_scaling=True, frameskip=5)
        policy_kwargs = dict(layers=[256, 256])

        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))
        model = SAC.load("models/SAC_09_17_2020_00_57_49_", env=env)
        model.learn(int(1e5), log_interval=1)
        env.save("normalized_env_"+date_time_str)
        return
    else:
        return

    print("save model: ", os.path.join('models', method+'_'+date_time_str))
    model.save(os.path.join('models', method+'_'+date_time_str))


def render():
    initializer = RandomInitializer(difficulty=1)

    def get_multi_process_env(num_of_envs):
        def _make_env(rank):
            def _init():
                out_env = CubeEnv(frameskip=5,
                              visualization=True,
                              initializer=initializer,
                              action_type=ActionType.POSITION,
                              observation_type=ObservationType.WITHOUT_GOALS)
                out_env.seed(seed=rank)
                out_env.action_space.seed(seed=rank)
                out_env = FlatObservationWrapper(out_env)
                return out_env

            return _init

        return DummyVecEnv([_make_env(rank=i) for i in range(num_of_envs)])

    render_env = VecNormalize.load("models/PPO_09_14_2020_19_06_26.pkl", get_multi_process_env(1))

    model = PPO2.load("models/checkpoint_saves/rl_model_10000000_steps", env=render_env)

    obs = model.env.reset()
    is_done = False
    while not is_done:
        action, _ = model.predict(obs)
        obs, rew, is_done, info = render_env.step(action)

    print("Reward at final step: {:.3f}".format(rew))


if __name__ == "__main__":
    train('SAC')
    # render()