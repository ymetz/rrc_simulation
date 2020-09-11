import gym

from rrc_simulation.gym_wrapper.envs import cube_env_modified
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper, HindsightExperienceReplayWrapper
from stable_baselines import HER, SAC
from stable_baselines.common.callbacks import CheckpointCallback


def train():

    initializer = cube_env_modified.RandomInitializer(difficulty=1)

    env = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v2",
        initializer=initializer,
        action_type=cube_env_modified.ActionType.POSITION,
        observation_type=cube_env_modified.ObservationType.BOX,
        frameskip=100,
        visualization=False
    )

    model = HER('MlpPolicy', env, SAC, n_sampled_goal=4, goal_selection_strategy='future', verbose=1,
                tensorboard_log="tblogs", batch_size=256, buffer_size=1000000, gamma=0.95, learning_rate=0.001,
                learning_starts=1000)
    model.learn(int(3e6), log_interval=10, callback=CheckpointCallback(save_freq=int(2e5),
                                                                       save_path='./models/checkpoint_saves'))

    model.save('./models/basic_her_train')

def render():
    initializer = cube_env_modified.RandomInitializer(difficulty=1)

    render_env = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v2",
        initializer=initializer,
        action_type=cube_env_modified.ActionType.POSITION,
        observation_type=cube_env_modified.ObservationType.BOX,
        frameskip=100,
        visualization=True
    )

    model = HER.load('./models/basic_her_train', env=render_env)

    obs = render_env.reset()
    is_done = False
    while not is_done:
        action, _ = model.predict(obs)
        obs, rew, is_done, info = render_env.step(action)

    print("Reward at final step: {:.3f}".format(rew))


if __name__ == "__main__":
    train()