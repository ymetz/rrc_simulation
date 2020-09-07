import gym

from rrc_simulation.gym_wrapper.envs import cube_env_modified
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper, HindsightExperienceReplayWrapper
from stable_baselines import HER, SAC

if __name__ == "__main__":

    initializer = cube_env_modified.RandomInitializer(difficulty=1)

    env = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v2",
        initializer=initializer,
        action_type=cube_env_modified.ActionType.POSITION,
        observation_type=cube_env_modified.ObservationType.BOX,
        frameskip=100,
        visualization=True
    )

    model = HER('MlpPolicy', env, SAC, n_sampled_goal=4, goal_selection_strategy='future', verbose=1)
    model.learn(1000)

    model.save('./models/basic_her_train')

    model.load('./models/basic_her_train', env=env)

    obs = env.reset()
    is_done = False
    while not is_done:
        action = model.predict(obs)
        obs, rew, is_done, info = env.step(action)

    print("Reward at final step: {:.3f}".format(rew))