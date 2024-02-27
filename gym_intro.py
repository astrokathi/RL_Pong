import random

import gym
import warnings


def gym_environment():
    # Create an environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Create an Observation, to reset the environment initially
    observation = env.reset()

    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        # print("Action space is: ", env.observation_space, "The performed action is: ", action)
        observation, reward, done, info, obj = env.step(action)
        # print(observation)

        if done:
            # If the training is done, then the env will be reset
            observation = env.reset()
    env.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    gym_environment()
    # gym.make("Pong-v4")
    # print(random.uniform(0, 1))
