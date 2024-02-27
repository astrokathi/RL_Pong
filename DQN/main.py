# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- import libraries --------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


import gym
# import atari_py
from agent import Agent
from utils import make_env


# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- main --------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # print(atari_py.list_games())

    env = gym.make("Pong-v4", render_mode="human")
    spec = gym.spec("PongNoFrameskip-v4")

    #   Actions:
    print(f"Action Space: {env.action_space}")

    #   States:
    print(f"Observation Space: {env.observation_space}")

    #   Rewards:
    print(f"Max Episode Steps: {spec.max_episode_steps}")

    # env = make_env("PongNoFrameskip-v4")
    train = 1
    test = 0
    num_episodes = 100
    graph = True

    file_type = 'tf'
    file = 'saved_networks1/dqn_model10'

    dqn_agent = Agent(lr=0.0001, discount_factor=0.99, num_actions=6, epsilon=1.0, batch_size=32,
                      input_dims=(210, 160, 3))

    if train and not test:
        dqn_agent.train_model(env, num_episodes, graph)
    else:
        dqn_agent.test(env, num_episodes, file_type, file, graph)
