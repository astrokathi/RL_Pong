import random

from ReinforcementLearning.custom_gym_impl import Environment, Policy, Simulation, train_policy, pickle_trained_policy, \
    load_trained_policy_from_pickle
import time

if __name__ == '__main__':
    environment = Environment()
    # policy = load_trained_policy_from_pickle()
    # untrained_policy = Policy(environment)
    simulation = Simulation(environment)
    # print(policy.print_action_table())
    # Start the loop till the seeker finds the goal
    # while not environment.is_done():
    #     # action = environment.action_space.sample()
    #     # We will now get the action from the policy
    #     action = policy.get_action(environment.get_observation(), epsilon=0.1)
    #     # action = environment.action_space.sample()
    #     print(action)
    #     observation, reward, done, sample_bool, info = environment.step(action)
    #     time.sleep(0.1)
    #     environment.render()
    # experiences = simulation.rollout(untrained_policy, render=True, explore=True, epsilon=0.1)

    # trained_policy = train_policy(environment, render=False)
    # Pickling the trained policy to be used on different maze problems within the same environment
    # pickle_trained_policy(trained_policy)

    trained_policy = load_trained_policy_from_pickle()
    environment.seeker = (random.randint(0, 4), random.randint(0, 4))
    environment.goal = (random.randint(0, 4), random.randint(0, 4))
    simulation.rollout(trained_policy, render=True)
