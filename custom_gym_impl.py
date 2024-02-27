import pickle
import random
import os
import time
from typing import Optional, Tuple, Union, List

import gym
from gym.core import ObsType, ActType, RenderFrame

import numpy as np


class Discrete:

    def __init__(self, num_actions):
        """
        This is the action space with the discrete values that will be performed
        Parameters
        ----------
        num_actions
        """

        self.n = num_actions

    def sample(self):
        return random.randint(0, self.n - 1)


class Environment(gym.Env):
    seeker, goal = (0, 0), (0, 4)
    info = {'seeker': seeker, 'goal': goal}

    def __init__(self, *args, **kwargs):
        self.action_space = Discrete(4)
        self.observation_space = Discrete(5 * 5)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self.seeker = (0, 0)
        self.goal = (4, 4)
        return self.get_observation()

    def get_observation(self):
        """
        Encode the seeker position as Integer
        Returns an integers
        -------

        """
        return 5 * self.seeker[0] + self.seeker[1]

    def get_reward(self):
        """
        1 if the seeker finds the goal and 0 if not
        Returns
        -------

        """
        return 1 if self.seeker == self.goal else 0

    def is_done(self):
        """
        Return the value of done, true if the seeker finds the goal, else Fasle
        Returns
        -------

        """
        return True if self.seeker == self.goal else False

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        if action == 0:
            # move down
            self.seeker = (min(self.seeker[0] + 1, 4), self.seeker[1])
        elif action == 1:
            # move left
            self.seeker = (self.seeker[0], max(self.seeker[0] - 1, 0))
        elif action == 2:
            # move up
            self.seeker = (max(self.seeker[0] - 1, 0), self.seeker[1])
        elif action == 3:
            # move right
            self.seeker = (self.seeker[0], min(self.seeker[1] + 1, 4))
        else:
            raise ValueError("No action specified")

        return self.get_observation(), self.get_reward(), self.is_done(), False, self.info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """
        Render the environmental representation for the maze problem
        Returns
        -------

        """
        os.system('cls' if os.name == 'nt' else 'clear')
        grid = [['| ' for _ in range(5)] + ['|\n'] for _ in range(5)]
        grid[self.seeker[0]][self.seeker[1]] = '|S'
        grid[self.goal[0]][self.goal[1]] = '|G'
        print(''.join([''.join(grid_row) for grid_row in grid]))


class Policy:
    """
    We define the policy to take specific actions, which will be more effective than taking random actions
    """

    def __init__(self, env: Environment):
        """
        Policy suggests actions based on the current action
        This can be done by tracking the value of each action pair
        Parameters
        ----------
        env
        """
        # self.sate_action_table = [
        #     [0 for _ in range(env.action_space.n)] for _ in range(env.observation_space.n)
        # ]
        # Initialize state-action table with random values
        self.env = env
        self.state_action_table = np.random.rand(self.env.observation_space.n, self.env.action_space.n)
        self.action_space = self.env.action_space

    def print_action_table(self):
        return self.state_action_table

    def get_action(self, state, explore=True, epsilon=0.1):
        """
        Exploration vs Exploitation
        If you don't want to explore then you will take the best available value for the current state
        Parameters
        ----------
        state
        explore
        epsilon

        Returns
        -------

        """
        # if explore and random.uniform(0, 1) < epsilon:
        #     return self.action_space.sample()
        # return np.argmax(self.sate_action_table[state])
        # Epsilon-greedy exploration
        if explore and np.random.rand() < epsilon:
            # Randomly choose an action
            action = np.random.randint(self.env.action_space.n)
        else:
            # Choose the action with the highest value in the table
            action = np.argmax(self.state_action_table[state])
        return action


class Simulation:

    def __init__(self, env: Environment):
        self.env = env

    def rollout(self, policy, render=False, explore=True, epsilon=0.1):

        experiences = []
        # state = self.env.reset()
        # we comment to reset the environment because, we need to randomize the goal position per episode
        # The new state now becomes
        state = 5 * self.env.seeker[0] + self.env.seeker[1]
        done = False

        while not done:
            action = policy.get_action(state, explore, epsilon)
            next_sate, reward, done, samp_bool, info = self.env.step(action)
            experiences.append([state, action, reward, next_sate])
            state = next_sate
            if render:
                time.sleep(0.05)
                self.env.render()
        return experiences


def update_policy(policy, experiences):
    weight = 0.1
    discount_factor = 0.9
    # The above two are the hyperparameters that needs tuning
    for state, action, reward, next_state in experiences:
        next_max = np.max(policy.state_action_table[next_state])
        value = policy.state_action_table[state][action]
        new_value = (1 - weight) * value + weight * (reward + discount_factor * next_max)
        # This is the bellman's equation, which is used in markov's decision process
        policy.state_action_table[state][action] = new_value


def train_policy(env: Environment, num_episodes=1000, render=False):
    policy = Policy(env)
    sim = Simulation(env)

    for i in range(num_episodes):
        env.seeker = (random.randint(0, 4), random.randint(0, 4))
        env.goal = (random.randint(0, 4), random.randint(0, 4))
        print("Changing the seeker and the goal positions for the episode {0} and they are {1} and {2} respectively"
              .format(i, env.seeker, env.goal))
        experiences = sim.rollout(policy, render=render, explore=True)
        update_policy(policy, experiences)
    return policy


def pickle_trained_policy(policy: Policy):
    with open("maze_state_action.pkl", 'wb') as f:
        pickle.dump(policy, f)


def load_trained_policy_from_pickle():
    with open("maze_state_action.pkl", 'rb') as f:
        trained_policy = pickle.load(f)
        return trained_policy
