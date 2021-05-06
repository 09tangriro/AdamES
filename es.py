import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale


class ES(object):
    def __init__(
        self,
        env,
        num_actions,
        sigma,
        alpha,
        action_ranges,
        global_optimum=None,
    ):
        """
        :param env: simulation environment
        :param num_actions: number of actions the agent can make
        :param sigma: diagonal components of covariance matrix, defines spread of sampled
            actions
        :param alpha: learning rate
        :param action_ranges: clip the search space to the desired action range,
            shape = (num_actions, 2)
        :param global_optimum: the optimum value for checking convergence
        """
        self.env = env
        self.num_actions = num_actions
        self.sigma = sigma
        self.alpha = alpha
        self.action_ranges = action_ranges
        self.global_optimum = global_optimum
        assert len(self.action_ranges) == num_actions

        self.learning_curve = []
        self.parameter_curve = []
        self.convergence_time = 0

    def _initialize_action(self):
        action = np.zeros(shape=(self.num_actions))

        for i, _ in enumerate(action):
            if self.action_ranges[i]:
                low = self.action_ranges[i][0]
                high = self.action_ranges[i][1]
            else:
                low = -sys.maxsize
                high = sys.maxsize

            action[i] = np.random.uniform(low, high)

        return action

    def _clip(self, batch):
        actions = batch.T

        for i, _ in enumerate(actions):
            if self.action_ranges[i]:
                low = self.action_ranges[i][0]
                high = self.action_ranges[i][1]

                actions[i] = np.clip(actions[i], low, high)

        return actions.T

    def plot_learning_curve(self):
        plt.plot(self.learning_curve)
        plt.title("Learning Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.show()

    def optimize(self, num_episodes, batch_size, verbose=0, plot=1, starting_point=None):
        """
        :param num_episodes: number of episodes to optimize over
        :param batch_size: number of env points to sample per episode
        :param verbose: toggle to print training progress
        :param plot: toggle to plot results
        :param starting_point: initial action vector, None or shape=(num_actions,)
        :return: predicted deterministic optimum of the env function
        """
        if starting_point:
            mu = starting_point
        else:
            mu = self._initialize_action()

        self.learning_curve = []
        self.parameter_curve = []
        rewards_dict = {}

        # main loop
        for i in range(num_episodes):
            rewards = np.zeros(batch_size)
            normal_dist = np.random.randn(batch_size, self.num_actions)
            batch = mu + self.sigma * normal_dist
            batch = self._clip(batch)

            # collect rewards
            for j, action in enumerate(batch):
                if tuple(action) not in rewards_dict:
                    _, reward, _ = self.env.step(action)
                    rewards_dict[tuple(action)] = reward
                else:
                    reward = rewards_dict[tuple(action)]

                rewards[j] = reward

            # scale rewards
            A = scale(rewards)

            # run optimization step
            mu += (self.alpha / (np.mean(self.sigma) * batch_size)) * np.dot(
                normal_dist.T, A
            )

            # clip result as needed
            mu = self._clip(mu)

            # get reward for new guess
            if tuple(mu) in rewards_dict:
                reward = rewards_dict[tuple(mu)]
            else:
                _, reward, _ = self.env.step(mu)
                rewards_dict[tuple(mu)] = reward

            if verbose > 0:
                print("EPISODE {}".format(i))
                print("New Guess: {} Reward: {}".format(mu, reward))

            self.learning_curve.append(reward)
            self.parameter_curve.append(mu.copy())

            if self.global_optimum and self.convergence_time == 0 and reward <= self.global_optimum + 0.5:
                self.convergence_time = i

        if plot == 1:
            self.plot_learning_curve()

        return mu
