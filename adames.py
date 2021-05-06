import numpy as np
from sklearn.preprocessing import scale

from es import ES

class AdamES(ES):
    def __init__(
        self,
        env,
        num_actions,
        sigma,
        alpha,
        action_ranges,
        beta1 = 0.9,
        beta2 = 0.999,
        global_optimum=None
    ):
        """
        :param env: simulation environment
        :param num_actions: number of actions the agent can make
        :param sigma: diagonal components of covariance matrix, defines spread of sampled
            actions
        :param alpha: learning rate
        :param action_ranges: clip the search space to the desired action range,
            shape = (num_actions, 2)
        :param beta1: adam coefficient 1
        :param beta2: adam coefficient 2
        :param global_optimum: the optimum value for checking convergence
        """
        super(AdamES, self).__init__(env, num_actions, sigma, alpha, action_ranges, global_optimum)

        self.beta1 = beta1
        self.beta2 = beta2

        self.momentum = 0
        self.dampener = 0

    def _adam(self, grad, t):
        self.momentum = (1 - self.beta1) * grad + self.beta1 * self.momentum
        self.dampener = (1 - self.beta2) * (grad * grad) + self.beta2 * self.dampener

        momentum_adj = self.momentum / (1 - (self.beta1 ** t))
        dampener_adj = self.dampener / (1 - (self.beta2 ** t))

        return momentum_adj / (np.sqrt(dampener_adj) + 1e-8)

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

            # run adam optimization step
            grad_approx = (1 / (np.mean(self.sigma) * batch_size)) * np.dot(normal_dist.T, A)
            optimization_direction = self._adam(grad_approx, i + 1)
            mu += self.alpha * optimization_direction

            # discretize and clip result as needed
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

            if self.global_optimum is not None and self.convergence_time == 0 and reward >= self.global_optimum - 0.5:
                self.convergence_time = i

        if plot == 1:
            self.plot_learning_curve()

        self.steady_state_rewards = self.learning_curve[self.convergence_time:]

        return mu
