import numpy as np
from numba import jit


class Policy():
    def __init__(self, state_dims=10, num_actions=5, seed=0, episode_end="dual",
                 death_threshold=0.1, recovery_threshold=0.999):
        np.random.seed(seed)
        self.state_dims = state_dims
        self.num_actions = num_actions

        # sample W
        self.W = np.random.uniform(-0.02, 0.02, size=(num_actions, state_dims, state_dims))
        self.sigma = None
        self.sigma = self.sample_sigma()
        self.state = np.array(None)

        # episode end strategies
        self.episode_end_map = {
            "dual": self.calculate_dual_threshold_episode_end,
            "probabilistic": self.calculate_probabilistic_episode_end
        }
        self.death_threshold = death_threshold
        self.recovery_threshold = recovery_threshold

        self.step_function = self.episode_end_map[episode_end]

        # initialize episode termination vector that will be used to determine
        # whether or not we randomly terminate the episode
        self.v = np.random.uniform(-10, 10, size=(self.state_dims, 1))


    @jit
    def sample_sigma(self):
        if self.sigma:
            print("Here")
            return self.sigma
        else:
            sigma = np.random.uniform(size=(self.num_actions, self.state_dims, self.state_dims))
            for i in range(self.num_actions):
                sigma[i] = np.inner(sigma[i], sigma[i])
        return sigma

    def calculate_probabilistic_episode_end(self, state):
        '''
        Function to check episode end based on a random vector and one threshold.
        :param state: <class 'numpy.ndarray'> current state vector
        :return: (<class 'boolean'>, <class 'string'> or <'class 'Nonetype'>, <class 'float'>)
                A tuple containing boolean to indicate episode end, cause of the episode end, reward value
        '''
        # initialize episode termination vector that will be used to determine
        # whether or not we randomly terminate the episode
        # v = np.random.uniform(-1, 1, size=(self.state_dims, 1))

        # if the sigmoid function sigmoid(v', state) < threshold, we say the episode finished successfully.
        score = np.dot(state, self.v)    # calculate score for sigmoid
        prob_episode_end = 1 / (1 + np.exp(-score))     # calculate sigmoid(v', state)
        reward = np.sum(np.where(np.logical_and(state < 2, state > 1)))
        return (True, "Random end with sigmoid < 0.3", reward) if prob_episode_end < self.death_threshold else (False, None, reward)

    @jit
    def calculate_dual_threshold_episode_end(self, state):
        '''
        Function to check episode end based on two probability thresholds. One to simulate recovery, other to simulate death.
        :param state: <class 'numpy.ndarray'> current state vector
        :return: (<class 'boolean'>, <class 'string'> or <'class 'Nonetype'>, <class 'float'>)
                A tuple containing boolean to indicate episode end, cause of the episode end, reward value
        '''
        score = np.dot(state, self.v)
        dual_episode_end = 1 / (1 + np.exp(-score))
        if dual_episode_end < self.death_threshold:
            return True, "Episode end (Probabilistic death)", -10
        elif dual_episode_end > self.recovery_threshold:
            return True, "Episode end (Probabilistic recovery)", +20
        else:
            return False, None, -0.5

    @jit
    def step(self, state, action_index):
        '''
        Function to simulate taking a step.
        :param state: <class 'numpy.ndarray'> current state vector
        :param action_index: <class 'integer'> integer indicating the action to be taken
        :return: (<class 'numpy.ndarray'>, <class 'float'>, <class 'boolean'>, <class 'string'>)
                A tuple containing next state, reward, boolean indicating episode end, message
        '''
        action = self.W[action_index]
        sigma = self.sigma[action_index]
        s = np.dot(state, action)
        noise = np.random.multivariate_normal(mean=s.squeeze(), cov=sigma)
        new_state = s + noise

        _ = "max iteration reached"
        done, message, reward = self.step_function(new_state)
        if done:
            _ = message
        return new_state, reward, done, _

    @jit
    def gen_data(self, max_rollouts=100, max_steps=100):
        '''
        Function to generate data by rolling out the expert policy. Not used for DQNs.
        :param max_rollouts: <class 'integer'> integer indicating max number of full rollouts (till episode end)
        :param max_steps: <class 'integer'> integer indicating max iterations per episode.
        :return: (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>)
                A tuple of episodic returns, state returns, all states, all actions
        '''
        observations = []
        state_returns = []
        episode_returns = []
        actions = []

        for rollout in range(max_rollouts):
            obs = self.reset()
            done = False

            totalr = 0.
            steps = 0
            r = 0.

            while not done:
                action_index = np.random.randint(0, self.num_actions)
                observations.append(obs)
                actions.append(action_index)
                state_returns.append(r)
                obs, r, done, _ = self.step(obs, action_index)
                totalr += r
                steps += 1
                if done:
                    print(_)
                if steps >= max_steps:
                    break

            episode_returns.append(totalr)
        return np.asarray(episode_returns), np.asarray(state_returns), np.asarray(observations), np.asarray(actions)

    def get_W(self):
        return self.W

    def get_sigma(self):
        return self.sigma

    def reset(self):
        if self.state.shape == ():
            self.state = np.random.uniform(size=(1, self.state_dims))
        return self.state


def test():
    env = Policy()
    s = env.reset()
    new_s, r, d, _ = env.step(s, 1)

    print(np.array_equal(new_s, s))

    new_s = env.reset()

    print(np.array_equal(new_s, s))

    W = env.get_W()
    sigma = env.get_sigma()
    print(W.shape, sigma.shape)

    env.reset()
    ret, state_ret, observs, acts = env.gen_data()
    print(ret.shape, state_ret.shape, observs.shape, acts.shape)


if __name__ == '__main__':
    test()

