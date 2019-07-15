import numpy as np
from numba import jit


class Policy():
    def __init__(self, state_dims=10, num_actions=5, seed=0):
        np.random.seed(seed)
        self.state_dims = state_dims
        self.num_actions = num_actions

        # sample W
        self.W = np.random.uniform(size=(num_actions, state_dims, state_dims))
        self.sigma = None
        self.sigma = self.sample_sigma()

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

    def calculate_probabilistic_episode_end(self, state, threshold=0.3):
        # initialize episode termination vector that will be used to determine
        # whether or not we randomly terminate the episode
        v = np.random.uniform(size=(self.state_dims, 1))

        # if the sigmoid function sigmoid(v', state) < threshold, we say the episode finished successfully.
        score = np.dot(state, v)    # calculate score for sigmoid
        prob_episode_end = 1 / (1 + np.exp(-score))     # calculate sigmoid(v', state)
        return True if prob_episode_end < threshold else False

    @jit
    def step(self, state, action, sigma):
        s = np.dot(state, action)
        noise = np.random.multivariate_normal(mean=s.squeeze(), cov=sigma)
        new_state = s + noise
        reward = np.sum(np.where(np.logical_and(new_state < 2, new_state > 1)))

        done = self.calculate_probabilistic_episode_end(new_state)
        if done:
            print("Episode end by done")
        return new_state, reward, done

    def generate_action(self):
        index = np.random.randint(0, self.num_actions)
        return self.W[index], self.sigma[index]

    @jit
    def gen_data(self, max_rollouts=100, max_steps=100):
        observations = []
        state_returns = []
        episode_returns = []
        actions = []

        for rollout in range(max_rollouts):
            obs = np.random.uniform(size=(1, self.state_dims))
            done = False

            totalr = 0.
            steps = 0
            r = 0.

            while not done:
                action, sigma = self.generate_action()
                observations.append(obs)
                actions.append(action)
                state_returns.append(r)
                obs, r, done = self.step(obs, action, sigma)
                totalr += r
                steps += 1

                if steps >= max_steps:
                    break

            episode_returns.append(totalr)
        return np.asarray(episode_returns), np.asarray(state_returns), np.asarray(observations), np.asarray(actions)

    def get_W(self):
        return self.W

    def get_sigma(self):
        return self.sigma


def main():
    env = Policy()
    W = env.get_W()
    sigma = env.get_sigma()
    print(W.shape, sigma.shape)

    ret, state_ret, observs, acts = env.gen_data()
    print(ret.shape, state_ret.shape, observs.shape, acts.shape)


if __name__ == '__main__':
    main()

