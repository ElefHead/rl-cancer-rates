from collections import namedtuple
from model import ModelParametersCopier
import plotting
from os import path, makedirs
import tensorflow as tf
import numpy as np
import itertools
from random import sample


def make_epsilon_greedy_policy(estimator, num_actions):

    def policy_fn(sess, observation, epsilon):
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        q_values = estimator.predict(sess, observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def deep_q_learning(sess, env, q_estimator, target_estimator, num_episodes, num_actions,
                    experiment_dir, replay_memory_size=500000, replay_memory_init_size=50000,
                    update_target_estimator_steps=10000, discount_factor=0.99, epsilon_start=1.0,
                    epsilon_end=0.1, epsilon_decay_steps=500000, batch_size=32):

    Transistion = namedtuple("Transition", [
        "state", "action", "reward", "next_state", "done"
    ])

    epsilon = 0

    replay_memory = []

    estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )

    checkpoint_dir = path.join(experiment_dir, "checkpoints")
    checkpoint_path = path.join(checkpoint_dir, "model")
    monitor_path = path.join(experiment_dir, "monitor")

    if not path.exists(checkpoint_dir):
        makedirs(checkpoint_dir)
    if not path.exists(monitor_path):
        makedirs(monitor_path)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    plc = make_epsilon_greedy_policy(
        q_estimator,
        num_actions
    )

    print("Generating replay memory")
    state = env.reset()
    for i in range(replay_memory_init_size):
        # action_probs = plc(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        action = np.random.choice(num_actions)
        # print("Action = {}".format(action))

        next_state, reward, done = env.step(state, action)
        replay_memory.append(Transistion(state, action, reward, next_state, done))
        if done:
            state = env.reset()
        else:
            state = next_state

    print("Finished populating replay memory")

    for i_episode in range(num_episodes):
        saver.save(tf.get_default_session(), checkpoint_path)

        #reset env
        state = env.reset()
        loss = None

        for t in itertools.count():

            # Epsilon for the time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            if total_t % update_target_estimator_steps == 0:
                estimator_copy.make(sess)
                print("Copied model params to target network")

            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                t, total_t, i_episode + 1, num_episodes, loss), end="")

            action_probs = plc(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, reward, done = env.step(state, action)

            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            replay_memory.append(Transistion(state, action, reward, next_state, done))

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            samples = sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            q_values_next = target_estimator.predict(sess, next_states_batch.reshape(-1, next_states_batch.shape[-1]))
            # print(q_values_next[:5])
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(
                q_values_next, axis=1)

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            states_batch_shape = states_batch.shape
            loss = q_estimator.update(
                sess, states_batch.reshape(-1, states_batch_shape[-1]),
                action_batch,
                targets_batch
            )

            if done:
                break

            state = next_state
            total_t += 1

        # episode_summary = tf.Summary()
        # episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
        # episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], tag="episode/reward")
        # episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], tag="episode/length")
        #
        # q_estimator.summary_writer.add_summary(episode_summary, i_episode)
        # q_estimator.summary_writer.flush()

        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode + 1],
            episode_rewards=stats.episode_rewards[:i_episode + 1])

    return stats
