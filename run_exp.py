import tensorflow as tf
from policy import Policy
from model import Estimator

from os import path

from dqn import deep_q_learning

if __name__ == '__main__':
    tf.reset_default_graph()

    # Where we save our checkpoints and graphs
    experiment_dir = path.abspath("./experiments/")

    # Create a global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create estimators
    q_estimator = Estimator(scope="q_estimator")
    target_estimator = Estimator(scope="target_q")

    env = Policy()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t, stats in deep_q_learning(sess,
                                        env,
                                        q_estimator=q_estimator,
                                        target_estimator=target_estimator,
                                        experiment_dir=experiment_dir,
                                        num_episodes=100,
                                        replay_memory_size=10000,
                                        replay_memory_init_size=1000,
                                        update_target_estimator_steps=1000,
                                        epsilon_start=1.0,
                                        epsilon_end=0.1,
                                        epsilon_decay_steps=5000,
                                        discount_factor=0.99,
                                        batch_size=32,
                                        num_actions=5):
            print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))
