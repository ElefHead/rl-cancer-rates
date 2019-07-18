import tensorflow as tf
from os import path, makedirs
from policy import Policy
import numpy as np


def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


class Estimator():
    def __init__(self, scope="estimator", num_states=10, num_actions=5, summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        self.input_shape = (None, num_states)
        self.num_actions = num_actions
        with tf.variable_scope(scope, initializer="he_normal"):
            # Build Graph
            self.build_model_()
            if summaries_dir:
                summaries_dir = path.join(summaries_dir, "summaries_{}".format(scope))
                if not path.exists(summaries_dir):
                    makedirs(summaries_dir)
                self.summary_writer = tf.summary.FileWriter(summaries_dir)

    def build_model_(self):
        self.x_placeholder = tf.placeholder(shape=self.input_shape, dtype=tf.float32, name="x")
        self.y_placeholder = tf.placeholder(shape=(None), dtype=tf.float32, name="y")
        self.action_placeholder = tf.placeholder(shape=(None), dtype=tf.int32, name="actions")

        # X = tf.reshape(self.x_placeholder, self.input_shape)

        layer1 = tf.contrib.layers.fully_connected(self.x_placeholder, 200)
        layer2 = tf.contrib.layers.fully_connected(layer1, 200)

        self.predictions = tf.contrib.layers.fully_connected(layer2, self.num_actions)

        train_shape = tf.shape(self.x_placeholder)
        pred_shape = tf.shape(self.predictions)

        indices = tf.range(train_shape[0]) * pred_shape[1] + self.action_placeholder
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), indices)

        self.losses = tf.squared_difference(self.y_placeholder, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        return sess.run(self.predictions, feed_dict={ self.x_placeholder: s })

    def update(self, sess, s, a, y):
        feed_dict = {self.x_placeholder: s, self.y_placeholder: y, self.action_placeholder: a}
        summaries, global_step, _, loss = sess.run([
            self.summaries,
            tf.contrib.framework.get_global_step(),
            self.train_op,
            self.loss
        ], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

        return loss


def main():
    tf.reset_default_graph()
    global_step = tf.Variable(0, name="global_step", trainable=False)

    e = Estimator(scope="test")
    env = Policy()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        observation = env.reset()

        observation.reshape((-1, observation.shape[-1]))

        print(e.predict(sess, observation))

        y = np.array([2.0])
        a = np.array([1])

        print(e.update(sess, observation, a, y))


if __name__ == '__main__':
    main()
