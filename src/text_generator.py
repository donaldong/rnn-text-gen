"""Authors: Alexandria Davis, Donald Dong
"""
import tensorflow as tf


class RNNTextGenerator:
    """A text generator using basic cell and dynamic rnn
    """
    def __init__(self, input_len, unique_chars_count):
        # RNN cell
        self.neurons_per_cell = 100
        self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.neurons_per_cell)
        # Dynamic RNN
        self.output_place = tf.placeholder(
            tf.float32, shape=(None, input_len, unique_chars_count)
        )
        self.input_place = tf.placeholder(
            tf.float32, shape=(None, input_len, 1)
        )
        self.dynamic_rnn, self.state = tf.nn.dynamic_rnn(
            self.rnn_cell, self.input_place, dtype=tf.float32
        )
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.dynamic_rnn,
                labels=self.output_place
            )
        )
        self.optimizer = tf.train.AdamOptimizer().minimize(cost)
        self.loss = None

    def predict(self):
        pass

    def score(self):
        pass

    def fit(self):
        pass
