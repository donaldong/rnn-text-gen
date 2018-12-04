"""Authors: Alexandria Davis, Donald Dong
"""
import tensorflow as tf


class RNNTextGenerator:
    """A text generator using basic cell and dynamic rnn
    """
    def __init__(self):
        # RNN cell
        self.neurons_per_cell = 100
        self.name = 'RNNTextGenerator'
        self.tf_loss = None
        self.tf_input = None
        self.tf_target = None
        self.tf_train = None
        self.tf_logits = None

    def _build_tf_graph(self, seq_length, vocab_size):
        """TF Graph Construction
        """
        graph = tf.Graph()
        with graph.as_default():
            self.tf_input = tf.placeholder(
                tf.float32, shape=(None, seq_length, 1)
            )
            self.tf_target = tf.placeholder(
                tf.float32, shape=(None, seq_length, vocab_size)
            )
            with tf.variable_scope(self.name):
                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(
                    self.neurons_per_cell
                )
                dynamic_rnn, states = tf.nn.dynamic_rnn(
                    rnn_cell, self.tf_input, dtype=tf.float32
                )
                self.tf_logits = tf.layers.dense(states, vocab_size)
                self.tf_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.tf_logits,
                        labels=self.tf_target
                    )
                )
                self.tf_train = tf.train.AdamOptimizer().minimize(
                    self.tf_loss
                )

    def fit(self, inputs, targets, vocab_size):
        """Fit and train the classifier
        Arguments
        ======================================================================
        inputs: np.ndarray
            A batch of input sequences.

        targets: np.ndarray
            A batch of target sequences.

        vocab_size: int
            The number of unique characters in the text
        """
        self._build_tf_graph(inputs.shape[1], vocab_size)

    def predict(self):
        pass

    def score(self):
        pass
