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
        self.loss = None
        self.input_place = None
        self.output_place = None
        self.rnn_cell = None
        self.dynamic_rnn = None
        self.state = None
        self.optimizer = None
        self.log_dir = None

    def _build_tf_graph(self, seq_length, vocab_size):
        graph = tf.Graph()
        with graph.as_default():
            self.input_place = tf.placeholder(
                tf.float32, shape=(None, seq_length, 1)
            )
            self.output_place = tf.placeholder(
                tf.float32, shape=(None, seq_length, vocab_size)
            )
            with tf.variable_scope(self.name):
                self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(
                    self.neurons_per_cell
                )
                # Dynamic RNN
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
