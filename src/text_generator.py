"""Authors: Alexandria Davis, Donald Dong
"""
import tensorflow as tf


class RNNTextGenerator:
    """A text generator using basic cell and dynamic rnn
    """
    def __init__(
            self,
            seq_length,
            vocab_size,
            neurons_per_cell,
            name='RNNTextGenerator'
    ):
        """Initialize the text generator and contruct the tf graph
        Arguments
        ======================================================================
        seq_length: int
            The number of characters in a sequence.

        vocab_size: int
            The number of unique characters in the text.

        neurons_per_cell: int
            The number of neurons in each RNN cell.

        name: string
            The name of the net (for graph visualization in tensorboard).
        """
        graph = tf.Graph()
        self.tf_sess = tf.Session(graph=graph)
        with graph.as_default():
            # One-hot encoded input and targets
            """placeholder
            Example
            [
                batch_0: [
                    seq_0: [
                        # encoded labels with 5 categories
                        [0, 0, 0, 1, 0],  # i = 0
                        [0, 0, 1, 0, 0],  # i = 1
                    ],
                    ...
                ],
                ...
            ]
            """
            self.tf_input = tf.placeholder(
                tf.int32, shape=(None, seq_length, vocab_size)
            )
            self.tf_target = tf.placeholder(
                tf.int32, shape=(None, seq_length, vocab_size)
            )
            with tf.variable_scope(name):
                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(neurons_per_cell)
                outputs, _ = tf.nn.dynamic_rnn(
                    rnn_cell,
                    tf.cast(self.tf_input, tf.float32),
                    dtype=tf.float32,
                )
                self.tf_logits = tf.layers.dense(outputs, vocab_size)
                self.tf_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.tf_logits,
                        labels=self.tf_target,
                    )
                )
                self.tf_train = tf.train.AdamOptimizer().minimize(
                    self.tf_loss
                )
                self.tf_acc = tf.reduce_mean(tf.cast(
                    tf.equal(
                        tf.argmax(self.tf_logits, 1),
                        tf.argmax(self.tf_target, 1),
                    ),
                    tf.float32
                ))
            # Initialize the tf session
            self.tf_sess.run(tf.global_variables_initializer())
            self.tf_sess.run(tf.local_variables_initializer())

    def fit(
            self,
            inputs,
            targets,
    ):
        """Fit and train the classifier with a batch of inputs and targets
        Arguments
        ======================================================================
        inputs: np.ndarray
            A batch of input sequences.

        targets: np.ndarray
            A batch of target sequences.
        """
        self.tf_sess.run(
            self.tf_train,
            feed_dict={
                self.tf_input: inputs,
                self.tf_target: targets,
            },
        )
        return self

    def predict(self):
        pass

    def score(self, inputs, targets):
        """Get the score for the batch
        Arguments
        ======================================================================
        inputs: np.ndarray
            A batch of input sequences.

        targets: np.ndarray
            A batch of target sequences.

        Returns
        ======================================================================
        accuracy: tf.float32
            The accuracy on this batch.

        loss: tf.float32
            The loss on this batch.
        """
        return self.tf_sess.run(
            [self.tf_acc, self.tf_loss],
            feed_dict={
                self.tf_input: inputs,
                self.tf_target: targets,
            },
        )
