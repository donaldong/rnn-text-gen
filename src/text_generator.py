"""Authors: Alexandria Davis, Donald Dong
"""
import tensorflow as tf
import numpy as np


class RNNTextGenerator:
    """A text generator using basic cell and dynamic rnn
    """
    def __init__(
            self,
            seq_length,
            vocab_size,
            rnn_cell=tf.nn.rnn_cell.BasicRNNCell,
            n_neurons=100,
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.001,
            name='RNNTextGenerator',
            logdir=None
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
        self.name = name
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.tf_sess = tf.Session()
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
                tf.float32, shape=(None, seq_length, vocab_size)
            )
            self.tf_target = tf.placeholder(
                tf.float32, shape=(None, seq_length, vocab_size)
            )
            with tf.variable_scope(name):
                self.tf_rnn_cell = rnn_cell(n_neurons)
                outputs, _ = tf.nn.dynamic_rnn(
                    self.tf_rnn_cell,
                    tf.cast(self.tf_input, tf.float32),
                    dtype=tf.float32,
                )
                logits = tf.layers.dense(outputs, vocab_size)
                self.tf_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=self.tf_target,
                    )
                )
                self.tf_train = optimizer(
                    learning_rate=learning_rate
                ).minimize(self.tf_loss)
                # Normilize the probablities
                y = tf.math.exp(logits)
                self.tf_prob = y / tf.reduce_sum(y, 2, keep_dims=True)
                self.tf_acc = tf.reduce_mean(tf.cast(
                    tf.equal(
                        tf.argmax(logits, 2),
                        tf.argmax(self.tf_target, 2),
                    ),
                    tf.float32
                ))
                self.tf_saver = tf.train.Saver()
                if logdir is not None:
                    self.logger = tf.summary.FileWriter(logdir, self.tf_graph)
            # Initialize the tf session
            self.tf_sess.run(tf.global_variables_initializer())
            self.tf_sess.run(tf.local_variables_initializer())

    def fit(self, dataset, epoch, batch_size):
        """Fit and train the classifier with a batch of inputs and targets
        Arguments
        ======================================================================
        inputs: np.ndarray
            A batch of input sequences.

        targets: np.ndarray
            A batch of target sequences.
        """
        for _ in range(epoch):
            for batch in dataset.batch(batch_size):
                self.tf_sess.run(
                    self.tf_train,
                    feed_dict={
                        self.tf_input: batch.inputs,
                        self.tf_target: batch.targets,
                    },
                )
        return self

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

    def predict(self, inputs):
        """Predict the probablities for the labels, for a batch of inputs
        Arguments
        ======================================================================
        inputs: np.ndarray
            A batch of input sequences.


        Returns
        ======================================================================
        predictions: np.ndarray
            A batch of sequences of probablities.
        """
        return self.tf_sess.run(
            self.tf_prob,
            feed_dict={
                self.tf_input: inputs,
            },
        )

    def save(self, path='./model'):
        """Save the model
        Arguments
        ======================================================================
        path: string
            The path to store the model.
        """
        self.tf_saver.save(
            self.tf_sess,
            path + '/' + self.name
        )

    def restore(self, path='./model'):
        """Restore the model
        Arguments
        ======================================================================
        path: string
            The path to store the weights.
        """
        self.tf_saver.restore(
            self.tf_sess,
            path + '/' + self.name
        )

    @staticmethod
    def sample(model, dataset, start_seq, length):
        """Generate the text using a saved model
        Arguments
        ======================================================================
        model: RNNTextGenerator
            The model to sample from.

        dataset: Dataset
            The dataset to encode and decode the labels.

        start_seq: int[]
            The sequence to begin with.

        length: int
            The length of the generated text.

        Returns
        ======================================================================
        text: int[]
            The one-hot encoded character labels.
        """
        text = [None] * length
        seq = dataset.encode(start_seq)
        for i in range(length):
            ix = np.random.choice(
                range(dataset.vocab_size),
                # pred[batch 0][last item in the sequence]
                p=model.predict([seq])[0][-1]
            )
            x = np.zeros(dataset.vocab_size)
            x[ix] = 1
            del seq[0]
            seq.append(x)
            text[i] = x
        return dataset.decode(text)
