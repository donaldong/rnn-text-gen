"""Authors: Alexandria Davis, Donald Dong
"""
import tensorflow as tf
import numpy as np
import pandas as pd


class RNNTextGenerator:
    """Creates a recurrent neural network with a tensorflow RNNCell cell (which
    performs dynamic unrolling of the `inputs`). It has an output projection
    layer which produces the final probability for each character class. It
    generates the text by sampling the next character based on the probability
    distribution of the last character of the current sequence.
    """
    def __init__(
            self,
            seq_length,
            vocab_size,
            rnn_cell=tf.nn.rnn_cell.BasicRNNCell,
            n_neurons=100,
            activation=tf.tanh,
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.001,
            epoch=5,
            batch_size=25,
            name='RNNTextGenerator',
            logdir=None,
    ):
        """Initialize the text generator and contruct the TensorFlow graph.
        Arguments
        ======================================================================
        seq_length: int
            The number of encoded labels in a sequence.

        vocab_size: int
            The number of unique characters in the text.

        rnn_cell: tf.nn.rnn_cell.*
            An RNN cell from `tf.nn.rnn_cell`. The cell has `n_neurons`
            neurons, takes the `activation` funtions, and goes into
            `tf.nn.dynamic_rnn`.

        n_neurons: int
            The number of neurons in the RNN cell.

        activation: Callable
            The activation function (callable) for the RNN cell.

        optimizer: tf.train.Optimizer
            A subclass of `tf.train.Optimizer`. The optimizer to use for
            minizing the loss.

        learning_rate:
            A Tensor or a floating point value. The learning rate of the
            optimizer.

        epoch: int
            The number of times to iterate through the dataset.

        batch_size: int
            The number of instances (sequences) in a single batch.

        name: string
            The name of the text generator. It is used for graph visualization
            in tensorboard (variable scope), and saving/restoring the model
            (checkpoint name).

        logdir: string
            The path to the tensorflow summary.
        """
        self._batch_size = batch_size
        self._epoch = epoch
        self._name = name
        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            self._tf_sess = tf.Session()
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
            self._tf_input = tf.placeholder(
                tf.float32,
                shape=(None, seq_length, vocab_size),
                name='inputs',
            )
            self._tf_target = tf.placeholder(
                tf.float32,
                shape=(None, seq_length, vocab_size),
                name='targets'
            )
            with tf.variable_scope(name):
                self._tf_rnn_cell = rnn_cell(
                    n_neurons,
                    activation=activation,
                )
                outputs, _ = tf.nn.dynamic_rnn(
                    self._tf_rnn_cell,
                    tf.cast(self._tf_input, tf.float32),
                    dtype=tf.float32,
                )
                logits = tf.layers.dense(
                    outputs,
                    vocab_size,
                    name='output_projection',
                )
                self._tf_prob = tf.nn.softmax(
                    logits,
                    name='probability',
                )
                with tf.variable_scope('loss'):
                    self._tf_loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                            logits=logits,
                            labels=self._tf_target,
                        )
                    )
                self._tf_train = optimizer(
                    learning_rate=learning_rate
                ).minimize(self._tf_loss)
                with tf.variable_scope('accuracy'):
                    self._tf_acc = tf.reduce_mean(tf.cast(
                        tf.equal(
                            tf.argmax(logits, 2),
                            tf.argmax(self._tf_target, 2),
                        ),
                        tf.float32,
                    ))
                self._tf_saver = tf.train.Saver()
                if logdir is not None:
                    self.logger = tf.summary.FileWriter(logdir, self._tf_graph)
            # Initialize the tf session
            self._tf_sess.run(tf.global_variables_initializer())
            self._tf_sess.run(tf.local_variables_initializer())
            self._params = {
                'vocab_size': vocab_size,
                'rnn_cell': rnn_cell,
                'n_neurons': n_neurons,
                'activation': activation,
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'epoch': epoch,
                'batch_size': batch_size,
                'name': name,
            }

    @property
    def params(self):
        """The parameters which define the text generator.
        """
        return self._params

    def fit(self, dataset, save_scores=False):
        """Feed the dataset `epoch` times, with batches of `batch_size`
        sequences.

        Arguments
        ======================================================================
        dataset: Dataset
            A `Dataset` object which creates batches to train the model.

        save_scores: boolean
            Whether to store the training accuracy and loss.

        Returns
        ======================================================================
            If `save_scores` is `True`, it returns a `pd.DataFrame` which
            stores the training accuracy and loss.
        """
        accs = []
        losses = []
        for _ in range(self._epoch):
            for batch in dataset.batch(self._batch_size):
                self._tf_sess.run(
                    self._tf_train,
                    feed_dict={
                        self._tf_input: batch.inputs,
                        self._tf_target: batch.targets,
                    },
                )
                if save_scores:
                    acc, loss = self._score(batch.inputs, batch.targets)
                    accs.append(acc)
                    losses.append(loss)
        if save_scores:
            return pd.DataFrame({
                'accuracy': accs,
                'loss': losses,
            })

    def score(self, dataset, batch_size=None, n_samples=5):
        """Measure the score of the text generator. The score is the average
        result of `n_samples` times sampling from the dataset. It tests how the
        model will perform on sequences it has not *completely* seen yet.

        Arguments
        ======================================================================
        dataset: Dataset
            A `Dataset` object to sample from. A sample is a single `Batch`.

        batch_size: int | None
            The number of instances (sequences) in a single batch. When
            `batch_size` is `None`, it uses the `batch_size` for training the
            model.

        n_samples: int
            The number of times to sample from the dataset for testing.

        Returns
        ======================================================================
        accuracy: tf.float32
            The average accuracy of the `n_samples` samples.

        loss: tf.float32
            The average loss of the `n_samples` samples.
        """
        if batch_size is None:
            batch_size = self._batch_size
        acc = [None] * n_samples
        loss = [None] * n_samples
        for i in range(n_samples):
            batch = dataset.sample(batch_size)
            acc[i], loss[i] = self._score(batch.inputs, batch.targets)
        return np.mean(acc), np.mean(loss)

    def predict(self, inputs):
        """Predict the probabilities of the next labels in each input sequence.

        Arguments
        ======================================================================
        inputs: tf.float32[][][]
            The input sequences (with one-hot encoded labels).

        Returns
        ======================================================================
        targets_prob: tf.float32[][][]
            The target sequences (with the probabilities of each label). The
            shape of the target sequences would be `[len(inputs), seq_length,
            vocab_size]`.
            
        """
        return self._tf_sess.run(
            self._tf_prob,
            feed_dict={
                self._tf_input: inputs,
            },
        )

    def save(self, path='./model'):
        """Save the model in the specified path. The files use the `name` of
        the text generator.

        Arguments
        ======================================================================
        path: string
            The path to store the model.
        """
        self._tf_saver.save(
            self._tf_sess,
            path + '/' + self._name
        )

    def restore(self, path='./model'):
        """Restore the model in the specified path. It assumes the files use
        the `name` of the text generator exists, or it throws exceptions.

        Arguments
        ======================================================================
        path: string
            The path where the model is stored.
        """
        self._tf_saver.restore(
            self._tf_sess,
            path + '/' + self._name
        )

    def sample(self, dataset, start_seq, length):
        """Sample from the text generator based on the predicted probability
        distribution for the next label. For example, assume the target for the
        input sequence `[l1, l1, l2]` is `[[l1: 90%, l2: 10%], [l1: 10%, l2:
        90%], [l1: 10%, l2: 90%]]`, the next character is sampled from `[l1:
        10%, l2: 90%]`. Thus the next character would be `l2 ` with a
        probability of `0.9`, or `l1` with a probability of `0.1`.

        Arguments
        ======================================================================
        dataset: Dataset
            A `Dataset` object to encode and decode the labels. This method is
            sampling from the text generator, not from the dataset.

        start_seq: string
            The character sequence to begin with.

        length: int
            The length of the generated text.

        Returns
        ======================================================================
        text: string
            The sampled text with `length` characters.
        """
        text = [None] * length
        seq = dataset.encode(start_seq)
        for i in range(length):
            ix = np.random.choice(
                range(dataset.vocab_size),
                # pred[batch 0][last item in the sequence]
                p=self.predict([seq])[0][-1]
            )
            x = np.zeros(dataset.vocab_size)
            x[ix] = 1
            del seq[0]
            seq.append(x)
            text[i] = x
        return dataset.decode(text)

    def generate(self, dataset, start_seq, length):
        """Generate the text from the text generator using the given
        `start_seq`. This method wraps the `sample`. It creates a new model
        with the new sequence length and restores the previous weights.

        Arguments
        ======================================================================
        dataset: Dataset
            A `Dataset` object to encode and decode the labels. This method is
            sampling from the text generator, not from the dataset.

        start_seq: string
            The character sequence to begin with.

        length: int
            The length of the generated text.

        Returns
        ======================================================================
        text: string
            The generated text with `length` characters.
        """
        self.save()
        model = RNNTextGenerator(
            len(start_seq),
            **self._params
        )
        model.restore()
        return model.sample(
            dataset,
            start_seq,
            length
        )

    def _score(self, inputs, targets):
        return self._tf_sess.run(
            [self._tf_acc, self._tf_loss],
            feed_dict={
                self._tf_input: inputs,
                self._tf_target: targets,
            },
        )

    def __repr__(self):
        return repr(self._params)

    def __str__(self):
        return str(self._params)
