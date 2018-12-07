"""author: Donald Dong
"""
import numpy as np
import tensorflow as tf


class Dataset:
    """A wrapper on top of `tf.data.Dataset` to fetch batches
    """
    def __init__(
            self,
            filenames,
            seq_length,
            shuffle=True,
            buffer_size=10000,
    ):
        """Creates a dataset
        Arguments
        ======================================================================
            filenames: string
                Path to one or more plain text files.
                The file contents are concatenated in the given order.

            seq_length: int
                The length of the text sequence.

            shuffle: boolean
                Whether to shuffle the sequences for the batches.

            buffer_size: int
                 The number of elements from this dataset from which the new
                 dataset will sample.
        """
        text = ''
        vocab = set()
        for filename in filenames:
            content = open(filename).read()
            text += content
            vocab = vocab.union(set(content))
        self.vocab_size = len(vocab)
        self.char_to_ix = {c: i for i, c in enumerate(vocab)}
        self.ix_to_char = list(vocab)
        self.text = text
        self.data = np.array([self.char_to_ix[c] for c in text])

        dataset = tf.data.Dataset.from_tensor_slices(
            tf.one_hot(self.data, self.vocab_size)
        )
        dataset = dataset.batch(seq_length + 1, drop_remainder=True)
        self.instances = dataset.map(lambda seq: {
            'input': seq[:-1],
            'target': seq[1:],
        })
        if shuffle:
            self.instances = self.instances.shuffle(
                buffer_size,
                reshuffle_each_iteration=True
            )

    def batch(
            self,
            batch_size,
            drop_remainder=True
    ):
        """Batch the instances
        Arguments
        ======================================================================
            batch_size: int
                The number of instances in a single batch.

            drop_remainder: boolean
                Whether the last batch should be dropped in the case its has
                fewer than batch_size elements.
        """
        return self.instances.batch(batch_size, drop_remainder)


    def _to_label(self, index):
        label = np.zeros(self.vocab_size)
        label[index] = 1.0
        return label

    def encode(self, text):
        """One-hot encode the text
        Arguments
        ======================================================================
            text: string
                The text to encode.

        Returns
        ======================================================================
            seq: int[]
                The one-hot encoded sequence.
        """
        return [self._to_label(self.char_to_ix[c]) for c in text]

    def decode(self, seq):
        """Decode the one-hot encoded sequence to text format
        Arguments
        ======================================================================
            text: string
                The text to encode.

        Returns
        ======================================================================
            seq: int[]
                The one-hot encoded sequence.
        """
        text = ''
        for label in seq:
            text += self.ix_to_char[np.argmax(label)]
        return text
