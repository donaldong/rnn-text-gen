"""author: Donald Dong
"""
import numpy as np


class Batch:
    def __init__(self, seqs):
        """Create a batch using the sequence
        Arguments
        ======================================================================
        seqs: int[][][]
            The one-hot encoded sequences.
        """
        self.inputs = [seq[:-1] for seq in seqs]
        self.targets = [seq[1:] for seq in seqs]


class Dataset:
    def __init__(
            self,
            filenames,
            seq_length,
            shuffle=True,
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
        """
        text = ''
        vocab = set()
        for filename in filenames:
            with open(filename) as file:
                content = file.read()
                text += content
                vocab = vocab.union(set(content))
        self.seq_length = seq_length
        self.vocab_size = len(vocab)
        self.char_to_ix = {c: i for i, c in enumerate(vocab)}
        self.ix_to_char = list(vocab)
        self.text = text
        self.data = np.array([self.char_to_ix[c] for c in text])
        self.shuffle = shuffle

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
        n_seq = len(self.data) // self.seq_length
        n_batch = n_seq // batch_size
        seq_ids = np.arange(n_seq)
        if self.shuffle:
            np.random.shuffle(seq_ids)
        i = 0
        for _ in range(n_batch):
            seqs = [None] * batch_size
            for j in range(batch_size):
                k = seq_ids[i] * self.seq_length
                seqs[j] = self._create_seq(k)
                i += 1
            yield Batch(seqs)
        if not drop_remainder:
            seqs = []
            for j in range(n_seq % batch_size):
                k = seq_ids[i] * self.seq_length
                seqs[j] = self._create_seq(k)
                i += 1
            yield Batch(seqs)

    def sample(self, batch_size):
        """Radomly select some sequences with replacement
        Arguments
        ======================================================================
        batch_size: int
            The number of instances in a single batch.
        """
        n = len(self.text) - self.seq_length
        return Batch([
            self._create_seq(np.random.randint(n))
            for _ in range(batch_size)
        ])

    def encode(self, text):
        """One-hot encode the text
        Arguments
        ======================================================================
        text: string
            The text to encode.

        Returns
        ======================================================================
        seq: int[][]
            The one-hot encoded sequence.
        """
        return [self._to_label(self.char_to_ix[c]) for c in text]

    def decode(self, seq):
        """Decode the one-hot encoded sequence to text format
        Arguments
        ======================================================================
        seq: int[][]
            The one-hot encoded sequence.

        Returns
        ======================================================================
        text: string
            The decoded text.
        """
        text = ''
        for label in seq:
            text += self.ix_to_char[np.argmax(label)]
        return text

    def _create_seq(self, i):
        j = i + self.seq_length + 1
        return list(map(self._to_label, self.data[i:j]))

    def _to_label(self, index):
        label = np.zeros(self.vocab_size)
        label[index] = 1.0
        return label
