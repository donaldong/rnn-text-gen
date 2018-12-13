"""author: Donald Dong
"""
import numpy as np


class Batch:
    """A batch has many input and target sequences (`inputs` and `targets`).
    """
    def __init__(self, seqs):
        """Create a batch using a number of sequences. Each input and target
        sequence is a list of encoded labels, and they offset by 1 timestep,
        thus they have the same length. With a sequence `[l0, l0, l1, l1, l2]`,
        the input sequence would be `[l0, l0, l1, l1]`, and the target sequence
        would be `[l0, l1, l1, l2]`.

        Arguments
        ======================================================================
        seqs: int[][][]
            The one-hot encoded sequences.
        """
        self._inputs = [seq[:-1] for seq in seqs]
        self._targets = [seq[1:] for seq in seqs]

    @property
    def inputs(self):
        """The input sequences.
        """
        return self._inputs

    @property
    def targets(self):
        """The target sequences.
        """
        return self._targets


class Dataset:
    """A dataset contains the one-hot encoded text data. It produces
    batches of sequences of encoded labels. We split the text data into batches
    are used to train the RNN, and we sample a random chuck of the text (with
    given length) to evaluate the performance of our data.
    """
    def __init__(
            self,
            filenames,
            seq_length,
            shuffle=True,
    ):
        """Create a dataset.
        Arguments
        ======================================================================
        filenames: string[]
            A list of filenames. They are the paths to one or more plain text
            files. The file contents are concatenated in the given order.

        seq_length: int
            The number of encoded labels in a sequence. It's the one-hot
            encoded output of a slice of consecutive characters in the text.

        shuffle: boolean
            Whether to shuffle the sequences. Default to `True`. When it is set
            to `False`, it will batch the sequences in order of the original
            text.
        """
        text = ''
        vocab = set()
        for filename in filenames:
            with open(filename) as file:
                content = file.read()
                text += content
                vocab = vocab.union(set(content))
        self._seq_length = seq_length
        self._vocab_size = len(vocab)
        self._char_to_ix = {c: i for i, c in enumerate(vocab)}
        self._ix_to_char = list(vocab)
        self._text = text
        self._data = np.array([self._char_to_ix[c] for c in text])
        self._shuffle = shuffle

    def batch(
            self,
            batch_size,
            drop_remainder=True
    ):
        """Batch the instances.
        Arguments
        ======================================================================
        batch_size: int
            The number of instances (sequences) in a single batch.

        drop_remainder: boolean
            Whether the last batch should be dropped in the case of having
            fewer than `batch_size` elements.

        Returns
        ======================================================================
        batches: Generator
            A number of batches which covers the text data.
        """
        n_seq = len(self._data) // self._seq_length
        n_batch = n_seq // batch_size
        seq_ids = np.arange(n_seq)
        if self._shuffle:
            np.random.shuffle(seq_ids)
        i = 0
        for _ in range(n_batch):
            seqs = [None] * batch_size
            for j in range(batch_size):
                k = seq_ids[i] * self._seq_length
                seqs[j] = self._create_seq(k)
                i += 1
            yield Batch(seqs)
        if not drop_remainder:
            seqs = []
            for j in range(n_seq % batch_size):
                k = seq_ids[i] * self._seq_length
                seqs[j] = self._create_seq(k)
                i += 1
            yield Batch(seqs)

    def sample(self, batch_size):
        """Radomly select some sequences (with replacement).
        Arguments
        ======================================================================
        batch_size: int
            The number of instances (sequences) in a single batch.

        Returns
        ======================================================================
        batch: Batch
            A single batch.
        """
        n = len(self._text) - self._seq_length
        return Batch([
            self._create_seq(np.random.randint(n))
            for _ in range(batch_size)
        ])

    def encode(self, text):
        """One-hot encode the text.
        Arguments
        ======================================================================
        text: string
            The original character sequence.

        Returns
        ======================================================================
        seq: int[][]
            The one-hot encoded character sequence.
        """
        return [self._to_label(self._char_to_ix[c]) for c in text]

    def decode(self, seq):
        """Decode the one-hot encoded sequence to text format.
        Arguments
        ======================================================================
        seq: int[][]
            The one-hot encoded character sequence.

        Returns
        ======================================================================
        text: string
            The original character sequence.
        """
        text = ''
        for label in seq:
            text += self._ix_to_char[np.argmax(label)]
        return text

    @property
    def seq_length(self):
        """The number of consecutive characters in a slice of the text data
        (for batching).
        """
        return self._seq_length

    @property
    def vocab_size(self):
        """The number of unique characters in the text data.
        """
        return self._vocab_size

    def _create_seq(self, i):
        j = i + self._seq_length + 1
        return list(map(self._to_label, self._data[i:j]))

    def _to_label(self, index):
        label = np.zeros(self._vocab_size)
        label[index] = 1.0
        return label
