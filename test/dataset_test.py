"""Tests for `/src/dataset.py`
"""

import unittest
from src.dataset import Dataset


def test_batch(filename, batch_size, seq_length):
    dataset = Dataset([filename], seq_length)
    for batch in dataset.batch(batch_size):
        # The number of elements in the batch is `batch_size`
        assert len(batch.inputs) == batch_size
        assert len(batch.targets) == batch_size
        for i in range(batch_size):
            # Each element in the batch is a sequence
            assert len(batch.inputs[i]) == seq_length
            assert len(batch.targets[i]) == seq_length
            for j in range(seq_length):
                # One-hot encoded
                assert sum(batch.inputs[i][j]) == 1
                assert len(batch.inputs[i][j]) == dataset.vocab_size


def test_sample(filename, batch_size, seq_length):
    dataset = Dataset([filename], seq_length)
    count = 0
    batch = dataset.sample(batch_size)
    for seq in batch.inputs:
        assert len(seq) == seq_length
        for i in range(seq_length):
            # One-hot encoded
            assert sum(seq[i]) == 1
            assert len(seq[i]) == dataset.vocab_size
        count += 1
    assert count == batch_size


def test_encode(filename, seq_length, text):
    dataset = Dataset([filename], seq_length)
    encoded = dataset.encode(text)
    assert len(encoded) == len(text)
    for label in encoded:
        assert sum(label) == 1
        assert len(label) == dataset.vocab_size

class TestDataset(unittest.TestCase):
    def test(self):
        print("-----------Testing Dataset Module -----------")
        batch_size = 5
        seq_length = 4
        filename = 'data/test.txt'
        test_batch(filename, batch_size, seq_length)
        test_sample(filename, batch_size, seq_length)
        test_encode(filename, seq_length, 'ab')

