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


class TestDataset(unittest.TestCase):
    def test(self):
        print("-----------Testing Dataset Module -----------")
        batch_size = 5
        seq_length = 100
        filename = 'data/alice.txt'
        test_batch(filename, batch_size, seq_length)
