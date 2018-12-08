"""Tests for `/src/dataset.py`
"""

import unittest
from src.dataset import Dataset


class TestDataset(unittest.TestCase):
    def test_batch_a_seq(self):
        batch_size = 5
        seq_length = 100
        filename = 'data/alice.txt'
        dataset = Dataset([filename], seq_length)
        for batch in dataset.batch(batch_size):
            self.assertTrue(len(batch.inputs) == batch_size)
            self.assertTrue(len(batch.targets) == batch_size)
            self.assertTrue(len(batch.inputs[-1]) == seq_length)
            self.assertTrue(len(batch.targets[-1]) == seq_length)
