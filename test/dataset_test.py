"""Tests for `/src/dataset.py`
"""

import unittest
from src.dataset import Dataset


class TestDataset(unittest.TestCase):
    # Enable eager execution first
    import tensorflow as tf
    tf.enable_eager_execution()

    def test_constructor(self):
        filename = 'data/alice.txt'
        seq_length = 100
        d1 = Dataset([filename], seq_length)
        d2 = Dataset([filename] * 2, seq_length)
        self.assertTrue(d1.text + d1.text == d2.text)

    def test_batch_a_seq(self):
        batch_size = 5
        seq_length = 100
        filename = 'data/alice.txt'
        dataset = Dataset([filename], seq_length)
        for batch in dataset.batch(batch_size):
            self.assertTrue(batch['input'].shape == (batch_size, seq_length))
            self.assertTrue(batch['target'].shape == (batch_size, seq_length))
