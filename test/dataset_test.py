"""Tests for `/src/dataset.py`
"""

import unittest
from src.dataset import Dataset


class TestDataset(unittest.TestCase):
    # Enable eager execution first
    import tensorflow as tf
    tf.enable_eager_execution()

    def test_batch_a_seq(self):
        batch_size = 5
        seq_length = 100
        filename = 'data/alice.txt'
        dataset = Dataset([filename], seq_length)
        for i, batch in enumerate(dataset.batch(batch_size)):
            print(i, 'th batch shape:', batch['input'].shape)
