"""Tests for `/src/dataset.py`
"""

import unittest
from src.dataset import Dataset


class TestDataset(unittest.TestCase):
    def test_constructor(self):
        filename = 'data/alice.txt'
        seq_length = 100
        d1 = Dataset([filename], seq_length)
        d2 = Dataset([filename] * 2, seq_length)
        self.assertTrue(d1.text + d1.text == d2.text)
