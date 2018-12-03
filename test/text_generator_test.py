"""Tests for `/src/text_generator.py`
"""

import unittest
import numpy as np
from src.text_generator import RNNTextGenerator
from src.dataset import Dataset


class TestTextGenerator(unittest.TestCase):
    def test_constructor(self):
        text_gen = RNNTextGenerator()

    def test_fit(self):
        text_gen = RNNTextGenerator()
        inputs = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [1, 1, 2],
        ])
        text_gen.fit(inputs, inputs, 2)
    def test_combo(self):
        batch_size = 5
        seq_length = 25
        filename = 'data/alice.txt'
        dataset = Dataset([filename], seq_length)
        
