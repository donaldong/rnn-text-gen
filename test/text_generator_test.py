"""Tests for `/src/text_generator.py`
"""

import unittest
import numpy as np
from src.text_generator import RNNTextGenerator


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
