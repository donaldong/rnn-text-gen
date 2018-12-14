"""Tests for `/src/text_generator.py`
"""

import unittest
from src.text_generator import RNNTextGenerator
from src.dataset import Dataset


def test_load(filename, start_seq):
    seq_length = 25
    dataset = Dataset([filename], seq_length)
    model = RNNTextGenerator(
        25,
        dataset.vocab_size,
        meta_graph='./model/RNNTextGenerator'
    )
    print(model.generate(
        dataset,
        start_seq,
        50
    ))


class TestLoadModel(unittest.TestCase):
    def test(self):
        test_load('data/alice.txt', 'alice')
