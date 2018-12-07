"""Tests for `/src/text_generator.py`
"""

import unittest
import numpy as np
from src.text_generator import RNNTextGenerator
from src.dataset import Dataset


def random_label(vocab_size):
    """randomly assign a label
    """
    label = np.random.randint(vocab_size)
    seq = np.zeros(vocab_size)
    seq[label] = 1.0
    return seq


def random_data(batch_size, seq_length, vocab_size):
    """generate random data
    """
    inputs = []
    targets = []
    for _ in range(batch_size):
        labels = [random_label(vocab_size) for _ in range(seq_length + 1)]
        inputs.append(labels[:-1])
        targets.append(labels[1:])
    return np.array(inputs), np.array(targets)


class TestTextGenerator(unittest.TestCase):
    def test_on_random_data(self):
        seq_length = 10
        vocab_size = 4
        batch_size = 2
        text_gen = RNNTextGenerator(
            seq_length,
            vocab_size,
        )
        print('first fit')
        inputs, targets = random_data(batch_size, seq_length, vocab_size)
        print('fit:', text_gen.fit(inputs, targets))
        print('score:', text_gen.score(inputs, targets))
        print('predictions:', text_gen.predict(inputs))
        print('true targets:', targets)

    def test_combo(self):
        batch_size = 5
        seq_length = 25
        filename = 'data/alice.txt'
        dataset = Dataset([filename], seq_length)
