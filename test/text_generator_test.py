"""Tests for `/src/text_generator.py`
"""

import unittest
from src.text_generator import RNNTextGenerator


class TestTextGenerator(unittest.TestCase):
    def test_save_restore(self):
        seq_length = 10
        vocab_size = 4
        text_gen = RNNTextGenerator(
            seq_length,
            vocab_size,
        )
        text_gen.save()

        seq_length = 5
        text_gen = RNNTextGenerator(
            seq_length,
            vocab_size
        )
        text_gen.restore()

    def test_log(self):
        seq_length = 10
        vocab_size = 4
        text_gen = RNNTextGenerator(
            seq_length,
            vocab_size,
            logdir='./tf_logs'
        )
