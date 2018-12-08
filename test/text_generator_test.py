"""Tests for `/src/text_generator.py`
"""

import unittest
from src.text_generator import RNNTextGenerator
        print("---------------Testing text generator with randomly generated data-------------")
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
        print("-------------Testing logs---------------")
        seq_length = 10
        vocab_size = 4
        text_gen = RNNTextGenerator(
            seq_length,
            vocab_size,
            logdir='./tf_logs'
        )
