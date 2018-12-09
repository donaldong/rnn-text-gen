"""Tests for `/src/text_generator.py`
"""

import unittest
from src.text_generator import RNNTextGenerator


def test_save_restore(l1, l2, vocab_size):
    text_gen = RNNTextGenerator(l1, vocab_size)
    text_gen.save()

    text_gen = RNNTextGenerator(l2, vocab_size)
    text_gen.restore()


def test_log(seq_length, vocab_size, logdir):
    import os
    RNNTextGenerator(
        seq_length,
        vocab_size,
        logdir=logdir,
    )
    assert os.path.exists(logdir)


class TestTextGenerator(unittest.TestCase):
    def test(self):
        print("---------------Testing text generator -------------")
        test_save_restore(4, 5, 10)
        test_log(4, 10, './tf_logs')
