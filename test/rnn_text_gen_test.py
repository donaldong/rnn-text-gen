"""Tests for `/src/rnn_text_gen.py`
"""

import unittest
from src.rnn_text_gen import RNN_Text_Gen


class TestRNN(unittest.TestCase):
    def test_constructor(self):
        input_len = 10111
        unique_chars_count=10
        d1 = RNN_Text_Gen(input_len, unique_chars_count)

