"""Tests for `/src/text_generator.py`
"""

import unittest
from src.text_generator import RNNTextGenerator


class TestTextGenerator(unittest.TestCase):
    def test_constructor(self):
        input_len = 10111
        unique_chars_count = 10
        text_gen = RNNTextGenerator(input_len, unique_chars_count)
