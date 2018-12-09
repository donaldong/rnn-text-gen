"""Tests for `/src/time_limit.py`
"""

import unittest
from src.time_limit import time_limit


class TestTimeLimit(unittest.TestCase):
    def test(self):
        for _ in time_limit(seconds=2):
            print('test')
