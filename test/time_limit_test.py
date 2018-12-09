"""Tests for `/src/time_limit.py`
"""

import unittest
from src.time_limit import time_limit
from datetime import datetime


class TestTimeLimit(unittest.TestCase):
    def test(self):
        print(datetime.now())
        for _ in time_limit(minutes=2):
            pass
        print(datetime.now())
