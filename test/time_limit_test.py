"""Tests for `/src/time_limit.py`
"""

import unittest
from src.time_limit import time_limit
from datetime import datetime


class TestTimeLimit(unittest.TestCase):
    def test(self):
        x=0
        print(datetime.now())
        for _ in time_limit(seconds=30):
            pass
            x+=1
        print(datetime.now())
        print("Itterations: ",x)
