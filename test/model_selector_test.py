"""Tests for `/src/model_selector.py`
"""

import unittest
from src.text_generator import RNNTextGenerator
from src.dataset import Dataset
from src.model_selector import ModelSelector
import numpy as np
import tensorflow as tf


def test_model_selector(dataset, params, n):
    selector = ModelSelector(dataset, params)
    for _ in range(n):
        selector.search()
    return selector.as_df()


class TestModelSelector(unittest.TestCase):
    def test(self):
        seq_length = 25
        filename = './data/alice.txt'
        dataset = Dataset([filename], seq_length)
        params = {
            'rnn_cell': [
                tf.contrib.rnn.BasicRNNCell
            ],
            'n_neurons': np.arange(1, 1000),
            'optimizer': [
                tf.train.AdamOptimizer,
            ],
            'learning_rate': np.linspace(0, 1, 10000, endpoint=False),
            'epoch': np.arange(1, 6),
            'batch_size': np.arange(25, 100),
        }
        print(test_model_selector(dataset, params, 3))
