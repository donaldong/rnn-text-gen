"""An end to end test using 
 ALICE'S ADVENTURES IN WONDERLAND
"""

import tensorflow as tf
tf.enable_eager_execution()

import unittest
import numpy as np
from src.text_generator import RNNTextGenerator
from src.dataset import Dataset


class TestAlice(unittest.TestCase):
    def test_alice(self):
        seq_length = 25
        batch_size = 25
        epoch = 5
        dataset = Dataset(['./data/alice.txt'], seq_length)
        model = RNNTextGenerator(seq_length, dataset.vocab_size)
        for _ in range(epoch):
            for batch in dataset.batch(batch_size):
                model.fit(batch['input'].numpy(), batch['target'].numpy())
        start_seq = 'h'
        text_gen = RNNTextGenerator(
            len(start_seq),
            dataset.vocab_size
        )
        print('generated text: ', dataset.decode(RNNTextGenerator.sample(
            text_gen,
            dataset.encode(start_seq),
            50
        )))
