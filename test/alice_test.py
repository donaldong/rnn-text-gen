"""An end to end test using
 ALICE'S ADVENTURES IN WONDERLAND
"""
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)

import unittest
from src.text_generator import RNNTextGenerator
from src.dataset import Dataset

def test_alice(filename, start_seq):
    seq_length = 25
    batch_size = 25
    learning_rate = 0.01
    epoch = 5
    dataset = Dataset([filename], seq_length)
    model = RNNTextGenerator(
        seq_length,
        dataset.vocab_size,
        learning_rate=learning_rate,
        epoch=epoch,
        batch_size=batch_size,
    )
    scores = model.fit(dataset, save_scores=True)
    print('>>>>>>\n{}'.format(start_seq))
    print(model.generate(
        dataset,
        start_seq,
        50
    ))
    print('<<<<<<')
    return scores

def test_restore(filename, start_seq):
    seq_length = 25
    batch_size = 25
    learning_rate = 0.01
    epoch = 5
    dataset = Dataset([filename], seq_length)
    model = RNNTextGenerator(
        seq_length,
        dataset.vocab_size,
        learning_rate=learning_rate,
        epoch=epoch,
        batch_size=batch_size,
    )
    model.restore()
    print('Using restored model')
    print(model.generate(
        dataset,
        start_seq,
        50
    ))


class TestAlice(unittest.TestCase):
    def test(self):
        print('test_alice:')
        test_alice('./data/alice.txt', 'I love cats.')
        test_restore('./data/alice.txt', 'I love cats.')
