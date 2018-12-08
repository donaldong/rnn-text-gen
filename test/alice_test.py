"""An end to end test using 
 ALICE'S ADVENTURES IN WONDERLAND
"""

import unittest
from src.text_generator import RNNTextGenerator
from src.dataset import Dataset


class TestAlice(unittest.TestCase):
    def test_alice(self):
        print("----------------Testing dataset Alice---------------")
        seq_length = 25
        batch_size = 25
        learning_rate = 0.01
        epoch = 10
        dataset = Dataset(['./data/alice.txt'], seq_length)
        model = RNNTextGenerator(
            seq_length,
            dataset.vocab_size,
            learning_rate=learning_rate,
            epoch=epoch,
            batch_size=batch_size,
        )
        model.fit(dataset)
        model.save()
        start_seq = 'hello'
        model = RNNTextGenerator(
            len(start_seq),
            dataset.vocab_size,
        )
        model.restore()
        print('>>>>> {}'.format(start_seq), RNNTextGenerator.sample(
            model,
            dataset,
            start_seq,
            50
        ))
        print('<<<<<<')
