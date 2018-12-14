"""Authors: Donald Dong
"""
import numpy as np
import pandas as pd
from src.text_generator import RNNTextGenerator


class ModelSelector:
    """Performs randomized search and rank the models by accuracy. It selects
    the best ranking models and allows lengthy searching (for hours/days).
    """
    def __init__(
            self,
            dataset,
            params,
            n_samples=5,
    ):
        """Creates a selector

        Arguments
        ======================================================================
        dataset: Dataset
            A `Dataset` object to train the model.

        params: dict
            A dictionary which describes the search space.  The each key of the
            dictionary stores a list of parameters. The selector randomly
            choice a parameter value from the list, for each parameter key.

        n_samples: int
            The number of times to sample from the dataset for testing. The
            selector uses the average accuracy to rank the models.
        """
        self._dataset = dataset
        self._params = params
        self._models = []
        self._acc = []
        self._loss = []
        self._n_samples = n_samples

    def search(self):
        """Search the parameter space. It generates a combination of
        parameters, fit, and score the text generator. The selector keeps track
        of the model and its average accuracy and score on the test data.

        Returns
        ======================================================================
        model: RNNTextGenerator
            A fitted `RNNTextGenerator`.
        """
        selected = {}
        for name, options in self._params.items():
            selected[name] = np.random.choice(options)
        model = RNNTextGenerator(
            self._dataset.seq_length,
            self._dataset.vocab_size,
            **selected
        )
        model.fit(self._dataset)
        acc, loss = model.score(
            self._dataset,
            n_samples=self._n_samples
        )
        self._models.append(model)
        self._acc.append(acc)
        self._loss.append(loss)
        return model

    def as_df(self):
        """Save the searching result (models and their scores) as a pandas data
        frame.
        ```
                                                   model  accuracy        loss
    0  {'vocab_size': 70, 'rnn_cell': <class 'tensorf...  0.094519   88.173103
    1  {'vocab_size': 70, 'rnn_cell': <class 'tensorf...  0.068282  104.829025
    2  {'vocab_size': 70, 'rnn_cell': <class 'tensorf...  0.052424   12.201582
        ```

        Returns
        ======================================================================
        df: pd.DataFrame
            A `pd.DataFrame` sorted by `accuracy` in non-increasing order.
        """
        return pd.DataFrame({
            'model': self._models,
            'accuracy': self._acc,
            'loss': self._loss,
        }).sort_values(
            by=['accuracy'],
            ascending=False,
        )

    def best_models(self, n):
        """Get the model with the highest accuracy. It wraps the `best_models`
        method.

        Arguments
        ======================================================================
        n: int
            The numer of best models.

        Returns
        ======================================================================
        models: RNNTextGenerator[]
            A list of `RNNTextGenerator` with length `n`.
        """
        return self.as_df().head(n)['model'].values[0]

    def best_model(self):
        """Get the model with the highest accuracy. It wraps the `best_models`
        method.

        Returns
        ======================================================================
        model: RNNTextGenerator
            An `RNNTextGenerator` with the highest accuracy among the models
            the selector has seen so far.
        """
        return self.best_models(1)
