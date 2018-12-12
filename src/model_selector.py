import numpy as np
import pandas as pd
from src.text_generator import RNNTextGenerator

class ModelSelector:
    def __init__(
            self,
            dataset,
            params,
            n_samples=5,
    ):
        """Initialize the text generator and contruct the tf graph
        Arguments
        ======================================================================
        dataset: Dataset
            The text dataset.

        params: dict
            The entire search space.

        n_samples: int
            The number of times to sample from the dataset for testing.
        """
        self.dataset = dataset
        self.params = params
        self.models = []
        self.acc = []
        self.loss = []
        self.n_samples = n_samples

    def search(self):
        """Perform a randomized search
        """
        selected = {}
        for name, options in self.params.items():
            selected[name] = np.random.choice(options)
        model = RNNTextGenerator(
            self.dataset.seq_length,
            self.dataset.vocab_size,
            **selected
        )
        model.fit(self.dataset)
        acc, loss = model.score(
            self.dataset,
            n_samples=self.n_samples
        )
        self.models.append(model)
        self.acc.append(acc)
        self.loss.append(loss)
        return model

    def as_df(self):
        """Save as a pandas Dataframe
        """
        return pd.DataFrame({
            'model': self.models,
            'accuracy': self.acc,
            'loss': self.loss,
        }).sort_values(
            by=['accuracy'],
            ascending=False,
        )

    def best_model(self):
        """Get the best model
        Returns
        ======================================================================
        model: RNNTextGenerator
            The model with highest accuracy.
        """
        return self.as_df().head(1)['model'].values[0]
