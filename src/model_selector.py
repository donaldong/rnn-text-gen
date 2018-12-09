import numpy as np
import pandas as pd
from src.text_generator import RNNTextGenerator

class ModelSelector:
    def __init__(
            self,
            dataset,
            params,
    ):
        """Initialize the text generator and contruct the tf graph
        Arguments
        ======================================================================
        dataset: Dataset
            The text dataset.

        params: dict
            The entire search space.
        """
        self.dataset = dataset
        self.params = params
        self.models = []

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
        acc, loss = model.score(self.dataset)
        self.models.append(pd.DataFrame({
            'model': [model],
            'accuracy': [acc],
            'loss': [loss],
        }))
        return model

    def as_df(self):
        """Save as a pandas Dataframe
        """
        return pd.concat(
            self.models,
            ignore_index=True,
        ).sort_values(
            by=['accuracy'],
            ascending=False,
        )
