"""
this external file provides a whole grid search algorithm
to optimize the topic modelling pipeline
"""
import numpy as np
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Dict
import random

class HyperparameterTuning:

    def __init__(self, data: np.array, pipeline_components: Dict[str, object], parameters: Dict[str,
                                                                                                Dict[str, List]]):
        """

        :param pipeline_components:
        :param parameters:
        """
        self.pipeline_components = pipeline_components
        self.parameters = parameters

        self.score = dict()

        self.tuning_data, self.test_data = self.split_data(data)


    @staticmethod
    def split_data(data: np.array, ratio: float = 0.8) -> Tuple:
        """
        perform a split in tuning and testing set
        :param data:
        :param ratio:
        :return:
        """
        n_data = len(data)
        random.seed(42)
        random.shuffle(data)
        split_length = int(n_data * ratio)
        tuning = data[:split_length]
        testing = data[split_length:]

        print(f"Performed split: {split_length} training,"
              f"{n_data - split_length} testing")

        return tuning, testing


    def build_grid(self):
        for component in self.pipeline_components:
            pipeline_fuction = self.pipeline_components[component]


