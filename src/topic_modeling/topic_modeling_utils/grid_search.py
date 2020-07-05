"""
this external file provides a whole grid search algorithm
to optimize the topic modelling pipeline
"""
import numpy as np
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Dict
import random
from itertools import product
from collections import Counter
from operator import itemgetter
import csv
import os.path
import datetime
from pymongo import errors

from topic_modeling.topic_modeling_utils.mongo_accessor import MongoAccessor

MONGO = MongoAccessor()

class HyperparameterTuning:
    """
    class object monitoring the
    hyperparameter tuning
    """

    def __init__(self, data: np.array,
                 pipeline: Dict[str, Dict]):
        """
        set the pipeline
        :param data:
        :param pipeline:
        """
        self.pipeline_components = dict()
        self.parameters = dict()
        self.algorithms = dict()
        for step in pipeline:
            self.pipeline_components.update({step: pipeline[step]['function']})
            self.parameters.update({step: pipeline[step]['parameters']})
            self.algorithms.update({step: pipeline[step]['name']})

        # perform data split for tuning and testing
        self.tuning_data = data

        # build the parameter grid
        # for the whole pipeline
        self.grid = dict()
        self.pipeline_grid = dict()
        self.build_grid()

        # store the results
        self.score_dict = dict()


    @staticmethod
    def dict_product(dictionary:Dict) -> List:
        configs = list()
        keys = dictionary.keys()
        for i, element in enumerate(product(*dictionary.values())):
             configs.append(dict(zip(keys, element)))
        return configs


    @staticmethod
    def silhouette_coef(values: List, clusters: List) -> float:
        '''
        From values and cluster assignments, computes a clustering evaluation as
        mean silhouette score.
        - param values: np array
        - param clusters: np array
        '''
        silhouette = silhouette_score(values, clusters, metric='cosine')
        print(f'Silhouette score: {round(silhouette, 3)}')
        # TODO: Integrate additional metric modifications.
        # IDEAS: silhouette, ssquare, outliers, n_clusters

        return silhouette


    def build_grid(self):
        """
        method tho set up the parameter grid
        by building flat grid for each pipeline step
        and than combining all
        :return:
        """
        for component in self.parameters:
            parameters = self.parameters[component]
            grid_list = self.dict_product(parameters)
            # flat grid
            self.grid.update({component: grid_list})
        # grid for the whole pipeline
        self.pipeline_grid = self.dict_product(self.grid)


    def evaluate_node(self,
                      values: List,
                      processed_data: List,
                      n_node: int,
                      memeory_dict: Dict):
        """
        every node of the grid is now evaluated
        :param values:
        :param processed_data:
        :param n_node:
        :param memeory_dict:
        :return:
        """
        # store a metric for evaluation
        score = self.silhouette_coef(values, processed_data)
        # setup counter for clusters
        cluster_counter = Counter()

        # fill counter
        for cluster in processed_data:
            cluster_counter.update([cluster])
        n_clusters = len(cluster_counter)

        # count outliers
        if -1 in cluster_counter:
            n_outliers = cluster_counter[-1]
        else:
            n_outliers = 0

        print(f"Number of clusters: {n_clusters}")
        print(f"Number of outliers: {n_outliers}")
        print("Representations per class:")
        for item in cluster_counter:
            print(f"{item}: {cluster_counter[item]}")

        # store relevant metrics
        self.score_dict.update({n_node: {'silhouette': score,
                                         'outliers': n_outliers}})
        # save score in mongo
        memeory_dict["grid_node"] = n_node
        memeory_dict["score"].update({'silhouette': float(score),
                                      'outliers': n_outliers})

        return score, n_clusters, n_outliers


    def log_node(self, n, node, score, n_clusters, n_outliers):
        """
        log node configuration and results
        :param node: dict (pipeline steps and values)
        :return:
        """
        filename = 'outputs\\grid_search_logs.csv'
        file_exists = os.path.isfile(filename)
        fields = ['node', 'score', 'n_clusters', 'n_outliers']
        parameters = [n+1, score, n_clusters, n_outliers]

        for pipeline_step in node:
            for params in node[pipeline_step]:
                field = self.algorithms[pipeline_step] + '__' + params
                param = node[pipeline_step][params]
                fields.append(field)
                parameters.append(param)

        with open(filename, 'a', newline='') as f:
            fieldnames = fields
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            node_dict = dict(zip(fields, parameters))
            writer.writerow(node_dict)


    def perform_grid_search(self,
                            embedding_type: str,
                            grid_type: str = "tuning",
                            data_base: object = None):
        """
        actual grid search
        :param embedding_type:
        :param data_base:
        :param grid_type: bool (tuning or test set)
        :return:
        """

        processed_data = self.tuning_data
        grid = self.pipeline_grid

        print("\n##################")
        print("Run grid search...")
        print("##################")

        for n, node in enumerate(grid):
            grid_search_data = processed_data

            # create initial dict
            # going to be forwarded to mongoDB
            mongo_dict = {
                "mode": grid_type,
                "embeddings": embedding_type,
                "parameters": {},
                "score": {},
                "grid_id": data_base.grid_id
            }

            print(f"\nRunning grid node {n+1} of {len(grid)}:")
            for pipeline_step in node:
                print(f"\nCurrent step: {self.algorithms[pipeline_step]}")

                mongo_dict["parameters"].update({self.algorithms[pipeline_step]:{}})

                print("Current configuration:")
                for params in node[pipeline_step]:
                    print(f"{params}: {node[pipeline_step][params]}")

                    mongo_dict["parameters"][self.algorithms[pipeline_step]].update({params:node[pipeline_step][params]})

                grid_search_data = self.pipeline_components[pipeline_step](grid_search_data,
                                                                            self.algorithms[pipeline_step],
                                                                            node[pipeline_step])

            score, n_clusters, n_outliers = self.evaluate_node(processed_data,
                                                               grid_search_data,
                                                               n,
                                                               mongo_dict)

            self.log_node(n, node, score, n_clusters, n_outliers)

            # add a timestamp for db
            mongo_dict.update({"timestamp": datetime.datetime.now()})

            # forward the information to mongoDB
            try:
                data_base.write(mongo_dict)
            except errors.InvalidDocument:
                print("Invalid mongo input:")
                print(mongo_dict)

        # Retrieve best configuration based on silhouette score.
        best_node = list(sorted(self.score_dict.items(),
                                key=lambda x: x[1]['silhouette'], reverse=True))[0][0]

        return grid[best_node]
