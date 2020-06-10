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
        self.tuning_data, self.test_data = self.split_data(data)

        # build the parameter grid
        # for the whole pipeline
        self.grid = dict()
        self.pipeline_grid = dict()
        self.build_grid()

        # store the results
        self.score_dict = dict()
        self.best_tuning_configs = dict()


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

        print(f"Performed split: {split_length} training, "
              f"{n_data - split_length} testing")

        return tuning, testing


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
                      n_node: int):
        """
        every node of the grid is now evaluated
        :param values:
        :param processed_data:
        :param n_node:
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

        return score, n_clusters, n_outliers


    def calculate_optimizer(self, params: List, config:List):
        """
        here a function is implemented
        calculation the general score
        which is then optimized
        :param params: parameters used for calculation
        :param config:
        :return:
        """
        #TODO: Come up with real calculation
        #      now were just minimizing outliers
        return self.score_dict[config]['outliers']


    def get_top_n(self,
                  n: int,
                  objective: List,
                  optimizer = "greatest") -> Dict:
        """
        find the top n configs based
        on their resulting score
        :param n:
        :param objective:
        :param optimizer:
        :return:
        """
        optimized_dict = dict()
        for config in self.score_dict:
            optimized_value = self.calculate_optimizer(objective, config)
            optimized_dict.update({config: optimized_value})

        # either max or minimize score
        if optimizer == "greatest":
            reverse = True
        else:
            reverse = False

        # select best configuration
        best_configs = dict(sorted(optimized_dict.items(), key=itemgetter(1), reverse=reverse)[:n])

        return best_configs


    def log_node(self, n, node, score, n_clusters, n_outliers):
        """
        log node configuration and results
        :param node: dict (pipeline steps and values)
        :return:
        """
        filename = 'grid_search_logs.csv'
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
                            grid_type: str="tuning"):
        """
        actual grid search
        also used for evaluation on
        test set
        :param grid_type: bool (tuning or test set)
        :return:
        """
        grid = dict()
        if grid_type == "tuning":
            processed_data = self.tuning_data
            grid = self.pipeline_grid

            print("\n##################")
            print("Run grid search...")
            print("##################")

        if grid_type == "testing":
            processed_data = self.test_data
            grid = [self.pipeline_grid[n_config]
                    for n_config in self.best_tuning_configs]

        for n, node in enumerate(grid):
            grid_search_data = processed_data
            print(f"\nRunning grid node {n+1} of {len(grid)}:")
            for pipeline_step in node:
                print(f"\nCurrent step: {self.algorithms[pipeline_step]}")
                print("Current configuration:")
                for params in node[pipeline_step]:
                    print(f"{params}: {node[pipeline_step][params]}")
                grid_search_data = self.pipeline_components[pipeline_step](grid_search_data,
                                                                            self.algorithms[pipeline_step],
                                                                            node[pipeline_step])
            score, n_clusters, n_outliers = self.evaluate_node(processed_data, grid_search_data, n)
            self.log_node(n, node, score, n_clusters, n_outliers)


    def run_on_test_set(self,
                        top_n: int=5,
                        optimize_for: List[str]=['outliers']) -> Dict:
        """
        evaluate the top n configs on the test set
        :param top_n:
        :param optimize_for:
        :return:
        """
        self.best_tuning_configs = self.get_top_n(top_n, optimize_for, optimizer="lowest")

        print(f"Top {top_n} configurations on tuning set_")
        for config in self.best_tuning_configs:
            print(f"#-Config: {config} -> Score: {self.best_tuning_configs[config]}")

        print("\n##################")
        print("Evaluate on test set...")
        print("##################")
        self.perform_grid_search(grid_type="testing")
        best_on_test = self.get_top_n(1, optimize_for, optimizer="lowest")

        print("\n##################")
        print("Best configuration on test set:")
        print("##################\n")
        for best_config in best_on_test:
            for component in self.pipeline_grid[best_config]:
                print(f"Component: {component}")
                for parameter in self.pipeline_grid[best_config][component]:
                    print(f"{parameter}: {self.pipeline_grid[best_config][component][parameter]}")
                print("\n")

            return self.pipeline_grid[best_config]
