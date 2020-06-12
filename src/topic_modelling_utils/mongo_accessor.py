"""
class for accessing monogoDB
"""

from pymongo import MongoClient, errors
from credentials import username, password
from typing import Dict

DB_NAME = 'pnlp'
URL = f'mongodb+srv://{username}:{password}@cluster0-8ejtu.azure.mongodb.net/{DB_NAME}?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE'

class MongoAccessor:
    """
    Object for data base access
    """
    def __init__(self):
        self.collection_name = ""
        self.client = MongoClient(URL)
        self.database = self.client[DB_NAME]
        self.collection = None
        self.collection_items = None
        self.grid_id = None


    def access_collection(self, collection_name: str):
        """
        define the collection you want to access
        e.g. 'runs', 'parameter_configurations'
        :return:
        """
        # set collection name for wrapper object
        self.collection_name = collection_name
        # get colletion object
        self.collection = self.database[collection_name]
        # get the items stored in the chosen collection
        self.collection_items = self.database.\
            get_collection(collection_name).find({})
        if collection_name == 'parameter_configurations':
            self.grid_id = self.get_grid_id()


    def write(self, item):
        """
        writes single entry or list to db
        :return:
        """
        try:
            if isinstance(item, list):
                self.collection.insert_many(item)
            else:
                self.collection.insert_one(item)
        except errors.WriteError:
            print("Couldn't write to mongoDB!")


    def get_grid_id(self):
        """
        search for maximum grid_id in collection
        :return:
        """
        grid_ids = list()

        for item in self.collection_items:
            if 'grid_id' in item:
                grid_ids.append(item['grid_id'])

        if not grid_ids:
            return 0

        return max(grid_ids) + 1


    def remove_grid_ids(self, grid_id: int):
        """
        remove all entries with certain id
        :return:
        """
        self.collection.delete_many({"grid_id": grid_id})


    def get_best_config(self,
                        grid_id: int,
                        score: str="silhouette",
                        mode: str="tuning",
                        optimum: str="max") -> Dict:
        """
        iterates through all data points and
        :param grid_id:
        :param optimum:
        :param mode:
        :param score:
        :return:
        """
        i = 0
        best_config = dict()
        best_score = 0

        for item in self.collection_items:
            try:
                if item["mode"] == mode:
                    if item["grid_id"] == grid_id:
                        if i == 0:
                            best_score = item["score"][score]
                            best_config = item
                            i = 1
                        else:
                            if optimum == "max" and item["score"][score] > best_score:
                                best_score = item["score"][score]
                                best_config = item
                            elif optimum == "min" and item["score"][score] < best_score:
                                best_score = item["score"][score]
                                best_config = item
            except KeyError:
                "Mongo item not in right shape for parameter search!"

        return best_config

