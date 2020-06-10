"""
class for accessing monogoDB
"""

from pymongo import MongoClient
from credentials import username, password

DB_NAME = 'pnlp'
URL = f'mongodb+srv://{username}:{password}@cluster0-8ejtu.azure.mongodb.net/{DB_NAME}?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE'

class MongoAccessor:
    """
    Object for data base access
    """
    def __init__(self, collection_name):
        self.client = MongoClient(URL)
        self.database = self.client[DB_NAME]
        self.collection = self.database[collection_name]
        self.collection_items = self.database.get_collection(collection_name).find({})
        self.grid_id = self.get_grid_id()

    def write(self, item):
        """
        writes single entry or list to db
        :return:
        """
        if isinstance(item, list):
            self.collection.insert_many(item)
        else:
            self.collection.insert_one(item)

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