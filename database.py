import pymongo
from bson import ObjectId

URL = "mongodb://filmsearcher:Qgf3KWPTqlLW76Te7ghm2nZfknyVqSQLQtviuGjgD1l8nYQqcVfh0H9lHN7SWhNxeNtmaZoHmPxDACDb5RwYIQ==@filmsearcher.mongo.cosmos.azure.com:10255/?ssl=true&retrywrites=false&replicaSet=globaldb&maxIdleTimeMS=120000&appName=@filmsearcher@"


class FilmsDB:
    def __init__(self, mongo_url=URL) -> None:
        self.client = pymongo.MongoClient(mongo_url)
        self.db = self.client["filmsClient"]
        self.collection = self.db["films"]

    def get_all_keywords(self):
        return self.collection.find({}, {'_id': 1, 'keywords': 1})

    def get_by_ids(self, indexes):
        return self.collection.find({"_id": {"$in": list(map(lambda x: ObjectId(x), indexes))}})

    def count_all(self):
        return self.collection.count_documents({})
