import logging
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pymongo.cursor
import yaml
from sklearn.metrics import pairwise_distances

from database import FilmsDB

SearchResult = namedtuple("SearchResult", "result,embedding")

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", force=True)


class EmbeddingSearch:
    def __init__(self, vectors: np.ndarray, embedder: callable, indexes: np.ndarray = None):
        self._vectors = vectors
        self.embedder = embedder
        self.indexes = indexes

    @classmethod
    def from_texts(cls, inputs: list[str], embedder: callable):
        _vectors = cls._create_db(inputs, embedder)
        return cls(np.array(_vectors), embedder)

    @classmethod
    def from_database(cls, database: FilmsDB, embedder):
        _indexes, _vectors = cls._create_embeddings(database.get_all_keywords(), embedder, database.count_all())
        return cls(np.array(_vectors), embedder, np.array(_indexes))

    @staticmethod
    def _create_embeddings(inputs: pymongo.cursor.Cursor, embedder, total):
        logging.debug("creating keyword embeddings")
        result = []
        indexes = []
        step = total // 100
        for i, doc in enumerate(inputs):
            index = doc['_id']
            text = doc['keywords']
            vec = embedder(text)
            result.append(vec)
            indexes.append(index)
            if i % step == 0:
                logging.debug("%s/%s", i, total)
        return indexes, result

    @staticmethod
    def _create_db(inputs: list[str], embedder):
        logging.debug("creating db")
        result = []
        total = len(inputs)
        step = total // 100
        for i, text in enumerate(inputs):
            vec = embedder(text)
            result.append(vec)
            if i % step == 0:
                logging.debug("%s/%s", i, total)
        return result

    def from_pickle(self, path):
        pass

    def get_closest(self, query: str, n: int = 1000) -> list[dict]:
        query_vec = self.embedder(query)
        dist = pairwise_distances(query_vec[None, ...], self._vectors, "cosine")
        dist = dist.ravel()
        idx = np.argsort(dist)[:n]

        if self.indexes is not None:
            result = [{"id": self.indexes[_id], "distance": dist} for _id, dist in zip(idx, dist[idx])]
        else:
            result = [{"id": _id, "distance": dist} for _id, dist in zip(idx, dist[idx])]

        return SearchResult(result, query_vec)

    def get_rerank(self, labeling: list[dict]):
        pass
