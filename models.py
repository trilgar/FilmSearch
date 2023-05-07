import logging
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import pairwise_distances

SearchResult = namedtuple("SearchResult", "result,embedding")


class EmbeddingSearch:
    def __init__(self, vectors: np.ndarray, embedder: callable):
        self._vectors = vectors
        self.embedder = embedder

    @classmethod
    def from_texts(cls, inputs: list[str], embedder: callable):
        _vectors = cls._create_db(inputs, embedder)
        return cls(_vectors, embedder)

    @staticmethod
    def _create_db(inputs, embedder):
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
        result = [{"id": _id, "distance": dist} for _id, dist in zip(idx, dist[idx])]

        return SearchResult(result, query_vec)

    def get_rerank(self, labeling: list[dict]):
        pass
