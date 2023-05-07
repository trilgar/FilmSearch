from models import EmbeddingSearch


class GUI:
    def __init__(self, embedding_search: EmbeddingSearch):
        super().__init__()
        self.embeddingSearch = embedding_search


