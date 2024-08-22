from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from ..utils.database import BookmarkDatabase

class EmbeddingService:
    def __init__(self, database: BookmarkDatabase):
        self.database = database
        self.embedding_model = None

    def generate_embeddings(self, embedding_model: str = "all-MiniLM-L6-v2"):
        current_model = self.database.get_metadata('embedding_model')
        if current_model == embedding_model:
            return {"message": "Embeddings are up to date"}

        bookmarks = self.database.get_bookmarks()
        if not bookmarks:
            return {"message": "No bookmarks found. Add some bookmarks first."}

        self.embedding_model = SentenceTransformer(embedding_model)
        docs = [f"{b['title']} {b['url']} {' '.join(b['tags'])}" for b in bookmarks]
        embeddings = self.embedding_model.encode(docs)

        for bookmark, embedding in zip(bookmarks, embeddings):
            self.database.update_bookmark_embedding(bookmark['id'], embedding)

        self.database.set_metadata('embedding_model', str(self.embedding_model.get_sentence_embedding_dimension()))
        return {"message": f"Generated embeddings for {len(bookmarks)} bookmarks using {embedding_model}"}

    def get_embeddings(self) -> np.ndarray:
        return self.database.get_embeddings()

    def generate_embedding_for_bookmark(self, bookmark: Dict[str, Any]) -> np.ndarray:
        if self.embedding_model is None:
            model_name = self.database.get_metadata('embedding_model') or "all-MiniLM-L6-v2"
            self.embedding_model = SentenceTransformer(model_name)

        doc = f"{bookmark['title']} {bookmark['url']} {' '.join(bookmark.get('tags', []))}"
        return self.embedding_model.encode([doc])[0]