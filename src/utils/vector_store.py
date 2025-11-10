from typing import List, Dict, Any
import faiss
import numpy as np
import os
import json


class FaissStore:
    def __init__(self, dim: int, store_path: str):
        self.dim = dim
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        self.index_path = os.path.join(store_path, "index.faiss")
        self.meta_path = os.path.join(store_path, "metadata.json")
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            # inner product on normalized vectors approximates cosine similarity
            self.index = faiss.IndexFlatIP(dim)
            self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings: List[np.ndarray], metadatas: List[Dict[str, Any]]):
        if not embeddings:
            return
        arr = np.stack(embeddings).astype("float32")
        self.index.add(arr)
        self.metadata.extend(metadatas)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        q = query_embedding.astype("float32").reshape(1, -1)
        scores, ids = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append({"score": float(score), "metadata": self.metadata[idx]})
        return results
class VectorStore:
    def __init__(self):
        self.store = {}

    def add_vector(self, key, vector):
        self.store[key] = vector

    def query_vector(self, query_vector):
        # Implement a method to find the closest vector in the store to the query_vector
        pass