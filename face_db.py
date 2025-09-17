import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import orjson


class FaceDB:
	def __init__(self, db_path: str = "face_db.json") -> None:
		self.db_path = db_path
		self.persons: List[Dict] = []
		self._load_if_exists()

	def _load_if_exists(self) -> None:
		if os.path.exists(self.db_path):
			self.load()

	def load(self) -> None:
		with open(self.db_path, "rb") as f:
			data = orjson.loads(f.read())
		self.persons = data.get("persons", [])

	def save(self) -> None:
		data = {"persons": self.persons}
		with open(self.db_path, "wb") as f:
			f.write(orjson.dumps(data))

	@staticmethod
	def _normalize(vec: np.ndarray) -> np.ndarray:
		norm = np.linalg.norm(vec) + 1e-12
		return vec / norm

	def add_embedding(self, name: str, embedding: np.ndarray) -> None:
		emb = self._normalize(embedding.astype(np.float32)).tolist()
		for person in self.persons:
			if person["name"].lower() == name.lower():
				person.setdefault("embeddings", []).append(emb)
				return
		self.persons.append({"name": name, "embeddings": [emb]})

	@staticmethod
	def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
		return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

	def _best_similarity_to_person(self, query: np.ndarray, person: Dict) -> float:
		if not person.get("embeddings"):
			return -1.0
		embs = np.array(person["embeddings"], dtype=np.float32)
		# Compare to centroid for stability
		centroid = self._normalize(embs.mean(axis=0))
		return self._cosine_similarity(query, centroid)

	def match(self, embedding: np.ndarray, threshold: float = 0.35) -> Tuple[Optional[str], float]:
		if not self.persons:
			return None, -1.0
		q = self._normalize(embedding.astype(np.float32))
		best_name: Optional[str] = None
		best_sim: float = -1.0
		for person in self.persons:
			sim = self._best_similarity_to_person(q, person)
			if sim > best_sim:
				best_sim = sim
				best_name = person["name"]
		if best_sim >= threshold:
			return best_name, best_sim
		return None, best_sim