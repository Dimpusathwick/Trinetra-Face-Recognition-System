import threading
from typing import Dict
import numpy as np

from mysql_db import load_person_centroids


class InsightFaceService:
	_instance = None
	_lock = threading.Lock()

	def __init__(self) -> None:
		from insightface.app import FaceAnalysis
		self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
		self.app.prepare(ctx_id=-1, det_size=(640, 640))
		self._centroids: Dict[str, np.ndarray] = load_person_centroids()

	@classmethod
	def get(cls) -> "InsightFaceService":
		if cls._instance is None:
			with cls._lock:
				if cls._instance is None:
					cls._instance = InsightFaceService()
		return cls._instance

	def reload_centroids(self) -> None:
		self._centroids = load_person_centroids()

	@property
	def centroids(self) -> Dict[str, np.ndarray]:
		return self._centroids

	@staticmethod
	def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
		return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))