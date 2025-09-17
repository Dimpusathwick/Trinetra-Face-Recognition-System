import sys
import time
from typing import Tuple

import cv2
import numpy as np

# MySQL helpers
from mysql_db import add_person_if_not_exists, add_embedding, load_person_centroids


def init_insightface() -> "FaceAnalysis":
	from insightface.app import FaceAnalysis
	app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])  # CPU by default
	app.prepare(ctx_id=-1, det_size=(640, 640))
	return app


def draw_label(img: np.ndarray, text: str, left_top: Tuple[int, int]) -> None:
	x, y = left_top
	cv2.putText(img, text, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
	cv2.putText(img, text, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
	return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def main() -> None:
	print("Initializing models (first run may download weights)...")
	app = init_insightface()

	print("Loading centroids from MySQL...")
	centroids = load_person_centroids()   # {name: normalized_centroid}
	threshold = 0.35  # 0.33 more permissive, 0.40 stricter

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("ERROR: Cannot open webcam. Make sure a camera is available.")
		return

	print("Controls: q=quit, a=add face (save to MySQL), r=reload centroids")
	last_fps_time = time.time()
	frames = 0
	fps = 0.0

	while True:
		ok, frame = cap.read()
		if not ok:
			print("WARN: Failed to read frame")
			break

		# Detect + embed (InsightFace expects BGR frame)
		faces = app.get(frame)

		# FPS calculation
		frames += 1
		if frames >= 10:
			now = time.time()
			fps = frames / max(1e-6, (now - last_fps_time))
			last_fps_time = now
			frames = 0

		# Track largest face for optional add ('a')
		largest_area = -1
		largest_face = None
		for face in faces:
			bbox = face.bbox.astype(int)
			x1, y1, x2, y2 = bbox.tolist()
			w = max(0, x2 - x1)
			h = max(0, y2 - y1)
			area = w * h
			if area > largest_area:
				largest_area = area
				largest_face = face

		# Draw + identify each face using MySQL centroids
		for face in faces:
			bbox = face.bbox.astype(int)
			x1, y1, x2, y2 = bbox.tolist()
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

			label = "Unknown"
			best_name = None
			best_sim = -1.0

			if face.embedding is not None and len(centroids) > 0:
				q = face.embedding.astype(np.float32)
				q = q / (np.linalg.norm(q) + 1e-12)

				for person_name, centroid in centroids.items():
					sim = cosine_similarity(q, centroid)
					if sim > best_sim:
						best_sim = sim
						best_name = person_name

				if best_name is not None and best_sim >= threshold:
					label = f"{best_name} ({best_sim:.2f})"
				else:
					label = f"Unknown ({best_sim:.2f})"

			draw_label(frame, label, (x1, y1))

		cv2.putText(
			frame,
			f"FPS: {fps:.1f} | Known persons: {len(centroids)}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(0, 0, 0),
			3,
			cv2.LINE_AA,
		)
		cv2.putText(
			frame,
			f"FPS: {fps:.1f} | Known persons: {len(centroids)}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)

		cv2.imshow("Webcam Face ID (MySQL + InsightFace)", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord('q'):
			break
		elif key == ord('r'):
			centroids = load_person_centroids()
			print("Reloaded centroids from MySQL")
		elif key == ord('a'):
			# Add the largest visible face to MySQL (manual enrollment still available)
			if largest_face is None or largest_face.embedding is None:
				print("No face available to add. Ensure your face is visible and try again.")
				continue
			try:
				name_input = input("Enter name for this face: ").strip()
			except Exception:
				name_input = ""
			if not name_input:
				print("Name is empty. Skipping.")
				continue

			person_id = add_person_if_not_exists(name_input)
			add_embedding(person_id, largest_face.embedding.astype(np.float32))
			centroids = load_person_centroids()
			print(f"Added embedding for {name_input}. Persons: {len(centroids)}")

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()