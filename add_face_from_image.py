import sys
import cv2
import numpy as np

from mysql_db import add_person_if_not_exists, add_embedding


def init_insightface() -> "FaceAnalysis":
	from insightface.app import FaceAnalysis
	app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
	app.prepare(ctx_id=-1, det_size=(640, 640))
	return app


def main():
	if len(sys.argv) < 3:
		print("Usage: python add_face_from_image.py <image_path> <person_name>")
		sys.exit(1)

	image_path = sys.argv[1]
	person_name = " ".join(sys.argv[2:]).strip()
	if not person_name:
		raise RuntimeError("Person name is empty.")

	img_bgr = cv2.imread(image_path)
	if img_bgr is None:
		raise RuntimeError(f"Failed to read image: {image_path}")

	app = init_insightface()
	faces = app.get(img_bgr)
	if not faces:
		raise RuntimeError("No face detected in the image.")

	# Pick the largest face
	largest = max(faces, key=lambda f: (int(f.bbox[2]-f.bbox[0]) * int(f.bbox[3]-f.bbox[1])))
	if largest.embedding is None:
		raise RuntimeError("No embedding extracted from the face.")

	emb = largest.embedding.astype(np.float32)
	person_id = add_person_if_not_exists(person_name)
	add_embedding(person_id, emb)
	print(f"Added embedding for '{person_name}' from {image_path}")


if __name__ == "__main__":
	main()