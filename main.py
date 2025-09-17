from typing import Dict

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2

from mysql_db import add_person_if_not_exists, add_embedding, list_persons
from insightface_service import InsightFaceService


app = FastAPI(title="Face ID API", default_response_class=JSONResponse)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
	return {"status": "ok"}


@app.post("/reload")
async def reload_centroids() -> Dict[str, str]:
	InsightFaceService.get().reload_centroids()
	return {"status": "reloaded"}


@app.get("/persons")
async def persons():
	return {"persons": list_persons()}


@app.post("/add_photo")
async def add_photo(person_name: str = Form(...), file: UploadFile = File(...)):
	content = await file.read()
	np_bytes = np.frombuffer(content, dtype=np.uint8)
	img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
	if img is None:
		return JSONResponse(status_code=400, content={"error": "Invalid image"})

	service = InsightFaceService.get()
	faces = service.app.get(img)
	if not faces:
		return JSONResponse(status_code=400, content={"error": "No face detected"})

	largest = max(faces, key=lambda f: (int(f.bbox[2]-f.bbox[0]) * int(f.bbox[3]-f.bbox[1])))
	if largest.embedding is None:
		return JSONResponse(status_code=400, content={"error": "No embedding"})

	person_id = add_person_if_not_exists(person_name.strip())
	add_embedding(person_id, largest.embedding.astype(np.float32))
	service.reload_centroids()
	return {"status": "added", "person": person_name}


@app.post("/match")
async def match(file: UploadFile = File(...)):
	content = await file.read()
	np_bytes = np.frombuffer(content, dtype=np.uint8)
	img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
	if img is None:
		return JSONResponse(status_code=400, content={"error": "Invalid image"})

	service = InsightFaceService.get()
	faces = service.app.get(img)
	if not faces:
		return {"match": None, "similarity": -1.0}

	largest = max(faces, key=lambda f: (int(f.bbox[2]-f.bbox[0]) * int(f.bbox[3]-f.bbox[1])))
	if largest.embedding is None or not service.centroids:
		return {"match": None, "similarity": -1.0}

	q = largest.embedding.astype(np.float32)
	q = q / (np.linalg.norm(q) + 1e-12)
	best_name = None
	best_sim = -1.0
	for name, centroid in service.centroids.items():
		sim = service.cosine_similarity(q, centroid)
		if sim > best_sim:
			best_sim = sim
			best_name = name
	return {"match": best_name, "similarity": round(float(best_sim), 4)}