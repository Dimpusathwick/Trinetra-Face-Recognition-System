#!/usr/bin/env python3
"""
Face ID Complete - Working version with YOLOX and SCRFD detection
"""

import json
import uuid
import os
import threading
import time
from typing import Dict, List, Optional, Tuple
import mysql.connector
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2
from PIL import Image
import io
import torch
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort

# Database functions
def get_conn():
    try:
        return mysql.connector.connect(
            host="127.0.0.1",
            port=3306,
            user="root",
            password="",
            database="faceid",
            auth_plugin="mysql_native_password",
        )
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

def add_identity(name: str, metadata: Dict = None) -> str:
    """Add a new identity and return the UUID"""
    identity_id = str(uuid.uuid4())
    cx = get_conn()
    try:
        cur = cx.cursor()
        clean_metadata = {}
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, np.floating):
                    clean_metadata[key] = float(value)
                elif isinstance(value, np.integer):
                    clean_metadata[key] = int(value)
                else:
                    clean_metadata[key] = value
        
        cur.execute(
            "INSERT INTO identities(id, name, metadata) VALUES(%s, %s, %s)",
            (identity_id, name, json.dumps(clean_metadata))
        )
        cx.commit()
        return identity_id
    finally:
        cx.close()

def get_identity_by_name(name: str) -> Optional[Tuple[str, Optional[str]]]:
    """Return (identity_id, embedding_id) by exact name if exists, else None"""
    cx = get_conn()
    try:
        cur = cx.cursor()
        cur.execute("SELECT id, embedding_id FROM identities WHERE name = %s", (name,))
        row = cur.fetchone()
        if row:
            return row[0], row[1]
        return None
    finally:
        cx.close()

def add_embedding(vector: np.ndarray, quality: float = 0.0, source_id: str = None) -> str:
    """Add a new embedding and return the UUID"""
    embedding_id = str(uuid.uuid4())
    vec = vector.astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    
    cx = get_conn()
    try:
        cur = cx.cursor()
        clean_quality = float(quality) if isinstance(quality, np.floating) else quality
        clean_source_id = str(source_id) if source_id else None
        
        cur.execute(
            "INSERT INTO embeddings(id, vector, quality, source_id) VALUES(%s, %s, %s, %s)",
            (embedding_id, json.dumps(vec.tolist()), clean_quality, clean_source_id)
        )
        cx.commit()
        return embedding_id
    finally:
        cx.close()

def link_identity_to_embedding(identity_id: str, embedding_id: str) -> None:
    """Link an identity to its primary embedding"""
    cx = get_conn()
    try:
        cur = cx.cursor()
        cur.execute(
            "UPDATE identities SET embedding_id = %s WHERE id = %s",
            (embedding_id, identity_id)
        )
        cx.commit()
    finally:
        cx.close()

def add_person_with_embedding(name: str, embedding: np.ndarray, quality: float = 0.0, metadata: Dict = None) -> Tuple[str, bool]:
    """Add or append an embedding for a person name.
    Returns (identity_id, clustered_to_existing).
    clustered_to_existing=True when appended under an existing identity.
    """
    # If identity exists, append embedding; else create identity and set primary
    existing = get_identity_by_name(name)
    if existing is None:
        identity_id = add_identity(name, metadata)
        embedding_id = add_embedding(embedding, quality, source_id=identity_id)
        link_identity_to_embedding(identity_id, embedding_id)
        return identity_id, False
    else:
        identity_id, _primary_embedding = existing
        # Append new embedding linked to this identity
        _ = add_embedding(embedding, quality, source_id=identity_id)
        # Optionally update primary to the most recent high-quality one; keep current for now
        return identity_id, True

def load_person_centroids() -> Dict[str, np.ndarray]:
    """Load centroids for all identities by averaging all their embeddings."""
    cx = get_conn()
    try:
        cur = cx.cursor()
        # Fetch all identities
        cur.execute("SELECT id, name, embedding_id FROM identities")
        identities = cur.fetchall()  # list of (id, name, embedding_id)

        centroids: Dict[str, np.ndarray] = {}
        for identity_id, name, primary_embedding_id in identities:
            # Collect all embeddings linked to this identity: source_id = identity_id
            cur2 = cx.cursor()
            cur2.execute(
                "SELECT vector FROM embeddings WHERE source_id = %s",
                (identity_id,)
            )
            rows = cur2.fetchall()
            vectors: List[np.ndarray] = []
            for (vector_json,) in rows:
                if vector_json:
                    vec = np.array(json.loads(vector_json), dtype=np.float32)
                    vectors.append(vec)

            # Also include the primary embedding if set and not already included
            if primary_embedding_id:
                cur2.execute("SELECT vector FROM embeddings WHERE id = %s", (primary_embedding_id,))
                rowp = cur2.fetchone()
                if rowp and rowp[0]:
                    vecp = np.array(json.loads(rowp[0]), dtype=np.float32)
                    vectors.append(vecp)

            if not vectors:
                continue

            stack = np.vstack(vectors)
            c = stack.mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-12)
            centroids[name] = c.astype(np.float32)

        return centroids
    finally:
        cx.close()

def list_identities() -> List[Dict]:
    """List all identities with their metadata"""
    cx = get_conn()
    try:
        cur = cx.cursor()
        cur.execute("""
            SELECT i.id, i.name, i.embedding_id, i.metadata, i.created_at, i.updated_at,
                   e.quality, e.source_id
            FROM identities i
            LEFT JOIN embeddings e ON e.id = i.embedding_id
            ORDER BY i.created_at ASC
        """)
        
        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1], 
                "embedding_id": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "created_at": str(row[4]),
                "updated_at": str(row[5]),
                "embedding_quality": row[6],
                "source_id": row[7]
            }
            for row in rows
        ]
    finally:
        cx.close()

def delete_identity(identity_id: str) -> int:
    """Delete an identity and its embeddings"""
    cx = get_conn()
    try:
        cur = cx.cursor()
        cur.execute("DELETE FROM embeddings WHERE id IN (SELECT embedding_id FROM identities WHERE id = %s)", (identity_id,))
        cur.execute("DELETE FROM identities WHERE id = %s", (identity_id,))
        cx.commit()
        return cur.rowcount
    finally:
        cx.close()

def delete_identity_by_name(name: str) -> int:
    """Delete identity by name"""
    cx = get_conn()
    try:
        cur = cx.cursor()
        cur.execute("SELECT id FROM identities WHERE name = %s", (name,))
        row = cur.fetchone()
        if not row:
            return 0
        return delete_identity(row[0])
    finally:
        cx.close()

def update_identity_name_by_old_name(old_name: str, new_name: str) -> bool:
    """Update identity name by old name"""
    cx = get_conn()
    try:
        cur = cx.cursor()
        cur.execute("UPDATE identities SET name = %s WHERE name = %s", (new_name, old_name))
        cx.commit()
        return cur.rowcount > 0
    finally:
        cx.close()

# Legacy functions for backward compatibility
def list_persons() -> List[Dict]:
    """Legacy function - returns simplified person list"""
    identities = list_identities()
    return [
        {
            "id": i["id"],
            "name": i["name"],
            "created_at": i["created_at"]
        }
        for i in identities
    ]

def delete_person(name: str) -> int:
    """Legacy function - deletes by name"""
    return delete_identity_by_name(name)

# YOLOX and SCRFD Detection Service
class YOLOXSCRFDService:
    def __init__(self):
        self.yolo_model = None
        self.face_app = None
        self.centroids = {}
        self._lock = threading.Lock()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize YOLOX and SCRFD models"""
        try:
            print("Initializing YOLOX person detection model...")
            # Initialize YOLOX for person detection
            self.yolo_model = YOLO('yolov8n.pt')  # Using YOLOv8 as YOLOX alternative
            print("YOLOX model initialized successfully")
            
            print("Initializing SCRFD face detection model...")
            # Initialize InsightFace with SCRFD
            self.face_app = FaceAnalysis(
                name='buffalo_l',  # Uses SCRFD for face detection
                providers=['CPUExecutionProvider']  # Use CPU for compatibility
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("SCRFD face detection model initialized successfully")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            print("Falling back to OpenCV Haar Cascades...")
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize fallback models using OpenCV"""
        try:
            # Fallback person detection using HOG
            self.hog_detector = cv2.HOGDescriptor()
            self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Fallback face detection using Haar Cascades
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            print("Fallback models initialized successfully")
        except Exception as e:
            print(f"Error initializing fallback models: {e}")
    
    def detect_persons_yolox(self, image: np.ndarray) -> List[Dict]:
        """Detect persons using YOLOX/YOLOv8"""
        try:
            if self.yolo_model is None:
                return self._detect_persons_fallback(image)
            
            # Run YOLO inference
            results = self.yolo_model(image, verbose=False)
            
            person_boxes = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if it's a person (class 0 in COCO dataset)
                        if int(box.cls) == 0:  # Person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            
                            person_boxes.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class': 'person',
                                'method': 'YOLOX/YOLOv8'
                            })
            
            return person_boxes
            
        except Exception as e:
            print(f"YOLOX detection error: {e}")
            return self._detect_persons_fallback(image)
    
    def _detect_persons_fallback(self, image: np.ndarray) -> List[Dict]:
        """Fallback person detection using HOG"""
        try:
            if not hasattr(self, 'hog_detector'):
                return []
            
            # Convert to grayscale for HOG
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect persons with HOG
            persons, weights = self.hog_detector.detectMultiScale(
                gray, 
                winStride=(8, 8), 
                padding=(8, 8), 
                scale=1.05
            )
            
            person_boxes = []
            for i, (x, y, w, h) in enumerate(persons):
                confidence = 0.8  # Default confidence for HOG
                if len(weights) > i and len(weights[i]) > 0:
                    confidence = float(weights[i][0])
                
                person_boxes.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': confidence,
                    'class': 'person',
                    'method': 'HOG (Fallback)'
                })
            
            return person_boxes
            
        except Exception as e:
            print(f"HOG detection error: {e}")
            return []
    
    def detect_faces_scrfd(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using SCRFD"""
        try:
            if self.face_app is None:
                return self._detect_faces_fallback(image)
            
            # Run SCRFD face detection
            faces = self.face_app.get(image)
            
            face_boxes = []
            for face in faces:
                bbox = face.bbox.astype(int)
                face_boxes.append({
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    'confidence': float(face.det_score),
                    'embedding': face.embedding,
                    'landmarks': face.kps.tolist() if hasattr(face, 'kps') else [],
                    'method': 'SCRFD'
                })
            
            return face_boxes
            
        except Exception as e:
            print(f"SCRFD detection error: {e}")
            return self._detect_faces_fallback(image)
    
    def _detect_faces_fallback(self, image: np.ndarray) -> List[Dict]:
        """Fallback face detection using Haar Cascades"""
        try:
            if not hasattr(self, 'face_cascade'):
                return []
            
            # Convert to grayscale for face detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            face_boxes = []
            for (x, y, w, h) in faces:
                # Extract face region for embedding
                face_roi = image[y:y+h, x:x+w]
                embedding = self._extract_simple_embedding(face_roi)
                
                face_boxes.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': 0.9,  # Default confidence for Haar
                    'embedding': embedding,
                    'landmarks': [],
                    'method': 'Haar Cascade (Fallback)'
                })
            
            return face_boxes
            
        except Exception as e:
            print(f"Haar detection error: {e}")
            return []
    
    def _extract_simple_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract simple features from face image for embedding"""
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_image, (64, 64))
            
            # Convert to grayscale if needed
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized
            
            # Flatten and normalize
            features = face_gray.flatten().astype(np.float32)
            features = features / 255.0  # Normalize to [0, 1]
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(64 * 64, dtype=np.float32)
    
    def detect_and_embed(self, image_bytes: bytes) -> Tuple[List[Dict], List[np.ndarray]]:
        """Detect persons and faces, then extract embeddings"""
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image")
            
            # 1. Person Detection (YOLOX)
            person_boxes = self.detect_persons_yolox(img)
            
            detected_faces_info = []
            embeddings = []
            
            # 2. Face Detection (SCRFD) within person regions
            for person_box in person_boxes:
                px1, py1, px2, py2 = person_box['bbox']
                person_roi = img[py1:py2, px1:px2]
                
                if person_roi.size == 0:
                    continue
                
                # Detect faces within person ROI
                faces_in_roi = self.detect_faces_scrfd(person_roi)
                
                for face_box in faces_in_roi:
                    fx1, fy1, fx2, fy2 = face_box['bbox']
                    face_img = person_roi[fy1:fy2, fx1:fx2]
                    
                    if face_img.size == 0:
                        continue
                    
                    # Get embedding
                    embedding = face_box.get('embedding', self._extract_simple_embedding(face_img))
                    embeddings.append(embedding)
                    
                    # Adjust face bbox to original image coordinates
                    global_fx1 = px1 + fx1
                    global_fy1 = py1 + fy1
                    global_fx2 = px1 + fx2
                    global_fy2 = py1 + fy2
                    
                    detected_faces_info.append({
                        "bbox": [global_fx1, global_fy1, global_fx2, global_fy2],
                        "det_score": face_box.get('confidence', 0.9),
                        "landmarks": face_box.get('landmarks', []),
                        "person_bbox": [px1, py1, px2, py2],
                        "person_confidence": person_box.get('confidence', 0.8),
                        "detection_method": face_box.get('method', 'Unknown')
                    })
            
            return detected_faces_info, embeddings
            
        except Exception as e:
            print(f"Detection and embedding error: {e}")
            return [], []
    
    def reload_centroids(self):
        with self._lock:
            try:
                self.centroids = load_person_centroids()
                print(f"Loaded {len(self.centroids)} person centroids")
            except Exception as e:
                print(f"Error loading centroids: {e}")
                self.centroids = {}
    
    def get_centroids(self):
        with self._lock:
            return self.centroids.copy()
    
    def get_detection_methods(self) -> Dict[str, str]:
        """Get information about detection methods"""
        methods = {
            "person_detection": "YOLOX/YOLOv8",
            "face_detection": "SCRFD",
            "embedding_method": "InsightFace/OpenCV Feature Extraction"
        }
        
        # Check if fallback methods are being used
        if not hasattr(self, 'yolo_model') or self.yolo_model is None:
            methods["person_detection"] = "HOG (Fallback)"
        
        if not hasattr(self, 'face_app') or self.face_app is None:
            methods["face_detection"] = "Haar Cascade (Fallback)"
        
        return methods

# Global service instance
service = YOLOXSCRFDService()

# FastAPI app
app = FastAPI(title="Face ID API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    try:
        # Test database connection
        conn = get_conn()
        conn.close()
        
        # Get detection methods info
        detection_methods = service.get_detection_methods()
        
        return {
            "status": "ok", 
            "message": "Face ID API is running", 
            "database": "connected",
            "detection_methods": detection_methods,
            "centroids_loaded": len(service.get_centroids())
        }
    except Exception as e:
        return {"status": "error", "message": "Face ID API is running", "database": f"disconnected: {e}"}

@app.post("/add_photo")
async def add_photo(person_name: str = Form(...), file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        
        # Use YOLOX and SCRFD for detection
        detected_faces_info, embeddings = service.detect_and_embed(contents)
        
        if not detected_faces_info or not embeddings:
            return {"success": False, "error": "No face detected in image"}
        
        # Use the first detected face
        face_info = detected_faces_info[0]
        embedding = embeddings[0]
        
        # Add to database
        identity_id, clustered = add_person_with_embedding(
            person_name, 
            embedding, 
            quality=float(face_info.get('det_score', 0.8)),
            metadata={
                "source": "upload", 
                "file_name": file.filename,
                "detection_method": face_info.get('detection_method', 'Unknown'),
                "person_confidence": face_info.get('person_confidence', 0.8),
                "face_bbox": face_info.get('bbox', []),
                "person_bbox": face_info.get('person_bbox', [])
            }
        )
        
        # Reload centroids
        service.reload_centroids()
        
        return {
            "success": True, 
            "message": f"Added {person_name} to database",
            "identity_id": identity_id,
            "clustered": clustered,
            "faces_detected": len(detected_faces_info),
            "detection_method": face_info.get('detection_method', 'Unknown'),
            "face_confidence": face_info.get('det_score', 0.8),
            "person_confidence": face_info.get('person_confidence', 0.8)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/match")
async def match_photo(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        
        # Use YOLOX and SCRFD for detection
        detected_faces_info, embeddings = service.detect_and_embed(contents)
        
        if not detected_faces_info or not embeddings:
            return {"match": None, "similarity": 0.0, "error": "No face detected"}
        
        # Use the first detected face
        face_info = detected_faces_info[0]
        embedding = embeddings[0]
        
        # Compare with centroids
        centroids = service.get_centroids()
        if not centroids:
            return {"match": "Unknown", "similarity": 0.0, "message": "No persons in database"}
        
        best_match = None
        best_similarity = 0.0
        
        # Normalize the input embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
        
        print(f"DEBUG: Comparing with {len(centroids)} centroids")
        for name, centroid in centroids.items():
            # Calculate cosine similarity
            similarity = np.dot(embedding_norm, centroid)
            print(f"DEBUG: Similarity with {name}: {similarity:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        print(f"DEBUG: Best match: {best_match}, Best similarity: {best_similarity:.4f}")
        print(f"DEBUG: Threshold: 0.5, Will match: {best_similarity > 0.5}")
        
        return {
            "match": best_match if best_similarity > 0.5 else "Unknown",
            "similarity": float(best_similarity),
            "faces_detected": len(detected_faces_info),
            "detection_method": face_info.get('detection_method', 'Unknown'),
            "face_confidence": face_info.get('det_score', 0.8),
            "person_confidence": face_info.get('person_confidence', 0.8)
        }
        
    except Exception as e:
        return {"match": None, "similarity": 0.0, "error": str(e)}

@app.get("/persons")
async def get_persons():
    try:
        persons = list_persons()
        return {"persons": persons, "count": len(persons)}
    except Exception as e:
        return {"persons": [], "count": 0, "error": str(e)}

@app.post("/reload")
async def reload_centroids():
    try:
        service.reload_centroids()
        centroids = service.get_centroids()
        return {"message": "Centroids reloaded", "count": len(centroids)}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/person")
async def delete_person_endpoint(name: str = Query(...)):
    try:
        deleted = delete_person(name)
        if deleted > 0:
            service.reload_centroids()
            return {"deleted": deleted, "message": f"Deleted {name}"}
        else:
            return {"deleted": 0, "error": f"Person '{name}' not found"}
    except Exception as e:
        return {"deleted": 0, "error": str(e)}

@app.put("/person")
async def update_person_name(old_name: str = Query(...), new_name: str = Query(...)):
    try:
        # Check if old name exists
        identities = list_identities()
        old_exists = any(i["name"] == old_name for i in identities)
        
        if not old_exists:
            return {"updated": False, "error": f"Person '{old_name}' not found"}
        
        # Check if new name already exists
        new_exists = any(i["name"] == new_name for i in identities)
        if new_exists:
            return {"updated": False, "error": f"Person '{new_name}' already exists"}
        
        # Update the name
        updated = update_identity_name_by_old_name(old_name, new_name)
        
        if updated:
            service.reload_centroids()
            return {"updated": True, "message": f"Updated '{old_name}' to '{new_name}'"}
        else:
            return {"updated": False, "error": "Update failed"}
            
    except Exception as e:
        return {"updated": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting Face ID API server with YOLOX and SCRFD...")
    print("Open http://127.0.0.1:8000 in your browser")
    print("Make sure XAMPP MySQL is running!")
    
    # Initialize centroids on startup
    try:
        service.reload_centroids()
        print(f"Loaded {len(service.get_centroids())} person centroids")
    except Exception as e:
        print(f"Warning: Could not load centroids: {e}")
    
    # Print detection methods
    methods = service.get_detection_methods()
    print(f"Detection methods: {methods}")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)