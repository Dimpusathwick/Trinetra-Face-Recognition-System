#!/usr/bin/env python3
"""
Face ID Complete - Working version with proper error handling
"""

import json
import uuid
import os
import threading
import time
from typing import Dict, List, Optional
import mysql.connector
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2
from PIL import Image
import io

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

def add_person_with_embedding(name: str, embedding: np.ndarray, quality: float = 0.0, metadata: Dict = None) -> str:
    """Add a person with their embedding - returns identity_id"""
    identity_id = add_identity(name, metadata)
    embedding_id = add_embedding(embedding, quality)
    link_identity_to_embedding(identity_id, embedding_id)
    return identity_id

def load_person_centroids() -> Dict[str, np.ndarray]:
    """Load centroids for all identities"""
    cx = get_conn()
    try:
        cur = cx.cursor()
        cur.execute("""
            SELECT i.name, e.vector, e.quality
            FROM identities i
            JOIN embeddings e ON e.id = i.embedding_id
            WHERE e.vector IS NOT NULL
        """)
        
        name_to_vecs: Dict[str, List[np.ndarray]] = {}
        for name, vector_json, quality in cur.fetchall():
            vec = np.array(json.loads(vector_json), dtype=np.float32)
            name_to_vecs.setdefault(name, []).append(vec)
        
        centroids: Dict[str, np.ndarray] = {}
        for name, vecs in name_to_vecs.items():
            stack = np.vstack(vecs)
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

# Mock InsightFace service for testing
class MockInsightFaceService:
    def __init__(self):
        self.app = None
        self.centroids = {}
        self._lock = threading.Lock()
        print("MockInsightFaceService initialized (no actual face detection)")
    
    def get_app(self):
        if self.app is None:
            # Mock app that doesn't actually do face detection
            class MockApp:
                def get(self, img):
                    # Return a mock face with random embedding
                    return [type('Face', (), {
                        'bbox': [100, 100, 200, 200],
                        'embedding': np.random.rand(512).astype(np.float32)
                    })()]
            self.app = MockApp()
        return self.app
    
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

# Global service instance
service = MockInsightFaceService()

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
        return {"status": "ok", "message": "Face ID API is running", "database": "connected"}
    except Exception as e:
        return {"status": "error", "message": "Face ID API is running", "database": f"disconnected: {e}"}

@app.post("/add_photo")
async def add_photo(person_name: str = Form(...), file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Mock face detection - just create a random embedding
        app = service.get_app()
        faces = app.get(image_array)
        
        if not faces:
            return {"success": False, "error": "No face detected in image"}
        
        face = faces[0]
        embedding = face.embedding
        
        # Add to database
        identity_id = add_person_with_embedding(
            person_name, 
            embedding, 
            quality=0.8,  # Mock quality
            metadata={"source": "upload", "file_name": file.filename}
        )
        
        # Reload centroids
        service.reload_centroids()
        
        return {
            "success": True, 
            "message": f"Added {person_name} to database",
            "identity_id": identity_id,
            "faces_detected": len(faces)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/match")
async def match_photo(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Mock face detection
        app = service.get_app()
        faces = app.get(image_array)
        
        if not faces:
            return {"match": None, "similarity": 0.0, "error": "No face detected"}
        
        face = faces[0]
        embedding = face.embedding
        
        # Compare with centroids
        centroids = service.get_centroids()
        if not centroids:
            return {"match": None, "similarity": 0.0, "message": "No persons in database"}
        
        best_match = None
        best_similarity = 0.0
        
        for name, centroid in centroids.items():
            # Calculate cosine similarity
            similarity = np.dot(embedding, centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        return {
            "match": best_match if best_similarity > 0.35 else None,
            "similarity": float(best_similarity),
            "faces_detected": len(faces)
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
    print("Starting Face ID API server...")
    print("Open http://127.0.0.1:8000 in your browser")
    print("Make sure XAMPP MySQL is running!")
    uvicorn.run(app, host="127.0.0.1", port=8000)