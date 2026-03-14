# Trinetra Face Recognition System

Trinetra is a real-time face recognition and identity management system designed for intelligent surveillance and security monitoring.  
The system detects people in video streams, extracts facial features, performs fast vector similarity search, and manages identities through a scalable backend architecture.

---

# System Pipeline

## 1. Person Detection
Uses YOLOX v0.3 as the primary object detector to identify people in video frames.  
This step filters the scene so only human regions are processed further, improving efficiency.

## 2. Face Detection
SCRFD is used for fast and lightweight face detection.  
Once a person is detected, the model accurately locates faces inside the bounding box.

## 3. Face Embeddings
ArcFace generates 512-dimensional face embeddings that numerically represent unique facial features.  
These embeddings enable reliable comparison between different faces.

## 4. Vector Search & Matching
Milvus is used as the vector database for high-speed similarity search.  
It compares incoming embeddings with stored embeddings to identify known individuals.

## 5. Offline Clustering
Unknown faces are grouped using clustering algorithms:

Primary: DBSCAN  
Backup: HDBSCAN  

This allows the system to group repeated appearances of unknown individuals.

## 6. Human-in-the-Loop (HIL)
Security operators review clustered unknown faces.  
They assign identities and promote them to known individuals, allowing continuous system improvement.

## 7. Identity Management
A central metadata database maintains:

- Known identities  
- Temporary IDs for unknown faces  
- Event logs and audit trails  

PostgreSQL is recommended for storing this metadata.

---

# System Architecture Flow

Camera Feed  
↓  
Person Detection (YOLOX)  
↓  
Face Detection (SCRFD)  
↓  
Face Embedding (ArcFace)  
↓  
Vector Matching (Milvus)  
↓  
Identity Database (PostgreSQL)  
↓  
Human Review & Clustering  

---

# Tech Stack

Backend Framework: FastAPI  
Database: PostgreSQL  
Vector Database: Milvus  

Computer Vision Models:

Person Detection: YOLOX  
Face Detection: SCRFD  
Face Embedding: ArcFace  

Clustering Algorithms:

DBSCAN  
HDBSCAN  

---

# Project Structure

```
Trinetra-Face-Recognition-System
│
├── face_id_complete.py
├── requirements.txt
├── DB.txt
├── static/
│   └── index.html
├── models/
├── database/
└── README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/Dimpusathwick/Trinetra-Face-Recognition-System.git
cd Trinetra-Face-Recognition-System
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
python face_id_complete.py
```

---

# Features

Real-time face recognition  
High-performance vector similarity search  
Unknown face clustering  
Human-in-the-loop identity verification  
Scalable backend architecture  




