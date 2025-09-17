import json
from typing import Dict, List
import mysql.connector
import numpy as np


def get_conn():
	return mysql.connector.connect(
		host="127.0.0.1",
		port=3306,
		user="root",
		password="",  # set your MySQL root password here if you have one
		database="faceid",
		auth_plugin="mysql_native_password",
	)


def add_person_if_not_exists(name: str) -> int:
	cx = get_conn()
	try:
		cur = cx.cursor()
		cur.execute("SELECT id FROM persons WHERE name=%s", (name,))
		row = cur.fetchone()
		if row:
			return int(row[0])
		cur.execute("INSERT INTO persons(name) VALUES(%s)", (name,))
		cx.commit()
		return cur.lastrowid
	finally:
		cx.close()


def add_embedding(person_id: int, embedding: np.ndarray) -> None:
	vec = embedding.astype(np.float32)
	vec = vec / (np.linalg.norm(vec) + 1e-12)
	cx = get_conn()
	try:
		cur = cx.cursor()
		cur.execute(
			"INSERT INTO embeddings(person_id, embedding_json) VALUES(%s, %s)",
			(person_id, json.dumps(vec.tolist())),
		)
		cx.commit()
	finally:
		cx.close()


def load_person_centroids() -> Dict[str, np.ndarray]:
	cx = get_conn()
	try:
		cur = cx.cursor()
		cur.execute(
			"""
			SELECT p.name, e.embedding_json
			FROM persons p
			JOIN embeddings e ON e.person_id = p.id
			"""
		)
		name_to_vecs: Dict[str, List[np.ndarray]] = {}
		for name, emb_json in cur.fetchall():
			vec = np.array(json.loads(emb_json), dtype=np.float32)
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


def list_persons() -> list[dict]:
	cx = get_conn()
	try:
		cur = cx.cursor()
		cur.execute("SELECT id, name, created_at FROM persons ORDER BY id ASC")
		rows = cur.fetchall()
		return [{"id": int(r[0]), "name": r[1], "created_at": str(r[2])} for r in rows]
	finally:
		cx.close()