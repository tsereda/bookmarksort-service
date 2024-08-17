# bookmark_organizer.py

import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from bertopic import BERTopic
import struct

class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

class DefaultEmbeddingModel(EmbeddingModel):
    def embed(self, text: str) -> np.ndarray:
        # Implement a default embedding method
        pass

class TopicModel(ABC):
    @abstractmethod
    def fit(self, docs: List[str]):
        pass

    @abstractmethod
    def transform(self, docs: List[str]) -> List[int]:
        pass

    @abstractmethod
    def get_hierarchical_topics(self, docs: List[str]) -> Dict[str, Any]:
        pass

class BERTopicModel(TopicModel):
    def __init__(self):
        self.model = BERTopic()

    def fit(self, docs: List[str]):
        self.model.fit(docs)

    def transform(self, docs: List[str]) -> List[int]:
        return self.model.transform(docs)[0]

    def get_hierarchical_topics(self, docs: List[str]) -> Dict[str, Any]:
        return self.model.hierarchical_topics(docs).to_dict(orient='records')

class BookmarkDatabase:
    def __init__(self, db_name: str = "bookmarks.db"):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS bookmarks
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          title TEXT NOT NULL,
                          url TEXT NOT NULL,
                          tags TEXT,
                          embedding BLOB,
                          topic INTEGER)''')
            conn.commit()

    def add_bookmark(self, bookmark: Dict[str, Any]) -> int:
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO bookmarks (title, url, tags, embedding, topic)
                VALUES (?, ?, ?, ?, ?)
            """, (
                bookmark['title'],
                bookmark['url'],
                ','.join(bookmark.get('tags', [])),
                np.array(bookmark.get('embedding', [])).tobytes(),
                int(bookmark.get('topic', -1))  # Ensure topic is stored as an integer
            ))
            return c.lastrowid

    def get_bookmarks(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("SELECT id, title, url, tags, topic FROM bookmarks")
            return [{"id": row[0], "title": row[1], "url": row[2],
                    "tags": row[3].split(',') if row[3] else [], 
                    "topic": int(row[4]) if row[4] is not None else None}
                    for row in c.fetchall()]

    def get_bookmark_texts(self) -> List[str]:
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("SELECT title FROM bookmarks")
            return [row[0] for row in c.fetchall()]
        
    def update_bookmark_topic(self, bookmark_id: int, topic: int):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("UPDATE bookmarks SET topic = ? WHERE id = ?", (topic, bookmark_id))
            conn.commit()

    def get_all_bookmark_ids_and_texts(self) -> List[Tuple[int, str]]:
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("SELECT id, title FROM bookmarks")
            return c.fetchall()

class BookmarkOrganizer:
    def __init__(self, 
                 embedding_model: Optional[EmbeddingModel] = None,
                 topic_model: Optional[TopicModel] = None,
                 database: Optional[BookmarkDatabase] = None):
        self.embedding_model = embedding_model or DefaultEmbeddingModel()
        self.topic_model = topic_model or BERTopicModel()
        self.database = database or BookmarkDatabase()

    def add_bookmark(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if 'embedding' not in data:
            data['embedding'] = self.embedding_model.embed(data['title'])
        
        bookmark_id = self.database.add_bookmark(data)
        return {**data, "id": bookmark_id}

    def list_bookmarks(self) -> List[Dict[str, Any]]:
        return self.database.get_bookmarks()

    def get_hierarchical_topics(self) -> Dict[str, Any]:
        docs = self.database.get_bookmark_texts()
        if not docs:
            return {"message": "No bookmarks found"}

        try:
            return self.topic_model.get_hierarchical_topics(docs)
        except Exception as e:
            return {"error": f"An error occurred while generating hierarchical topics: {str(e)}"}

    def update_topics(self):
        bookmarks = self.database.get_all_bookmark_ids_and_texts()
        if not bookmarks:
            return {"message": "No bookmarks found"}

        bookmark_ids, docs = zip(*bookmarks)
        
        # Fit the topic model
        self.topic_model.fit(docs)
        
        # Transform the documents to get their topics
        topics = self.topic_model.transform(docs)
        
        # Update topics in the database
        for bookmark_id, topic in zip(bookmark_ids, topics):
            self.database.update_bookmark_topic(bookmark_id, int(topic))
            print(f"{bookmark_id}: {topic}")

        return {"message": f"Updated topics for {len(bookmarks)} bookmarks"}