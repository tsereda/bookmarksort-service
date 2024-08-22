import sqlite3
import numpy as np
from typing import List, Dict, Any

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
            c.execute('''CREATE TABLE IF NOT EXISTS metadata
                         (key TEXT PRIMARY KEY,
                          value TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS topics
                         (id INTEGER PRIMARY KEY,
                          name TEXT)''')
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
                bookmark.get('embedding', None),
                int(bookmark.get('topic', -1))
            ))
            return c.lastrowid

    def get_bookmarks(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("SELECT id, title, url, tags, topic FROM bookmarks")
            bookmarks = []
            for row in c.fetchall():
                bookmark = {
                    'id': row[0],
                    'title': row[1],
                    'url': row[2],
                    'tags': row[3].split(',') if row[3] else [],
                    'topic': row[4]
                }
                bookmarks.append(bookmark)
            return bookmarks

    def get_embeddings(self) -> np.ndarray:
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("SELECT embedding FROM bookmarks WHERE embedding IS NOT NULL")
            embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in c.fetchall()]
            return np.array(embeddings)

    def update_bookmark_embedding(self, bookmark_id: int, embedding: np.ndarray):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("UPDATE bookmarks SET embedding = ? WHERE id = ?", (embedding.tobytes(), bookmark_id))
            conn.commit()

    def set_metadata(self, key: str, value: str):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
            conn.commit()

    def get_metadata(self, key: str) -> str:
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            result = c.fetchone()
            return result[0] if result else None
        
    def update_bookmark_topic(self, bookmark_id: int, topic: int):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("UPDATE bookmarks SET topic = ? WHERE id = ?", (topic, bookmark_id))
            conn.commit()

    def get_untagged_bookmarks(self, limit: int = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            query = """
                SELECT id, title, url FROM bookmarks
                WHERE tags IS NULL OR tags = ''
            """
            if limit:
                query += f" LIMIT {limit}"
            c.execute(query)
            return [{'id': row[0], 'title': row[1], 'url': row[2]} for row in c.fetchall()]

    def update_bookmark_tags(self, bookmark_id: int, tags: List[str]):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("""
                UPDATE bookmarks
                SET tags = ?
                WHERE id = ?
            """, (','.join(tags), bookmark_id))
            conn.commit()

    def update_topic_name(self, topic_id: int, name: str):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO topics (id, name)
                VALUES (?, ?)
            """, (topic_id, name))
            conn.commit()

    def get_topic_names(self) -> Dict[int, str]:
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("SELECT id, name FROM topics")
            return {row[0]: row[1] for row in c.fetchall()}