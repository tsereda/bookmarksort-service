import sqlite3
import numpy as np
from bertopic import BERTopic
from typing import List, Dict, Any

DB_NAME = "bookmarks.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS bookmarks
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      title TEXT NOT NULL,
                      url TEXT NOT NULL,
                      tags TEXT,
                      embedding BLOB,
                      topic INTEGER)''')
        conn.commit()

class BookmarkOrganizer:
    def __init__(self):
        self.topic_model = BERTopic()

    def list_bookmarks(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("SELECT id, title, url, tags, topic FROM bookmarks")
            return [{"id": row[0], "title": row[1], "url": row[2],
                     "tags": row[3].split(',') if row[3] else [], "topic": row[4]}
                    for row in c.fetchall()]

    def add_bookmark(self, data: Dict[str, Any]) -> Dict[str, Any]:
        title = data['title']
        url = data['url']
        tags = ','.join(data['tags']) if data.get('tags') else ''
        embedding = np.array(data['embedding']).tobytes() if data.get('embedding') else None
        topic = data.get('topic')
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO bookmarks (title, url, tags, embedding, topic) VALUES (?, ?, ?, ?, ?)",
                      (title, url, tags, embedding, topic))
            bookmark_id = c.lastrowid
        return {"id": bookmark_id, "title": title, "url": url, "tags": data.get('tags', []), "topic": topic}

    def get_hierarchical_topics(self) -> Dict[str, Any]:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("SELECT title, url FROM bookmarks")
            docs = [row[0] for row in c.fetchall()]

        if not docs:
            return {"message": "No bookmarks found"}

        try:
            # Check if the topic model has been fitted
            if not hasattr(self.topic_model, 'topics_') or self.topic_model.topics_ is None:
                # If not fitted, fit the model
                self.topic_model.fit(docs)

            # Generate hierarchical topics
            print(self.topic_model.get_topic_info())

            hierarchical_topics = self.topic_model.hierarchical_topics(docs)
            return hierarchical_topics.to_dict(orient='records')
        except Exception as e:
            return {"error": f"An error occurred while generating hierarchical topics: {str(e)}"}