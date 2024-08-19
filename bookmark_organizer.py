import sqlite3
import numpy as np
from typing import List, Dict, Any
from bertopic import BERTopic
import pandas as pd
from umap import UMAP
from sentence_transformers import SentenceTransformer
from scipy.cluster import hierarchy as sch

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

    def update_bookmark_topic(self, bookmark_id: int, topic: int):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("UPDATE bookmarks SET topic = ? WHERE id = ?", (topic, bookmark_id))
            conn.commit()

    def update_bookmark_embedding(self, bookmark_id: int, embedding: np.ndarray):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("UPDATE bookmarks SET embedding = ? WHERE id = ?", (embedding.tobytes(), bookmark_id))
            conn.commit()

    def get_embeddings(self) -> np.ndarray:
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("SELECT embedding FROM bookmarks WHERE embedding IS NOT NULL")
            embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in c.fetchall()]
            return np.array(embeddings)

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
class BookmarkOrganizer:
    def __init__(self, database: BookmarkDatabase):
        self.database = database
        self.topic_model = None
        self.umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
        self.hierarchical_topics = None

    def generate_embeddings(self, embedding_model: str = "all-MiniLM-L6-v2"):
        current_model = self.database.get_metadata('embedding_model')
        if current_model == embedding_model:
            return {"message": "Embeddings are up to date"}

        bookmarks = self.database.get_bookmarks()
        if not bookmarks:
            return {"message": "No bookmarks found. Add some bookmarks first."}

        embedding_model = SentenceTransformer(embedding_model)
        docs = [f"{b['title']} {b['url']}" for b in bookmarks]
        embeddings = embedding_model.encode(docs)

        for bookmark, embedding in zip(bookmarks, embeddings):
            self.database.update_bookmark_embedding(bookmark['id'], embedding)

        self.database.set_metadata('embedding_model', embedding_model.get_sentence_embedding_dimension())
        return {"message": f"Generated embeddings for {len(bookmarks)} bookmarks using {embedding_model}"}

    def get_topic_representation(self, topic_id: int):
        if self.topic_model is None:
            raise ValueError("Topics have not been created. Call create_topics() first.")
        
        topic_words = self.topic_model.get_topic(topic_id)
        if not topic_words:
            return None
        
        return {
            'id': topic_id,
            'name': f"Topic {topic_id}",
            'count': len(topic_words),
            'representation': [{'word': word, 'score': score} for word, score in topic_words]
        }

    def get_scatter_plot_data(self):
        if self.topic_model is None:
            raise ValueError("Topics have not been created. Call create_topics() first.")
        
        embeddings = self.database.get_embeddings()
        if embeddings.size == 0:
            raise ValueError("No embeddings found. Generate embeddings first.")
        
        reduced_embeddings = self.umap_model.fit_transform(embeddings)
        
        bookmarks = self.database.get_bookmarks()
        scatter_data = []
        
        for i, bookmark in enumerate(bookmarks):
            scatter_data.append({
                'id': bookmark['id'],
                'x': float(reduced_embeddings[i, 0]),
                'y': float(reduced_embeddings[i, 1]),
                'topic': int(bookmark['topic'])
            })
        
        return scatter_data

    def create_topics(self):
        bookmarks = self.database.get_bookmarks()
        if not bookmarks:
            return {"message": "No bookmarks found. Add some bookmarks first."}

        # Check if embeddings exist and are up-to-date
        current_model = self.database.get_metadata('embedding_model')
        if not current_model or current_model != "all-MiniLM-L6-v2":
            self.generate_embeddings()

        embeddings = self.database.get_embeddings()
        if embeddings.size == 0:
            return {"message": "Failed to generate embeddings. Please try again."}

        docs = [f"{b['title']} {b['url']}" for b in bookmarks]

        try:
            self.topic_model = BERTopic(embedding_model=None)
            topics, _ = self.topic_model.fit_transform(docs, embeddings=embeddings)
            
            for bookmark, topic in zip(bookmarks, topics):
                self.database.update_bookmark_topic(bookmark['id'], int(topic))
            
            # Create hierarchical topics
            self.hierarchical_topics = self.create_hierarchical_topics(docs)
            
            return {"message": f"Created topics for {len(bookmarks)} bookmarks"}
        except Exception as e:
            return {"message": f"An error occurred while creating topics: {str(e)}"}

    def create_hierarchical_topics(self, docs):
        try:
            linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)
            hierarchical_topics = self.topic_model.hierarchical_topics(docs, linkage_function=linkage_function)
            return hierarchical_topics
        except Exception as e:
            raise ValueError(f"Error creating hierarchical topics: {str(e)}")

    def get_topic_tree(self):
        if self.topic_model is None or self.hierarchical_topics is None:
            raise ValueError("Topics have not been created. Call create_topics() first.")
        
        hier_topics = self.hierarchical_topics
        tree = self.topic_model.get_topic_tree(hier_topics)
        return tree
    
    def get_bookmarks(self) -> List[Dict[str, Any]]:
        return self.database.get_bookmarks()

    def get_tree_json(self):
        if self.hierarchical_topics is None:
            raise ValueError("Hierarchical topics have not been created. Call create_topics() first.")
        
        hier_topics = self.hierarchical_topics
        root = {
            "name": "Topics",
            "children": []
        }

        def add_children(node, parent_id):
            children = hier_topics[
                (hier_topics['Parent_ID'] == parent_id) &
                (hier_topics['Child_Left_ID'] != hier_topics['Child_Right_ID'])
            ]
            for _, row in children.iterrows():
                left_child = {
                    "name": row['Child_Left_Name'],
                    "children": []
                }
                right_child = {
                    "name": row['Child_Right_Name'],
                    "children": []
                }
                node["children"].extend([left_child, right_child])
                add_children(left_child, row['Child_Left_ID'])
                add_children(right_child, row['Child_Right_ID'])

        add_children(root, hier_topics['Parent_ID'].max())
        return root

    def get_sunburst_data(self):
        if self.hierarchical_topics is None:
            raise ValueError("Hierarchical topics have not been created. Call create_topics() first.")
        
        tree_json = self.get_tree_json()
        
        def add_values(node):
            if not node["children"]:
                node["value"] = 1
            else:
                for child in node["children"]:
                    add_values(child)
                node["value"] = sum(child["value"] for child in node["children"])
        
        add_values(tree_json)
        return tree_json

    def add_bookmark(self, bookmark_data: Dict[str, Any]) -> Dict[str, Any]:
        bookmark_id = self.database.add_bookmark(bookmark_data)
        
        if self.topic_model is not None:
            new_topic = self._assign_topic_to_bookmark(bookmark_data)
            self.database.update_bookmark_topic(bookmark_id, new_topic)
        
        return {"message": "Bookmark added successfully", "id": bookmark_id}

    def _assign_topic_to_bookmark(self, bookmark_data: Dict[str, Any]) -> int:
        doc = f"{bookmark_data['title']} {bookmark_data['url']}"
        embedding_model = SentenceTransformer(self.database.get_metadata('embedding_model'))
        embedding = embedding_model.encode([doc])[0]
        topic, _ = self.topic_model.transform([doc], embeddings=[embedding])
        return topic[0]

    def search_bookmarks(self, query: str) -> List[Dict[str, Any]]:
        bookmarks = self.database.get_bookmarks()
        return [b for b in bookmarks if query.lower() in b['title'].lower() or query.lower() in b['url'].lower()]

# Initialize components
database = BookmarkDatabase()
bookmark_organizer = BookmarkOrganizer(database)