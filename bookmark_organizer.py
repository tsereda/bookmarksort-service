# bookmark_organizer.py

import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from bertopic import BERTopic
import struct
from collections import defaultdict
import logging

class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

class DefaultEmbeddingModel(EmbeddingModel):
    def embed(self, text: str) -> np.ndarray:
        # Implement a default embedding method
        pass

# Topic Tree Structure

class Topic:
    def __init__(self, id: str, name: str):
        self.id: str = id
        self.name: str = name
        self.subtopics: Dict[str, 'Topic'] = {}
        self.bookmarks: List[Bookmark] = []

class Bookmark:
    def __init__(self, id: int, title: str, url: str, topic_id: str):
        self.id: int = id
        self.title: str = title
        self.url: str = url
        self.topic_id: str = topic_id

class TopicTree:
    def __init__(self):
        self.root: Dict[str, Topic] = {}

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

    def get_hierarchical_topics(self, docs: List[str]) -> List[Dict[str, Any]]:
        hierarchical_topics = self.model.hierarchical_topics(docs)
        return [
            {
                "Parent_ID": str(topic.get('Parent_ID', '')),
                "Parent_Name": str(topic.get('Parent_Name', '')),
                "Topics": [str(t) for t in topic.get('Topics', [])],
                "Child_Left_ID": str(topic.get('Child_Left_ID', '')),
                "Child_Left_Name": str(topic.get('Child_Left_Name', '')),
                "Child_Right_ID": str(topic.get('Child_Right_ID', '')),
                "Child_Right_Name": str(topic.get('Child_Right_Name', '')),
                "Distance": float(topic.get('Distance', 0.0))
            }
            for topic in hierarchical_topics.to_dict(orient='records')
        ]

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
        self.logger = logging.getLogger(__name__)

    def add_bookmark(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if 'embedding' not in data:
            data['embedding'] = self.embedding_model.embed(data['title'])
        
        bookmark_id = self.database.add_bookmark(data)
        return {**data, "id": bookmark_id}

    def list_bookmarks(self) -> List[Dict[str, Any]]:
        return self.database.get_bookmarks()

    def get_hierarchical_topics(self) -> List[Dict[str, Any]]:
        docs = self.database.get_bookmark_texts()
        if not docs:
            return []

        try:
            return self.topic_model.get_hierarchical_topics(docs)
        except Exception as e:
            self.logger.exception(f"An error occurred while generating hierarchical topics: {str(e)}")
            return []

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

        return {"message": f"Updated topics for {len(bookmarks)} bookmarks"}

import logging
from collections import defaultdict
from typing import List, Dict, Any

class BookmarkTopicTree:
    def __init__(self, bookmark_organizer):
        self.bookmark_organizer = bookmark_organizer
        self.logger = logging.getLogger(__name__)
        self.topic_tree = TopicTree()

    def build_tree(self) -> Dict[str, Any]:
        try:
            hierarchical_topics = self.bookmark_organizer.get_hierarchical_topics()
            self.logger.debug(f"Hierarchical topics: {hierarchical_topics}")
            bookmarks = self.bookmark_organizer.list_bookmarks()
            self.logger.debug(f"Number of bookmarks: {len(bookmarks)}")
            
            self._create_tree_structure(hierarchical_topics)
            self._add_bookmarks_to_tree(bookmarks)
            
            return self._convert_tree_to_dict(self.topic_tree.root)
        except Exception as e:
            self.logger.exception(f"Error building topic tree: {str(e)}")
            return {"error": f"An error occurred while building the topic tree: {str(e)}"}

    def _create_tree_structure(self, hierarchical_topics: List[Dict[str, Any]]):
        for topic in hierarchical_topics:
            parent_id = str(topic['Parent_ID'])
            parent_name = str(topic['Parent_Name'])
            child_topics = topic['Topics']
            
            if parent_id not in self.topic_tree.root:
                self.topic_tree.root[parent_id] = Topic(parent_id, parent_name)
            
            for child_id in child_topics:
                child_id = str(child_id)
                child_name = next((t['Child_Left_Name'] for t in hierarchical_topics if t['Child_Left_ID'] == child_id), '')
                if not child_name:
                    child_name = next((t['Child_Right_Name'] for t in hierarchical_topics if t['Child_Right_ID'] == child_id), '')
                
                if child_id not in self.topic_tree.root[parent_id].subtopics:
                    self.topic_tree.root[parent_id].subtopics[child_id] = Topic(child_id, child_name)

    def _add_bookmarks_to_tree(self, bookmarks: List[Dict[str, Any]]):
        for bookmark_data in bookmarks:
            bookmark = Bookmark(
                id=bookmark_data['id'],
                title=bookmark_data['title'],
                url=bookmark_data['url'],
                topic_id=str(bookmark_data['topic']) if bookmark_data['topic'] is not None else "Uncategorized"
            )

            
            topic = self._find_topic_by_id(self.topic_tree.root, bookmark.topic_id)
            if topic:
                topic.bookmarks.append(bookmark)
                print("Appending bookmark to topic: {}".format(topic))
            else:
                if "Uncategorized" not in self.topic_tree.root:
                    self.topic_tree.root["Uncategorized"] = Topic("Uncategorized", "Uncategorized")
                self.topic_tree.root["Uncategorized"].bookmarks.append(bookmark)

    def _find_topic_by_id(self, tree: Dict[str, Topic], topic_id: str) -> Optional[Topic]:
        if topic_id in tree:
            print("Found topic with id: ", topic_id)
            return tree[topic_id]
        
        for topic in tree.values():
            result = self._find_topic_by_id(topic.subtopics, topic_id)
            if result:
                return result
        print("No topic found with id: ", topic_id)
        return None

    def _convert_tree_to_dict(self, tree: Dict[str, Topic]) -> Dict[str, Any]:
        return {
            topic_id: {
                "id": topic.id,
                "name": topic.name,
                "subtopics": self._convert_tree_to_dict(topic.subtopics),
                "bookmarks": [self._simplify_bookmark(b) for b in topic.bookmarks],
                "bookmark_count": len(topic.bookmarks),
                "subtopic_count": len(topic.subtopics)
            }
            for topic_id, topic in tree.items()
        }

    def _simplify_bookmark(self, bookmark: Bookmark) -> Dict[str, Any]:
        return {
            "id": bookmark.id,
            "title": bookmark.title,
            "url": bookmark.url,
            "tags": []
        }

    def _generate_meta_info(self) -> Dict[str, int]:
        return {
            "total_topics": self._count_topics(self.topic_tree.root),
            "total_bookmarks": self._count_bookmarks(self.topic_tree.root),
            "max_depth": self._get_max_depth(self.topic_tree.root)
        }

    def _count_topics(self, tree: Dict[str, Topic]) -> int:
        return len(tree) + sum(self._count_topics(topic.subtopics) for topic in tree.values())

    def _count_bookmarks(self, tree: Dict[str, Topic]) -> int:
        return sum(len(topic.bookmarks) + self._count_bookmarks(topic.subtopics) for topic in tree.values())

    def _get_max_depth(self, tree: Dict[str, Topic]) -> int:
        if not tree:
            return 0
        return 1 + max((self._get_max_depth(topic.subtopics) for topic in tree.values()), default=0)
