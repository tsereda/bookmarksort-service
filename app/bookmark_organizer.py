from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Union
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Bookmark, db
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class BookmarkOrganizer:
    def __init__(self, db_url='sqlite:///bookmarks.db'):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.topic_model = None
        self.is_ready = False
        self.is_initializing = False
        self.is_fitted = False
        self.hierarchical_topics = None

        self.logger = logging.getLogger(__name__)

    def initialize(self, embedding_model="all-MiniLM-L6-v2", nr_topics="auto", top_n_words=10):
        if self.is_initializing:
            return
        self.is_initializing = True
        logger.info("Initializing BERTopic model...")
        try:
            representation_model = KeyBERTInspired()
            self.topic_model = BERTopic(
                embedding_model=embedding_model,
                nr_topics=nr_topics,
                top_n_words=top_n_words,
                representation_model=representation_model
            )
            self.is_ready = True
            logger.info("BERTopic model initialization complete.")
        except Exception as e:
            logger.error(f"Failed to initialize BERTopic model: {str(e)}")
            raise
        finally:
            self.is_initializing = False

    def fit_model(self):
        if not self.is_ready:
            raise RuntimeError("BERTopic model is not initialized")
        
        session = self.Session()
        try:
            bookmarks = session.query(Bookmark).all()
            if not bookmarks:
                self.logger.warning("No bookmarks available to fit the model.")
                return

            texts = [f"{b.title} {b.url}" for b in bookmarks]
            
            # Fit the model
            topics, _ = self.topic_model.fit_transform(texts)
            self.is_fitted = True
            self.logger.info("BERTopic model fitted successfully.")
            
            # Generate hierarchical topics
            hierarchical_topics = self.topic_model.hierarchical_topics(texts)
            self.hierarchical_topics = hierarchical_topics

            self.logger.info(f"Hierarchical topics: {hierarchical_topics}")
            
            # Generate topic names directly from hierarchical topics
            self.topic_names = {topic[0]: f"Topic_{topic[0]}" for topic in self.hierarchical_topics if len(topic) >= 2}
            
            # Update bookmarks with topic names
            for bookmark, topic in zip(bookmarks, topics):
                topic_id = int(topic)
                bookmark.topic = self.topic_names.get(topic_id, f"Topic_{topic_id}")
                session.add(bookmark)
            
            session.commit()
            self.logger.info("Hierarchical topics generated successfully.")
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to fit BERTopic model: {str(e)}")
            raise
        finally:
            session.close()

    def _generate_topic_names_from_hierarchy(self):
        topic_names = {}
        for topic in self.hierarchical_topics:
            if len(topic) < 2:
                continue
            topic_id, _, *_ = topic
            topic_names[topic_id] = f"Topic_{topic_id}"
        return topic_names


    def add_bookmark(self, bookmark: Dict) -> Dict:
        if not self.is_ready or not self.is_fitted:
            raise RuntimeError("BERTopic model is not initialized or fitted")

        session = self.Session()
        try:
            text = f"{bookmark['title']} {bookmark['url']}"
            topics, _ = self.topic_model.transform([text])
            topic = topics[0]
            topic_name = f"Topic_{topic}"
            
            new_bookmark = Bookmark(url=bookmark["url"], title=bookmark["title"], topic=topic_name)
            session.add(new_bookmark)
            session.commit()

            return {topic_name: [{"id": new_bookmark.id, "url": bookmark["url"], "title": bookmark["title"]}]}
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding bookmark: {str(e)}")
            raise
        finally:
            session.close()

    def list_bookmarks(self, topic: str = None, page: int = 1, per_page: int = 20) -> Dict:
        session = self.Session()
        try:
            query = session.query(Bookmark)
            if topic:
                query = query.filter(Bookmark.topic == topic)
            
            total = query.count()
            bookmarks = query.offset((page - 1) * per_page).limit(per_page).all()
            
            return {
                "bookmarks": [
                    {
                        "id": bookmark.id,
                        "url": bookmark.url,
                        "title": bookmark.title,
                        "topic": bookmark.topic
                    }
                    for bookmark in bookmarks
                ],
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page
            }
        finally:
            session.close()

    def search_bookmarks(self, query: str) -> List[Dict]:
        if not self.is_ready or not self.is_fitted:
            raise RuntimeError("BERTopic model is not initialized or fitted")

        session = self.Session()
        try:
            all_bookmarks = session.query(Bookmark).all()
            texts = [f"{b.title} {b.url}" for b in all_bookmarks]
            similarities = self.topic_model.embedding_model_.encode([query], texts)
            
            results = [(bookmark, similarity) for bookmark, similarity in zip(all_bookmarks, similarities[0])]
            results.sort(key=lambda x: x[1], reverse=True)
            
            return [
                {
                    "id": bookmark.id,
                    "url": bookmark.url,
                    "title": bookmark.title,
                    "topic": bookmark.topic,
                    "similarity": float(similarity)
                }
                for bookmark, similarity in results[:10]  # Return top 10 results
            ]
        finally:
            session.close()

    def process_bookmarks(self) -> Dict:
        if not self.is_ready or not self.is_fitted:
            raise RuntimeError("BERTopic model is not initialized or fitted")

        session = self.Session()
        try:
            bookmarks = session.query(Bookmark).all()
            if not bookmarks:
                logger.warning("No bookmarks found to process.")
                return {}

            texts = [f"{b.title} {b.url}" for b in bookmarks]
            topics, _ = self.topic_model.transform(texts)

            organized_bookmarks = {}
            for bookmark, topic in zip(bookmarks, topics):
                topic_name = f"Topic_{topic}"
                if topic_name not in organized_bookmarks:
                    organized_bookmarks[topic_name] = []
                
                organized_bookmarks[topic_name].append({
                    "id": bookmark.id,
                    "url": bookmark.url,
                    "title": bookmark.title,
                })

                bookmark.topic = topic_name
                session.add(bookmark)

            session.commit()
            logger.info(f"Processed {len(bookmarks)} bookmarks into {len(organized_bookmarks)} topics.")
            return organized_bookmarks
        except Exception as e:
            session.rollback()
            logger.error(f"Error processing bookmarks: {str(e)}")
            raise
        finally:
            session.close()

    def get_topics(self) -> List[Dict]:
        if not self.is_ready or not self.is_fitted:
            raise RuntimeError("BERTopic model is not initialized or fitted")
        
        topic_info = self.topic_model.get_topic_info()
        return [
            {
                "topic": int(row['Topic']),
                "count": int(row['Count']),
                "name": row['Name'],
                "representation": [
                    {"word": word, "score": score}
                    for word, score in self.topic_model.get_topic(row['Topic'])
                ]
            }
            for _, row in topic_info.iterrows()
        ]

    def get_hierarchical_topics(self) -> List[Dict]:
        if not self.is_ready or not self.is_fitted or self.hierarchical_topics is None:
            raise RuntimeError("Hierarchical topics are not available")
        
        topic_hierarchy = {}
        
        for topic in self.hierarchical_topics:
            topic_id = str(topic[0])
            parent_id = str(topic[1]) if pd.notnull(topic[1]) else None
            
            # Handle the distance value more carefully
            distance = None
            if len(topic) > 2 and pd.notnull(topic[2]):
                try:
                    distance = float(topic[2])
                except ValueError:
                    # If conversion to float fails, log a warning and set distance to None
                    logging.warning(f"Could not convert distance value '{topic[2]}' to float for topic {topic_id}")
            
            topic_info = {
                "id": topic_id,
                "name": f"Topic_{topic_id}",
                "parent": parent_id,
                "distance": distance,
                "children": []
            }
            
            topic_hierarchy[topic_id] = topic_info
        
        # Build the hierarchical structure
        root_topics = []
        for topic_id, topic_info in topic_hierarchy.items():
            if topic_info['parent'] is None:
                root_topics.append(topic_info)
            else:
                parent = topic_hierarchy.get(topic_info['parent'])
                if parent:
                    parent['children'].append(topic_info)
                else:
                    # If parent is not found, treat as root topic
                    root_topics.append(topic_info)
        
        return root_topics

    def get_visualization_data(self) -> Dict:
        if not self.is_ready or not self.is_fitted:
            raise RuntimeError("BERTopic model is not initialized or fitted")
        
        session = self.Session()
        try:
            bookmarks = session.query(Bookmark).all()
            
            # Build hierarchical topic structure
            topic_hierarchy = self._build_hierarchical_topics()
            
            # Prepare bookmark data
            bookmarks_data = [
                {
                    'id': bookmark.id,
                    'topic_id': bookmark.topic.split('_')[1],
                    'topic_name': bookmark.topic,
                    'topic_probability': 1.0,  # We don't have this information readily available
                    'url': bookmark.url,
                    'title': bookmark.title
                }
                for bookmark in bookmarks
            ]
            
            # Prepare metadata
            metadata = {
                "total_bookmarks": len(bookmarks),
                "total_topics": len(self.topic_names),
                "max_hierarchy_depth": self._get_max_depth(topic_hierarchy),
                "visualization_type": "tree"
            }
            
            return {
                "metadata": metadata,
                "topic_hierarchy": topic_hierarchy,
                "bookmarks": bookmarks_data
            }
        finally:
            session.close()

    def _build_hierarchical_topics(self) -> List[Dict]:
        topic_info = self.topic_model.get_topic_info()
        hierarchy = {}

        for topic in self.hierarchical_topics:
            if len(topic) < 2:
                self.logger.warning(f"Skipping topic due to insufficient elements: {topic}")
                continue

            parent, child, *_ = topic

            if child not in self.topic_names:
                self.logger.warning(f"Skipping topic {child} as it is not found in topic_names")
                continue

            if child not in hierarchy:
                hierarchy[child] = {
                    'id': self.topic_names[child],
                    'name': self.topic_names[child],
                    'count': topic_info.loc[topic_info['Topic'] == child, 'Count'].values[0],
                    'children': []
                }

            if parent in hierarchy:
                hierarchy[parent]['children'].append(hierarchy[child])

        # Return only root topics (those without parents)
        root_topics = [
            topic for topic_id, topic in hierarchy.items()
            if topic_id not in [child for parent, child, *_ in self.hierarchical_topics if child is not None]
        ]

        return root_topics

    def _get_max_depth(self, hierarchy):
        def get_depth(topic):
            if not topic['children']:
                return 1
            return 1 + max(get_depth(child) for child in topic['children'])
        
        return max(get_depth(topic) for topic in hierarchy)

    def _get_max_depth(self, hierarchy):
        def get_depth(topic):
            if not topic['children']:
                return 1
            return 1 + max(get_depth(child) for child in topic['children'])
        
        return max(get_depth(topic) for topic in hierarchy)

    def update_parameters(self, new_params: Dict):
        if self.is_initializing:
            raise RuntimeError("Cannot update parameters while model is initializing")

        self.is_ready = False
        self.is_initializing = True
        self.is_fitted = False

        try:
            self.initialize(**new_params)
            self.fit_model()
            self.process_bookmarks()
        except Exception as e:
            logger.error(f"Error updating parameters: {str(e)}")
            raise
        finally:
            self.is_initializing = False

def create_bookmark_organizer(db_url='sqlite:///bookmarks.db'):
    return BookmarkOrganizer(db_url)