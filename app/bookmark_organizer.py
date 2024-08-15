from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Union
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from models import Bookmark, Base
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class BookmarkOrganizer:
    def __init__(self, db_url='sqlite:///bookmarks.db'):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self.topic_model = None
        self.is_ready = False
        self.is_initializing = False
        self.is_fitted = False
        self.hierarchical_topics = None

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
        finally:
            self.is_initializing = False

    def fit_model(self):
        if not self.is_ready:
            raise RuntimeError("BERTopic model is not initialized")
        
        session = self.Session()
        try:
            bookmarks = session.query(Bookmark).all()
            if not bookmarks:
                logger.warning("No bookmarks available to fit the model.")
                return

            texts = [f"{b.title} {b.url}" for b in bookmarks]
            self.topic_model.fit(texts)
            self.is_fitted = True
            logger.info("BERTopic model fitted successfully.")
            
            # Generate hierarchical topics
            self.hierarchical_topics = self.topic_model.hierarchical_topics(
                docs=texts,
                linkage_function=None,
                distance_function=None
            )
            logger.info(f"Hierarchical topics generated. Type: {type(self.hierarchical_topics)}")
            logger.info(f"Hierarchical topics shape: {self.hierarchical_topics.shape}")
            logger.info(f"Hierarchical topics columns: {self.hierarchical_topics.columns}")
            logger.debug(f"Hierarchical topics content:\n{self.hierarchical_topics}")
        except Exception as e:
            logger.error(f"Failed to fit BERTopic model: {str(e)}")
            logger.exception("Stack trace:")
        finally:
            session.close()

    def partial_fit(self, new_texts: List[str]):
        if not self.is_ready:
            raise RuntimeError("BERTopic model is not initialized")
        
        try:
            self.topic_model.partial_fit(new_texts)
            logger.info("Partial fit successful.")
        except Exception as e:
            logger.error(f"Failed to perform partial fit: {str(e)}")

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
        logger.info("Entering get_hierarchical_topics method")
        if self.hierarchical_topics is None:
            logger.warning("Hierarchical topics have not been generated")
            return []
        
        logger.debug(f"Hierarchical topics type: {type(self.hierarchical_topics)}")
        logger.debug(f"Hierarchical topics columns: {self.hierarchical_topics.columns}")
        logger.debug(f"First few rows of hierarchical_topics:\n{self.hierarchical_topics.head()}")
        
        if isinstance(self.hierarchical_topics, pd.DataFrame):
            topics = []
            for _, row in self.hierarchical_topics.iterrows():
                topic = {
                    "id": str(row.iloc[0]),
                    "name": str(row.iloc[0]),
                    "parent": str(row.iloc[1]) if pd.notnull(row.iloc[1]) else None,
                }
                if len(row) > 2:
                    distance_values = row.iloc[2:][pd.notnull(row.iloc[2:])]
                    if len(distance_values) > 0:
                        distance = distance_values.iloc[0]
                        if isinstance(distance, list):
                            topic["distance"] = float(distance[0])
                        else:
                            topic["distance"] = float(distance)
                    else:
                        topic["distance"] = None
                else:
                    topic["distance"] = None
                topics.append(topic)
            
            # Build hierarchical structure
            topic_map = {topic['id']: topic for topic in topics}
            root_topics = []
            for topic in topics:
                if topic['parent'] is None or topic['parent'] not in topic_map:
                    root_topics.append(topic)
                else:
                    parent = topic_map[topic['parent']]
                    if 'children' not in parent:
                        parent['children'] = []
                    parent['children'].append(topic)
            
            logger.debug(f"Processed topics: {topics}")
            logger.debug(f"Root topics: {root_topics}")
            return root_topics
        else:
            logger.error(f"Unexpected hierarchical_topics type: {type(self.hierarchical_topics)}")
            return []

    def get_visualization_data(self) -> Dict:
        if not self.is_ready or not self.is_fitted:
            raise RuntimeError("BERTopic model is not initialized or fitted")
        
        session = self.Session()
        try:
            bookmarks = session.query(Bookmark).all()
            docs = [f"{b.title} {b.url}" for b in bookmarks]
            
            topic_visualization = self.topic_model.visualize_topics()
            document_visualization = self.topic_model.visualize_documents(docs)
            hierarchy_visualization = self.topic_model.visualize_hierarchy()
            
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(v) for v in obj]
                else:
                    return obj
            
            return {
                "topic_visualization": convert_to_json_serializable(topic_visualization.to_dict()),
                "document_visualization": convert_to_json_serializable(document_visualization.to_dict()),
                "hierarchy_visualization": convert_to_json_serializable(hierarchy_visualization.to_dict())
            }
        finally:
            session.close()

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