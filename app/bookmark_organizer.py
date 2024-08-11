from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import numpy as np
from typing import List, Dict
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from models import Bookmark, Base
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class BookmarkOrganizer:
    def __init__(self, db_url='sqlite:///bookmarks.db'):
        self.embedding_model = None
        self.umap_model = None
        self.hdbscan_model = None
        self.topic_model = None
        self.is_ready = False
        self.is_initializing = False
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

    def initialize(self, embedding_model="all-MiniLM-L6-v2", umap_n_neighbors=15, umap_n_components=5, 
                   umap_min_dist=0.0, hdbscan_min_cluster_size=15, hdbscan_min_samples=10, 
                   nr_topics="auto", top_n_words=10):
        if self.is_initializing:
            return
        self.is_initializing = True
        logger.info("Initializing models...")
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.umap_model = umap.UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, 
                                        min_dist=umap_min_dist, metric='cosine')
            self.hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, 
                                                 min_samples=hdbscan_min_samples, metric='euclidean', 
                                                 cluster_selection_method='eom')
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model,
                nr_topics=nr_topics,
                top_n_words=top_n_words
            )
            self.is_ready = True
            logger.info("Model initialization complete.")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
        finally:
            self.is_initializing = False

    def process_bookmarks(self) -> Dict:
        if not self.is_ready:
            raise RuntimeError("Model is not initialized")

        session = self.Session()
        try:
            bookmarks = session.query(Bookmark).all()
            texts = [f"{b.title} {b.url}" for b in bookmarks]
            
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            topics, _ = self.topic_model.fit_transform(texts, embeddings)

            organized_bookmarks = {}
            for bookmark, topic, embedding in zip(bookmarks, topics, embeddings):
                topic_name = f"Topic_{topic}" if topic != -1 else "Uncategorized"
                if topic_name not in organized_bookmarks:
                    organized_bookmarks[topic_name] = []
                
                organized_bookmarks[topic_name].append({
                    "url": bookmark.url,
                    "title": bookmark.title,
                })

                bookmark.embedding = embedding.tolist()
                bookmark.topic = topic_name
                session.add(bookmark)

            session.commit()
            return organized_bookmarks
        except Exception as e:
            session.rollback()
            logger.error(f"Error processing bookmarks: {str(e)}")
            raise
        finally:
            session.close()

    def add_bookmark(self, bookmark: Dict) -> Dict:
        if not self.is_ready:
            raise RuntimeError("Model is not initialized")

        session = self.Session()
        try:
            text = f"{bookmark['title']} {bookmark['url']}"
            embedding = self.embedding_model.encode([text], show_progress_bar=False)[0]
            topic, _ = self.topic_model.transform([text], [embedding])

            topic_name = f"Topic_{topic[0]}" if topic[0] != -1 else "Uncategorized"
            
            new_bookmark = Bookmark(url=bookmark["url"], title=bookmark["title"], 
                                    embedding=embedding.tolist(), topic=topic_name)
            session.add(new_bookmark)
            session.commit()

            return {topic_name: [{"url": bookmark["url"], "title": bookmark["title"]}]}
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
        if not self.is_ready:
            raise RuntimeError("Model is not initialized")

        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]

        session = self.Session()
        try:
            all_bookmarks = session.query(Bookmark).all()
            results = []
            for bookmark in all_bookmarks:
                similarity = np.dot(query_embedding, np.array(bookmark.embedding))
                results.append((bookmark, similarity))

            results.sort(key=lambda x: x[1], reverse=True)
            return [
                {
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
        session = self.Session()
        try:
            topics = session.query(Bookmark.topic, func.count(Bookmark.id)).group_by(Bookmark.topic).all()
            return [{"topic": topic, "count": count} for topic, count in topics]
        finally:
            session.close()

    def get_visualization_data(self) -> Dict:
        session = self.Session()
        try:
            bookmarks = session.query(Bookmark).all()
            
            embeddings = np.array([bookmark.embedding for bookmark in bookmarks])
            pca = PCA(n_components=2)
            projections = pca.fit_transform(embeddings)
            
            visualization_data = {
                "nodes": [
                    {
                        "id": bookmark.id,
                        "url": bookmark.url,
                        "title": bookmark.title,
                        "topic": bookmark.topic,
                        "x": float(projections[i][0]),
                        "y": float(projections[i][1])
                    }
                    for i, bookmark in enumerate(bookmarks)
                ],
                "links": []  # You can add links between related bookmarks if needed
            }
            return visualization_data
        finally:
            session.close()

    def update_parameters(self, new_params: Dict):
        if self.is_initializing:
            raise RuntimeError("Cannot update parameters while model is initializing")

        # Reset the model
        self.is_ready = False
        self.is_initializing = True

        try:
            # Update the parameters
            self.initialize(**new_params)
            # Re-process all bookmarks with new parameters
            self.process_bookmarks()
        except Exception as e:
            logger.error(f"Error updating parameters: {str(e)}")
            raise
        finally:
            self.is_initializing = False

def create_bookmark_organizer(db_url='sqlite:///bookmarks.db'):
    return BookmarkOrganizer(db_url)