from typing import List, Dict, Any
from ..utils.database import BookmarkDatabase
from .embedding_service import EmbeddingService
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
from scipy.cluster import hierarchy as sch
from openai import AsyncOpenAI
import asyncio

class TopicService:
    def __init__(self, database: BookmarkDatabase, embedding_service: EmbeddingService):
        self.database = database
        self.embedding_service = embedding_service
        self.topic_model = None
        self.umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
        self.hierarchical_topics = None
        self.client = AsyncOpenAI()

        # HDBSCAN clustering parameters
        self.min_cluster_size = 6
        self.min_samples = None
        self.metric = 'euclidean'
        self.cluster_selection_method = 'eom'
        self.prediction_data = True

    def get_embeddings(self) -> np.ndarray:
        return self.embedding_service.get_embeddings()
    
    def reduce_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self.umap_model is None:
            self.umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
        return self.umap_model.fit_transform(embeddings)

    def create_topics(self):
        bookmarks = self.database.get_bookmarks()
        if not bookmarks:
            return {"message": "No bookmarks found. Add some bookmarks first."}

        embeddings = self.database.get_embeddings()
        if embeddings.size == 0:
            return {"message": "Failed to generate embeddings. Please try again."}

        docs = [f"{b['title']} {b['url']} {' '.join(b['tags'])}" for b in bookmarks]

        try:
            if self.topic_model is None:
                hdbscan_model = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric=self.metric,
                    cluster_selection_method=self.cluster_selection_method,
                    prediction_data=self.prediction_data
                )
                self.topic_model = BERTopic(
                    embedding_model=None,
                    hdbscan_model=hdbscan_model,
                    umap_model=self.umap_model
                )
            
            topics, _ = self.topic_model.fit_transform(docs, embeddings=embeddings)
            
            for bookmark, topic in zip(bookmarks, topics):
                self.database.update_bookmark_topic(bookmark['id'], int(topic))
            
            self.hierarchical_topics = self.create_hierarchical_topics(docs)
            
            return {"message": f"Created topics for {len(bookmarks)} bookmarks"}
        except Exception as e:
            return {"message": f"An error occurred while creating topics: {str(e)}"}

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

    def get_tree_json(self):
        if self.hierarchical_topics is None:
            raise ValueError("Hierarchical topics have not been created. Call create_topics() first.")
        
        hier_topics = self.hierarchical_topics
        root = {"name": "Topics", "children": []}

        def build_tree(node, parent_id):
            children = hier_topics[
                (hier_topics['Parent_ID'] == parent_id) &
                (hier_topics['Child_Left_ID'] != hier_topics['Child_Right_ID'])
            ]
            for _, row in children.iterrows():
                left_child = {"name": row['Child_Left_Name'], "children": []}
                right_child = {"name": row['Child_Right_Name'], "children": []}
                node["children"].extend([left_child, right_child])
                build_tree(left_child, row['Child_Left_ID'])
                build_tree(right_child, row['Child_Right_ID'])

        build_tree(root, hier_topics['Parent_ID'].max())
        return root

    def assign_topic_to_bookmark(self, bookmark_data: Dict[str, Any], embedding: np.ndarray) -> int:
        if self.topic_model is None:
            raise ValueError("Topics have not been created. Call create_topics() first.")
        
        doc = f"{bookmark_data['title']} {bookmark_data['url']} {' '.join(bookmark_data.get('tags', []))}"
        topic, _ = self.topic_model.transform([doc], embeddings=[embedding])
        return topic[0]

    async def regenerate_topic_names(self):
        if self.topic_model is None:
            raise ValueError("Topics have not been created. Call create_topics() first.")

        topics = self.topic_model.get_topics()
        new_names = {}

        for topic_id, topic_words in topics.items():
            if topic_id != -1:  # Skip the outlier topic
                words = [word for word, _ in topic_words[:10]]  # Get top 10 words
                new_name = await self._generate_topic_name(words)
                new_names[topic_id] = new_name

        # Update topic names in the model
        self.topic_model.set_topic_labels(new_names)

        # Update topic names in the database (assuming you have a method for this)
        for topic_id, name in new_names.items():
            self.database.update_topic_name(topic_id, name)

        return {"message": f"Regenerated names for {len(new_names)} topics"}

    async def _generate_topic_name(self, words: List[str]) -> str:
        prompt = f"Given the following words representing a topic, generate a short, descriptive name for this topic: {', '.join(words)}"
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates concise and descriptive topic names."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                n=1,
                stop=None,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating topic name: {str(e)}")
            return f"Topic {' '.join(words[:3])}"  # Fallback name