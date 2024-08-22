from typing import List, Dict, Any
from ..utils.database import BookmarkDatabase
from .embedding_service import EmbeddingService
from .topic_service import TopicService
import asyncio

class BookmarkService:
    def __init__(self, database: BookmarkDatabase, embedding_service: EmbeddingService, topic_service: TopicService):
        self.database = database
        self.embedding_service = embedding_service
        self.topic_service = topic_service

    def get_bookmarks(self) -> List[Dict[str, Any]]:
        return self.database.get_bookmarks()

    def add_bookmark(self, bookmark_data: Dict[str, Any]) -> Dict[str, Any]:
        bookmark_id = self.database.add_bookmark(bookmark_data)
        
        # Generate embedding for the new bookmark
        embedding = self.embedding_service.generate_embedding_for_bookmark(bookmark_data)
        self.database.update_bookmark_embedding(bookmark_id, embedding)
        
        # Assign topic to the new bookmark if topics have been created
        if self.topic_service.topic_model is not None:
            new_topic = self.topic_service.assign_topic_to_bookmark(bookmark_data, embedding)
            self.database.update_bookmark_topic(bookmark_id, new_topic)
        
        return {"message": "Bookmark added successfully", "id": bookmark_id}

    def get_untagged_bookmarks(self, limit: int = None) -> List[Dict[str, Any]]:
        return self.database.get_untagged_bookmarks(limit)

    def update_bookmark_tags(self, bookmark_id: int, tags: List[str]):
        self.database.update_bookmark_tags(bookmark_id, tags)

    def update_bookmark_topic(self, bookmark_id: int, topic: int):
        self.database.update_bookmark_topic(bookmark_id, topic)

    def search_bookmarks(self, query: str) -> List[Dict[str, Any]]:
        bookmarks = self.database.get_bookmarks()
        return [b for b in bookmarks if query.lower() in b['title'].lower() or query.lower() in b['url'].lower()]

    async def batch_tag_bookmarks(self, bookmarks: List[Dict[str, Any]], max_tags: int = 10) -> Dict[str, Any]:
        if not bookmarks:
            return {"message": "No bookmarks to tag", "tagged_count": 0}

        tags = await self._generate_tags_for_bookmarks(bookmarks, max_tags)
        tagged_count = 0
        for bookmark, bookmark_tags in zip(bookmarks, tags):
            self.update_bookmark_tags(bookmark['id'], bookmark_tags)
            tagged_count += 1

        return {
            "message": f"Tagged {tagged_count} bookmarks",
            "tagged_count": tagged_count
        }

    async def batch_tag_all_untagged_bookmarks(self, max_concurrent: int = 5, max_tags: int = 10, batch_size: int = 20) -> Dict[str, Any]:
        untagged_bookmarks = self.get_untagged_bookmarks()
        
        if not untagged_bookmarks:
            return {"message": "No untagged bookmarks found"}

        total_bookmarks = len(untagged_bookmarks)
        tagged_count = 0
        progress_updates = []

        async def process_batch(batch):
            nonlocal tagged_count
            try:
                result = await self.batch_tag_bookmarks(batch, max_tags)
                tagged_count += result['tagged_count']
                progress = (tagged_count / total_bookmarks) * 100
                return f"Progress: {progress:.2f}% - Tagged {tagged_count} out of {total_bookmarks} bookmarks"
            except Exception as e:
                return f"Error tagging batch: {str(e)}"

        semaphore = asyncio.Semaphore(max_concurrent)
        async def process_with_semaphore(batch):
            async with semaphore:
                return await process_batch(batch)

        batches = [untagged_bookmarks[i:i+batch_size] for i in range(0, total_bookmarks, batch_size)]
        results = await asyncio.gather(*[process_with_semaphore(batch) for batch in batches])
        progress_updates.extend(results)

        return {
            "message": f"Tagged {tagged_count} out of {total_bookmarks} bookmarks",
            "progress_updates": progress_updates
        }

    async def _generate_tags_for_bookmarks(self, bookmarks: List[Dict[str, Any]], max_tags: int) -> List[List[str]]:
        # This method should be implemented to generate tags for bookmarks
        # You might want to use an AI service or some other method to generate tags
        # For now, we'll return empty tag lists
        return [[] for _ in bookmarks]