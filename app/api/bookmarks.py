from flask import request, jsonify
from flask_restx import Namespace, Resource
from ..models.api_models import create_models
import asyncio

bookmarks_ns = Namespace('bookmarks', description='Bookmark operations')

bookmark_model, bookmark_response, _, _, embedding_request = create_models(bookmarks_ns)

@bookmarks_ns.route('/')
class BookmarkList(Resource):
    @bookmarks_ns.doc('list_bookmarks',
        description='Retrieve a list of all bookmarks with their associated topics.',
        responses={
            200: 'Success. Returns a list of all bookmarks.',
            500: 'Server error. An error occurred while fetching bookmarks.'
        })
    @bookmarks_ns.marshal_list_with(bookmark_response)
    def get(self):
        """List all bookmarks with their topics"""
        try:
            bookmarks = bookmarks_ns.bookmark_service.get_bookmarks()
            if not bookmarks:
                return [], 200
            return bookmarks, 200
        except Exception as e:
            bookmarks_ns.logger.error(f"Error fetching bookmarks: {str(e)}")
            return {'message': 'An error occurred while fetching bookmarks', 'error': str(e)}, 500

    @bookmarks_ns.doc('add_bookmark',
        description='Add a new bookmark to the database.',
        responses={
            201: 'Bookmark added successfully.',
            500: 'Server error. An error occurred while adding the bookmark.'
        })
    @bookmarks_ns.expect(bookmark_model)
    def post(self):
        """Add a new bookmark"""
        try:
            data = request.json
            result = bookmarks_ns.bookmark_service.add_bookmark(data)
            return result, 201
        except Exception as e:
            return {'message': 'An error occurred while adding the bookmark', 'error': str(e)}, 500

@bookmarks_ns.route('/embeddings')
class BookmarkEmbeddings(Resource):
    @bookmarks_ns.doc('generate_embeddings',
        description='Generate embeddings for all bookmarks.',
        responses={
            200: 'Embeddings generated successfully.',
            400: 'Bad request. Invalid embedding model.',
            500: 'Server error. An error occurred while generating embeddings.'
        })
    @bookmarks_ns.expect(embedding_request)
    def post(self):
        """Generate embeddings for all bookmarks"""
        try:
            data = request.json
            embedding_model = data.get('embedding_model', 'all-MiniLM-L6-v2')
            result = bookmarks_ns.embedding_service.generate_embeddings(embedding_model)
            return result, 200
        except ValueError as e:
            return {'message': str(e)}, 400
        except Exception as e:
            return {'message': 'An error occurred while generating embeddings', 'error': str(e)}, 500

@bookmarks_ns.route('/batch_tag')
class BatchTagBookmarks(Resource):
    @bookmarks_ns.doc('batch_tag_bookmarks',
        description='Tag all untagged bookmarks concurrently with progress updates.',
        responses={
            200: 'Batch tagging process completed successfully.',
            400: 'Bad request. Invalid parameters.',
            500: 'Server error. An error occurred while batch tagging bookmarks.'
        })
    def post(self):
        """Tag all untagged bookmarks concurrently with progress updates"""
        try:
            if request.is_json:
                data = request.get_json()
            elif request.form:
                data = request.form
            else:
                data = {}

            max_concurrent = int(data.get('max_concurrent', 5))
            max_tags = int(data.get('max_tags', 10))
            batch_size = int(data.get('batch_size', 20))

            if max_concurrent <= 0 or max_tags <= 0 or batch_size <= 0:
                return {'message': 'Invalid max_concurrent, max_tags, or batch_size. All must be positive integers.'}, 400

            result = bookmarks_ns.bookmark_service.batch_tag_all_untagged_bookmarks(max_concurrent, max_tags, batch_size)
            return result, 200
        except ValueError as ve:
            return {'message': f'Invalid parameter: {str(ve)}'}, 400
        except Exception as e:
            return {'message': 'An error occurred while batch tagging bookmarks', 'error': str(e)}, 500