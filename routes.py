from flask_restx import Namespace, Resource, Api
from models import create_models
from bookmark_organizer import BookmarkTopicTree
from flask import request
import logging

ns_bookmarks = Namespace('bookmarks', description='Bookmark operations')
ns_topics = Namespace('topics', description='Topic operations')

def setup_routes(api: Api, bookmark_organizer):
    # Create models
    bookmark_model, bookmark_response, hierarchical_topic_model, topic_tree = create_models(api)

    @ns_bookmarks.route('/')
    class BookmarkList(Resource):
        @ns_bookmarks.doc('list_bookmarks')
        @ns_bookmarks.marshal_list_with(bookmark_response)
        @ns_bookmarks.response(500, 'Internal Server Error')
        def get(self):
            """List all bookmarks"""
            try:
                return bookmark_organizer.list_bookmarks()
            except Exception as e:
                return {'message': 'An error occurred while fetching bookmarks', 'error': str(e)}, 500

        @ns_bookmarks.doc('create_bookmark')
        @ns_bookmarks.expect(bookmark_model)
        @ns_bookmarks.marshal_with(bookmark_response, code=201)
        @ns_bookmarks.response(400, 'Validation Error')
        @ns_bookmarks.response(500, 'Internal Server Error')
        def post(self):
            """Create a new bookmark"""
            try:
                return bookmark_organizer.add_bookmark(ns_bookmarks.payload), 201
            except ValueError as e:
                return {'message': 'Invalid input', 'error': str(e)}, 400
            except Exception as e:
                return {'message': 'An error occurred while creating the bookmark', 'error': str(e)}, 500

    @ns_bookmarks.route('/<int:id>')
    class Bookmark(Resource):
        @ns_bookmarks.doc('get_bookmark')
        @ns_bookmarks.marshal_with(bookmark_response)
        @ns_bookmarks.response(404, 'Bookmark not found')
        @ns_bookmarks.response(500, 'Internal Server Error')
        def get(self, id):
            """Fetch a bookmark by its ID"""
            try:
                bookmark = bookmark_organizer.get_bookmark(id)
                if bookmark is None:
                    return {'message': 'Bookmark not found'}, 404
                return bookmark
            except Exception as e:
                return {'message': 'An error occurred while fetching the bookmark', 'error': str(e)}, 500

        @ns_bookmarks.doc('update_bookmark')
        @ns_bookmarks.expect(bookmark_model)
        @ns_bookmarks.marshal_with(bookmark_response)
        @ns_bookmarks.response(404, 'Bookmark not found')
        @ns_bookmarks.response(400, 'Validation Error')
        @ns_bookmarks.response(500, 'Internal Server Error')
        def put(self, id):
            """Update a bookmark"""
            try:
                updated_bookmark = bookmark_organizer.update_bookmark(id, ns_bookmarks.payload)
                if updated_bookmark is None:
                    return {'message': 'Bookmark not found'}, 404
                return updated_bookmark
            except ValueError as e:
                return {'message': 'Invalid input', 'error': str(e)}, 400
            except Exception as e:
                return {'message': 'An error occurred while updating the bookmark', 'error': str(e)}, 500

        @ns_bookmarks.doc('delete_bookmark')
        @ns_bookmarks.response(204, 'Bookmark deleted')
        @ns_bookmarks.response(404, 'Bookmark not found')
        @ns_bookmarks.response(500, 'Internal Server Error')
        def delete(self, id):
            """Delete a bookmark"""
            try:
                if bookmark_organizer.delete_bookmark(id):
                    return '', 204
                return {'message': 'Bookmark not found'}, 404
            except Exception as e:
                return {'message': 'An error occurred while deleting the bookmark', 'error': str(e)}, 500

    @ns_topics.route('/hierarchical')
    class HierarchicalTopics(Resource):
        @ns_topics.doc('get_hierarchical_topics')
        @ns_topics.marshal_with(hierarchical_topic_model)
        @ns_topics.response(500, 'Internal Server Error')
        def get(self):
            """Get hierarchical topics"""
            try:
                return bookmark_organizer.get_hierarchical_topics()
            except Exception as e:
                return {'message': 'An error occurred while fetching hierarchical topics', 'error': str(e)}, 500

    @ns_topics.route('/update')
    class UpdateTopics(Resource):
        @ns_topics.doc('update_topics')
        @ns_topics.response(200, 'Topics updated successfully')
        @ns_topics.response(500, 'Internal Server Error')
        def post(self):
            """Update topics for all bookmarks"""
            try:
                result = bookmark_organizer.update_topics()
                return result, 200
            except Exception as e:
                return {'message': 'An error occurred while updating topics', 'error': str(e)}, 500

    @ns_topics.route('/tree')
    class TopicTree(Resource):
        @ns_topics.doc('get_topic_tree')
        @ns_topics.response(200, 'Success')
        @ns_topics.response(500, 'Internal Server Error')
        def get(self):
            """Get the bookmark topic tree"""
            self.logger = logging.getLogger(__name__)
            try:
                tree_builder = BookmarkTopicTree(bookmark_organizer)
                topic_tree = tree_builder.build_tree()
                
                # Log the entire topic_tree
                # elf.logger.debug(f"Full topic tree: {topic_tree}")
                
                # Check if the tree is empty
                if not topic_tree:
                    self.logger.warning("Topic tree is empty")
                    return {"message": "Topic tree is empty"}, 204
                
                # Log the first topic and its bookmarks
                #first_topic_id = next(iter(topic_tree))
                #first_topic = topic_tree[first_topic_id]
                # self.logger.debug(f"First topic: ID={first_topic_id}, Name='{first_topic['name']}', Bookmarks={first_topic['bookmark_count']}")
                #if first_topic['bookmarks']:
                    #self.logger.debug(f"First bookmark in first topic: {first_topic['bookmarks'][0]}")
                
                response = {"root": topic_tree}
                return response
            except Exception as e:
                self.logger.exception(f"Error in get_topic_tree: {str(e)}")
                return {'message': 'An error occurred while building the topic tree', 'error': str(e)}, 500

    # Add namespaces to the API
    api.add_namespace(ns_bookmarks, path='/bookmarks')
    api.add_namespace(ns_topics, path='/topics')