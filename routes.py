from flask_restx import Namespace, Resource, Api
from models import create_models


ns_bookmarks = Namespace('bookmarks', description='Bookmark operations')
ns_topics = Namespace('topics', description='Topic operations')

def setup_routes(api: Api, bookmark_organizer):
    # Create models
    bookmark_model, bookmark_response, hierarchical_topic_model = create_models(api)

    @ns_bookmarks.route('/')
    class BookmarkList(Resource):
        @ns_bookmarks.doc('list_bookmarks')
        @ns_bookmarks.marshal_list_with(bookmark_response)
        def get(self):
            """List all bookmarks"""
            return bookmark_organizer.list_bookmarks()

        @ns_bookmarks.doc('create_bookmark')
        @ns_bookmarks.expect(bookmark_model)
        @ns_bookmarks.marshal_with(bookmark_response, code=201)
        def post(self):
            """Create a new bookmark"""
            return bookmark_organizer.add_bookmark(ns_bookmarks.payload), 201

    @ns_topics.route('/hierarchical')
    class HierarchicalTopics(Resource):
        @ns_topics.doc('get_hierarchical_topics')
        @ns_topics.marshal_with(hierarchical_topic_model)
        def get(self):
            """Get hierarchical topics"""
            return bookmark_organizer.get_hierarchical_topics()

    @ns_topics.route('/update')
    class UpdateTopics(Resource):
        @ns_topics.doc('update_topics')
        def post(self):
            """Update topics for all bookmarks"""
            result = bookmark_organizer.update_topics()
            return result, 200

    # Add namespaces to the API
    api.add_namespace(ns_bookmarks, path='/bookmarks')
    api.add_namespace(ns_topics, path='/topics')