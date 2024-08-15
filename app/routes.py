from flask import request, current_app
from flask_restx import Namespace, Resource, fields
from http import HTTPStatus
from models import Bookmark
from functools import wraps

ns = Namespace('bookmarks', description='Bookmark operations')

status_model = ns.model('Status', {
    'is_ready': fields.Boolean(description='Whether the bookmark organizer is ready'),
    'status': fields.String(description='Current status of the bookmark organizer'),
    'version': fields.String(description='Version of the bookmark organizer')
})

# API models
bookmark_model = ns.model('Bookmark', {
    'url': fields.String(required=True, description='The bookmark URL'),
    'title': fields.String(required=True, description='The bookmark title')
})

bookmarks_input_model = ns.model('BookmarksInput', {
    'bookmarks': fields.List(fields.Nested(bookmark_model), required=True, description='List of bookmarks to add')
})

bookmark_response_model = ns.model('BookmarkResponse', {
    'success': fields.Boolean(description='Whether the operation was successful'),
    'organized_bookmarks': fields.Raw(description='Organized bookmarks by topic'),
    'errors': fields.List(fields.Nested(ns.model('ErrorDetail', {
        'url': fields.String(description='URL of the bookmark that failed'),
        'error': fields.String(description='Error message')
    })), description='List of errors for failed bookmarks')
})

bookmarks_list_model = ns.model('BookmarksList', {
    'bookmarks': fields.List(fields.Nested(ns.model('BookmarkDetail', {
        'url': fields.String(description='The bookmark URL'),
        'title': fields.String(description='The bookmark title'),
        'topic': fields.String(description='The bookmark topic')
    }))),
    'total': fields.Integer(description='Total number of bookmarks'),
    'page': fields.Integer(description='Current page number'),
    'per_page': fields.Integer(description='Number of bookmarks per page'),
    'total_pages': fields.Integer(description='Total number of pages')
})

search_result_model = ns.model('SearchResult', {
    'url': fields.String(description='The bookmark URL'),
    'title': fields.String(description='The bookmark title'),
    'topic': fields.String(description='The bookmark topic'),
    'similarity': fields.Float(description='Similarity score')
})

visualization_model = ns.model('Visualization', {
    'success': fields.Boolean(description='Whether the operation was successful'),
    'visualization_data': fields.Raw(description='Visualization data for bookmarks')
})

update_params_model = ns.model('UpdateParams', {
    'embedding_model': fields.String(description='Embedding model name'),
    'umap_n_neighbors': fields.Integer(description='UMAP n_neighbors parameter'),
    'umap_n_components': fields.Integer(description='UMAP n_components parameter'),
    'umap_min_dist': fields.Float(description='UMAP min_dist parameter'),
    'hdbscan_min_cluster_size': fields.Integer(description='HDBSCAN min_cluster_size parameter'),
    'hdbscan_min_samples': fields.Integer(description='HDBSCAN min_samples parameter'),
    'nr_topics': fields.String(description='Number of topics'),
    'top_n_words': fields.Integer(description='Number of top words per topic')
})

word_score_model = ns.model('WordScore', {
    'word': fields.String(description='Topic word'),
    'score': fields.Float(description='Word score')
})

topic_model = ns.model('Topic', {
    'topic': fields.Integer(description='Topic ID'),
    'count': fields.Integer(description='Number of bookmarks in this topic'),
    'name': fields.String(description='Topic name'),
    'representation': fields.List(fields.Nested(word_score_model), description='Topic representation')
})

hierarchical_topic_model = ns.model('HierarchicalTopic', {
    'id': fields.String(description='Topic ID or name'),
    'name': fields.String(description='Topic name'),
    'parent': fields.String(description='Parent Topic ID or name', required=False),
    'distance': fields.Float(description='Distance from parent topic', required=False),
    'children': fields.List(fields.Nested(lambda: hierarchical_topic_model), required=False)
})

hierarchical_topics_model = ns.model('HierarchicalTopics', {
    'topics': fields.List(fields.Nested(hierarchical_topic_model), description='Hierarchical topic structure')
})

visualization_data_model = ns.model('VisualizationData', {
    'topic_visualization': fields.Raw(description='Topic visualization data'),
    'document_visualization': fields.Raw(description='Document visualization data'),
    'hierarchy_visualization': fields.Raw(description='Hierarchy visualization data')
})

def require_model_ready(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_app.organizer.is_ready:
            return {"success": False, "error": "Model is still initializing. Please try again later."}, HTTPStatus.SERVICE_UNAVAILABLE
        if not current_app.organizer.is_fitted:
            return {"success": False, "error": "Model is not fitted yet. Please add some bookmarks first."}, HTTPStatus.SERVICE_UNAVAILABLE
        return f(*args, **kwargs)
    return decorated

def init_routes(api):
    api.add_namespace(ns)

    @ns.route('/status')
    class Status(Resource):
        @ns.marshal_with(status_model)
        def get(self):
            """Get the current status of the bookmark organizer"""
            return {
                'is_ready': current_app.organizer.is_ready,
                'status': 'ready' if current_app.organizer.is_ready else 'initializing',
                'version': current_app.config.get('VERSION', 'unknown')
            }

    @ns.route('/process')
    class ProcessBookmarks(Resource):
        @ns.marshal_with(bookmark_response_model)
        @require_model_ready
        def post(self):
            """Process all bookmarks in the database and organize them by topics"""
            try:
                result = current_app.organizer.process_bookmarks()
                return {"success": True, "organized_bookmarks": result}
            except Exception as e:
                current_app.logger.error(f"Error processing bookmarks: {str(e)}")
                return {"success": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR

    @ns.route('/add')
    class AddBookmarks(Resource):
        @ns.expect(bookmarks_input_model)
        @ns.marshal_with(bookmark_response_model)
        def post(self):
            """Add one or more bookmarks"""
            data = request.json
            bookmarks = data.get('bookmarks', [])
            
            if not isinstance(bookmarks, list):
                return {"success": False, "error": "Invalid input: 'bookmarks' must be a list"}, HTTPStatus.BAD_REQUEST

            results = []
            errors = []

            for bookmark in bookmarks:
                try:
                    result = current_app.organizer.add_bookmark(bookmark)
                    results.append(result)
                except Exception as e:
                    current_app.logger.error(f"Error adding bookmark {bookmark.get('url', 'unknown')}: {str(e)}")
                    errors.append({"url": bookmark.get('url', 'unknown'), "error": str(e)})

            organized_bookmarks = {}
            for result in results:
                for topic, bookmarks in result.items():
                    if topic not in organized_bookmarks:
                        organized_bookmarks[topic] = []
                    organized_bookmarks[topic].extend(bookmarks)

            return {
                "success": len(errors) == 0,
                "organized_bookmarks": organized_bookmarks,
                "errors": errors
            }

    @ns.route('/list')
    @ns.param('topic', 'Filter bookmarks by topic (optional)')
    @ns.param('page', 'Page number (default: 1)')
    @ns.param('per_page', 'Number of bookmarks per page (default: 20)')
    class ListBookmarks(Resource):
        @ns.marshal_with(bookmarks_list_model)
        def get(self):
            """List all bookmarks, optionally filtered by topic"""
            topic = request.args.get('topic')
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 20))
            return current_app.organizer.list_bookmarks(topic, page, per_page)

    @ns.route('/search')
    @ns.param('q', 'Search query')
    class SearchBookmarks(Resource):
        @ns.marshal_with(search_result_model)
        @require_model_ready
        def get(self):
            """Search bookmarks by keyword"""
            query = request.args.get('q')
            if not query:
                return {"error": "Search query is required"}, HTTPStatus.BAD_REQUEST
            return current_app.organizer.search_bookmarks(query)

    @ns.route('/topics')
    class Topics(Resource):
        @ns.marshal_with(topic_model)
        def get(self):
            """Get all topics and their bookmark counts"""
            return current_app.organizer.get_topics()

    @ns.route('/visualization')
    class Visualization(Resource):
        @ns.marshal_with(visualization_model)
        @require_model_ready
        def get(self):
            """Get visualization data for bookmarks"""
            try:
                visualization_data = current_app.organizer.get_visualization_data()
                return {"success": True, "visualization_data": visualization_data}
            except Exception as e:
                current_app.logger.error(f"Error getting visualization data: {str(e)}")
                return {"success": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR

            
    @ns.route('/hierarchical_topics')
    class HierarchicalTopics(Resource):
        @ns.marshal_with(hierarchical_topics_model)
        @require_model_ready
        def get(self):
            """Get hierarchical topic structure"""
            topics = current_app.organizer.get_hierarchical_topics()
            return {"topics": topics}

    @ns.route('/visualization_data')
    class VisualizationData(Resource):
        @ns.marshal_with(visualization_data_model)  # You'll need to define this model
        @require_model_ready
        def get(self):
            """Get visualization data for topics and documents"""
            return current_app.organizer.get_visualization_data()
        
    @ns.route('/update_params')
    class UpdateParams(Resource):
        @ns.expect(update_params_model)
        def post(self):
            """Update the parameters for the bookmark organizer"""
            new_params = request.json
            try:
                current_app.organizer.update_parameters(new_params)
                return {"success": True, "message": "Parameters updated successfully"}
            except Exception as e:
                current_app.logger.error(f"Error updating parameters: {str(e)}")
                return {"success": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR