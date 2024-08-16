from flask import request, current_app
from flask_restx import Namespace, Resource, fields, abort
from http import HTTPStatus
from functools import wraps

# Create namespaces
main_ns = Namespace('', description='Main operations')
bookmarks_ns = Namespace('bookmarks', description='Bookmark operations')
topics_ns = Namespace('topics', description='Topic operations')
visualization_ns = Namespace('visualization', description='Visualization operations')

# Helper function for consistent error responses
def error_response(message, code):
    return {'success': False, 'error': message}, code

# Decorator to check if model is ready
def require_model_ready(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_app.organizer.is_ready:
            abort(503, "Model is still initializing. Please try again later.")
        if not current_app.organizer.is_fitted:
            abort(503, "Model is not fitted yet. Please add some bookmarks first.")
        return f(*args, **kwargs)
    return decorated

# Define models
status_model = main_ns.model('Status', {
    'is_ready': fields.Boolean(description='Whether the bookmark organizer is ready'),
    'is_fitted': fields.Boolean(description='Whether the model is fitted'),
    'status': fields.String(description='Current status of the bookmark organizer'),
    'version': fields.String(description='Version of the bookmark organizer')
})

bookmark_model = bookmarks_ns.model('Bookmark', {
    'url': fields.String(required=True, description='The bookmark URL'),
    'title': fields.String(required=True, description='The bookmark title')
})

bookmark_detail_model = bookmarks_ns.model('BookmarkDetail', {
    'id': fields.Integer(description='The bookmark ID'),
    'url': fields.String(description='The bookmark URL'),
    'title': fields.String(description='The bookmark title'),
    'topic': fields.String(description='The bookmark topic')
})

bookmark_response_model = bookmarks_ns.model('BookmarkResponse', {
    'success': fields.Boolean(description='Whether the operation was successful'),
    'organized_bookmarks': fields.Raw(description='Organized bookmarks by topic'),
    'errors': fields.List(fields.Nested(bookmarks_ns.model('ErrorDetail', {
        'url': fields.String(description='URL of the bookmark that failed'),
        'error': fields.String(description='Error message')
    })), description='List of errors for failed bookmarks')
})

bookmarks_list_model = bookmarks_ns.model('BookmarksList', {
    'bookmarks': fields.List(fields.Nested(bookmark_detail_model)),
    'total': fields.Integer(description='Total number of bookmarks'),
    'page': fields.Integer(description='Current page number'),
    'per_page': fields.Integer(description='Number of bookmarks per page'),
    'total_pages': fields.Integer(description='Total number of pages')
})

search_result_model = bookmarks_ns.model('SearchResult', {
    'id': fields.Integer(description='The bookmark ID'),
    'url': fields.String(description='The bookmark URL'),
    'title': fields.String(description='The bookmark title'),
    'topic': fields.String(description='The bookmark topic'),
    'similarity': fields.Float(description='Similarity score')
})

word_score_model = topics_ns.model('WordScore', {
    'word': fields.String(description='Topic word'),
    'score': fields.Float(description='Word score')
})

topic_model = topics_ns.model('Topic', {
    'topic': fields.Integer(description='Topic ID'),
    'count': fields.Integer(description='Number of bookmarks in this topic'),
    'name': fields.String(description='Topic name'),
    'representation': fields.List(fields.Nested(word_score_model), description='Topic representation')
})

hierarchical_topic_model = topics_ns.model('HierarchicalTopic', {
    'id': fields.String(description='Topic ID or name'),
    'name': fields.String(description='Topic name'),
    'parent': fields.String(description='Parent Topic ID or name', required=False),
    'distance': fields.Float(description='Distance from parent topic', required=False),
    'children': fields.List(fields.Raw(description='Child topics'), required=False)
})

visualization_data_model = visualization_ns.model('VisualizationData', {
    'topics': fields.List(fields.Nested(visualization_ns.model('TopicData', {
        'id': fields.Integer(description='Topic ID'),
        'name': fields.String(description='Topic name'),
        'count': fields.Integer(description='Number of documents in this topic'),
        'top_words': fields.List(fields.String(description='Top words in this topic'))
    }))),
    'documents': fields.List(fields.Nested(visualization_ns.model('DocumentData', {
        'id': fields.Integer(description='Document ID'),
        'topic': fields.Integer(description='Assigned topic ID'),
        'probability': fields.Float(description='Probability of assignment to this topic'),
        'url': fields.String(description='Bookmark URL'),
        'title': fields.String(description='Bookmark title')
    }))),
    'hierarchical_topics': fields.List(fields.Nested(hierarchical_topic_model))
})

update_params_model = main_ns.model('UpdateParams', {
    'embedding_model': fields.String(description='Embedding model name'),
    'nr_topics': fields.String(description='Number of topics'),
    'top_n_words': fields.Integer(description='Number of top words per topic')
})

def init_routes(api):
    api.add_namespace(main_ns)
    api.add_namespace(bookmarks_ns)
    api.add_namespace(topics_ns)
    api.add_namespace(visualization_ns)

    @main_ns.route('/status')
    class Status(Resource):
        @main_ns.marshal_with(status_model)
        def get(self):
            """Get the current status of the bookmark organizer"""
            organizer = current_app.organizer
            return {
                'is_ready': organizer.is_ready,
                'is_fitted': organizer.is_fitted,
                'status': 'ready' if organizer.is_ready and organizer.is_fitted else 'initializing',
                'version': current_app.config.get('VERSION', 'unknown')
            }

    @bookmarks_ns.route('/')
    class BookmarkList(Resource):
        @bookmarks_ns.expect(bookmarks_ns.model('BookmarksInput', {
            'bookmarks': fields.List(fields.Nested(bookmark_model), required=True, description='List of bookmarks to add')
        }))
        @bookmarks_ns.marshal_with(bookmark_response_model)
        def post(self):
            """Add one or more bookmarks"""
            data = request.json
            bookmarks = data.get('bookmarks', [])
            
            if not isinstance(bookmarks, list):
                abort(400, "Invalid input: 'bookmarks' must be a list")

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

        @bookmarks_ns.marshal_with(bookmarks_list_model)
        @bookmarks_ns.param('topic', 'Filter bookmarks by topic (optional)')
        @bookmarks_ns.param('page', 'Page number (default: 1)')
        @bookmarks_ns.param('per_page', 'Number of bookmarks per page (default: 20)')
        def get(self):
            """List all bookmarks, optionally filtered by topic"""
            topic = request.args.get('topic')
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 20))
            return current_app.organizer.list_bookmarks(topic, page, per_page)

    @bookmarks_ns.route('/search')
    class BookmarkSearch(Resource):
        @bookmarks_ns.marshal_list_with(search_result_model)
        @bookmarks_ns.param('q', 'Search query')
        @require_model_ready
        def get(self):
            """Search bookmarks by keyword"""
            query = request.args.get('q')
            if not query:
                abort(400, "Search query is required")
            return current_app.organizer.search_bookmarks(query)

    @bookmarks_ns.route('/process')
    class ProcessBookmarks(Resource):
        @bookmarks_ns.marshal_with(bookmark_response_model)
        @require_model_ready
        def post(self):
            """Process all bookmarks in the database and organize them by topics"""
            try:
                result = current_app.organizer.process_bookmarks()
                return {"success": True, "organized_bookmarks": result}
            except Exception as e:
                current_app.logger.error(f"Error processing bookmarks: {str(e)}")
                return error_response(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    @topics_ns.route('/')
    class TopicList(Resource):
        @topics_ns.marshal_list_with(topic_model)
        @require_model_ready
        def get(self):
            """Get all topics and their bookmark counts"""
            return current_app.organizer.get_topics()

    @topics_ns.route('/hierarchical')
    class HierarchicalTopics(Resource):
        @topics_ns.marshal_with(topics_ns.model('HierarchicalTopics', {
            'topics': fields.List(fields.Nested(hierarchical_topic_model))
        }))
        @require_model_ready
        def get(self):
            """Get hierarchical topic structure"""
            topics = current_app.organizer.get_hierarchical_topics()
            return {"topics": topics}

    @visualization_ns.route('/data')
    class VisualizationData(Resource):
        @visualization_ns.marshal_with(visualization_data_model)
        @require_model_ready
        def get(self):
            """Get visualization data for topics and documents"""
            return current_app.organizer.get_visualization_data()

    @main_ns.route('/update_params')
    class UpdateParams(Resource):
        @main_ns.expect(update_params_model)
        def post(self):
            """Update the parameters for the bookmark organizer"""
            new_params = request.json
            try:
                current_app.organizer.update_parameters(new_params)
                return {"success": True, "message": "Parameters updated successfully"}
            except Exception as e:
                current_app.logger.error(f"Error updating parameters: {str(e)}")
                return error_response(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)
