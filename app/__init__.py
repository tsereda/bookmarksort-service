from flask import Flask
from flask_cors import CORS
from flask_restx import Api

from .api.bookmarks import bookmarks_ns
from .api.topics import topics_ns
from .api.visualizations import visualizations_ns
from .api.search import search_ns
from .services.bookmark_service import BookmarkService
from .services.topic_service import TopicService
from .services.visualization_service import VisualizationService
from .services.embedding_service import EmbeddingService
from .utils.database import BookmarkDatabase

def create_app():
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, resources={r"/*": {
        "origins": ["chrome-extension://*", "moz-extension://*", "http://localhost:*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
    }}, supports_credentials=True)

    api = Api(app, version='1.0', title='Bookmark Organizer API',
              description='A sophisticated Bookmark Organizer API with topic modeling capabilities')

    # Initialize services
    database = BookmarkDatabase()
    embedding_service = EmbeddingService(database)
    topic_service = TopicService(database, embedding_service)
    bookmark_service = BookmarkService(database, embedding_service, topic_service)
    visualization_service = VisualizationService(topic_service, bookmark_service)

    # Register namespaces
    api.add_namespace(bookmarks_ns)
    api.add_namespace(topics_ns)
    api.add_namespace(visualizations_ns)
    api.add_namespace(search_ns)

    # Inject services into namespaces
    bookmarks_ns.bookmark_service = bookmark_service
    bookmarks_ns.embedding_service = embedding_service
    topics_ns.topic_service = topic_service
    visualizations_ns.visualization_service = visualization_service
    search_ns.bookmark_service = bookmark_service

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    return app