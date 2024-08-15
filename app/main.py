import argparse
from flask import Flask
from flask_restx import Api
from flask_cors import CORS
from flask_caching import Cache
from config import Config
from models import init_db
from routes import init_routes
from bookmark_organizer import create_bookmark_organizer
from threading import Thread
import logging

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)

    # Set up CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Set up caching
    cache = Cache(app)

    # Set up API
    api = Api(app, version='1.0', title='Bookmark Organizer API',
              description='An API for organizing bookmarks using BERTopic')

    # Initialize database
    init_db(app)

    # Initialize routes
    init_routes(api)

    # Create and initialize bookmark organizer
    app.organizer = create_bookmark_organizer()

    def initialize_model_async():
        try:
            app.organizer.initialize(
                embedding_model="all-MiniLM-L6-v2",
                nr_topics="auto",
                top_n_words=10
            )
            app.organizer.fit_model()
        except Exception as e:
            app.logger.error(f"Error initializing or fitting model: {str(e)}")

    # Start the initialization in a separate thread
    Thread(target=initialize_model_async).start()

    @app.route('/status')
    @cache.cached(timeout=60)  # Cache for 1 minute
    def status():
        if app.organizer.is_ready and app.organizer.is_fitted:
            return {"status": "ready"}, 200
        elif app.organizer.is_ready and not app.organizer.is_fitted:
            return {"status": "ready but not fitted"}, 202
        elif app.organizer.is_initializing:
            return {"status": "initializing"}, 202
        else:
            return {"status": "not started"}, 500



    return app

    @app.errorhandler(500)
    def handle_500_error(e):
        app.logger.error(f'An unhandled exception occurred: {str(e)}')
        return jsonify(error=str(e)), 500

def main():
    parser = argparse.ArgumentParser(description='Bookmark Organizer API')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()

    app = create_app()
    logging.basicConfig(level=logging.DEBUG)
    app.logger.setLevel(logging.DEBUG)
    app.run(debug=args.debug, port=args.port, threaded=True)

if __name__ == '__main__':
    main()