import argparse
from flask import Flask, jsonify
from flask_restx import Api
from flask_cors import CORS
from flask_caching import Cache
from config import Config
from models import init_db, db
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
    with app.app_context():
        init_db(app)

    # Initialize routes
    api = init_routes(api)  # Update this line

    # Create and initialize bookmark organizer
    app.organizer = create_bookmark_organizer(app.config['SQLALCHEMY_DATABASE_URI'])

    def initialize_model_async():
        with app.app_context():
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
    @cache.cached(timeout=10)  # Cache for 10 seconds
    def status():
        organizer_status = {
            "is_ready": app.organizer.is_ready,
            "is_fitted": app.organizer.is_fitted,
            "is_initializing": app.organizer.is_initializing
        }
        if organizer_status["is_ready"] and organizer_status["is_fitted"]:
            return jsonify({"status": "ready", **organizer_status}), 200
        elif organizer_status["is_initializing"]:
            return jsonify({"status": "initializing", **organizer_status}), 202
        else:
            return jsonify({"status": "not started", **organizer_status}), 500

    @app.errorhandler(500)
    def handle_500_error(e):
        app.logger.error(f'An unhandled exception occurred: {str(e)}')
        return jsonify(error=str(e)), 500

    return app

def main():
    parser = argparse.ArgumentParser(description='Bookmark Organizer API')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()

    app = create_app()
    if args.debug:
        app.logger.setLevel(logging.DEBUG)
    app.run(debug=args.debug, port=args.port, threaded=True)

if __name__ == '__main__':
    main()