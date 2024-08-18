from flask import Flask
from flask_cors import CORS
from flask_restx import Api
from bookmark_organizer import BookmarkOrganizer, DefaultEmbeddingModel, BERTopicModel, BookmarkDatabase
from routes import setup_routes
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/*": {
    "origins": ["chrome-extension://*", "moz-extension://*", "http://localhost:*"],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
}}, supports_credentials=True)

api = Api(app, version='1.0', title='Bookmark Organizer API',
          description='A sophisticated Bookmark Organizer API with topic modeling capabilities')

# Initialize components
embedding_model = DefaultEmbeddingModel()
topic_model = BERTopicModel()
database = BookmarkDatabase()

# Create BookmarkOrganizer instance
bookmark_organizer = BookmarkOrganizer(
    embedding_model=embedding_model,
    topic_model=topic_model,
    database=database
)

# Setup routes
setup_routes(api, bookmark_organizer)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True)