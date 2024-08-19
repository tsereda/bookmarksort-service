from flask import Flask
from flask_cors import CORS
from flask_restx import Api
from bookmark_organizer import BookmarkOrganizer, BookmarkDatabase
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
database = BookmarkDatabase()

# Create BookmarkOrganizer instance
bookmark_organizer = BookmarkOrganizer(database)

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