from flask import Flask, jsonify
from flask_cors import CORS
from flask_restx import Api
from bookmark_organizer import init_db, BookmarkOrganizer
from routes import setup_routes

app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/*": {
    "origins": ["chrome-extension://*", "moz-extension://*", "http://localhost:*"],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
}}, supports_credentials=True)

api = Api(app, version='1.0', title='BERTopic Bookmark API',
          description='A simple BERTopic Bookmark API with Swagger support')

# Initialize database
init_db()

# Create BookmarkOrganizer instance
bookmark_organizer = BookmarkOrganizer()

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