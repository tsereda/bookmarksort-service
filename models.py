from flask_restx import fields, Model

def create_models(api):
    bookmark_model = api.model('Bookmark', {
        'title': fields.String(required=True, description='The bookmark title'),
        'url': fields.String(required=True, description='The bookmark URL'),
        'tags': fields.List(fields.String, description='List of tags'),
        'embedding': fields.List(fields.Float, description='Embedding vector'),
        'topic': fields.Integer(description='Topic ID')
    })

    bookmark_response = api.inherit('BookmarkResponse', bookmark_model, {
        'id': fields.Integer(description='The bookmark ID')
    })

    return bookmark_model, bookmark_response