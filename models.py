from flask_restx import fields, Model

def create_models(api):
    bookmark_model = api.model('Bookmark', {
        'title': fields.String(required=True, description='The bookmark title'),
        'url': fields.String(required=True, description='The bookmark URL'),
        'tags': fields.List(fields.String, description='List of tags'),
        'embedding': fields.List(fields.Float, description='Embedding vector'),
        'topic': fields.Integer(description='Topic ID', allow_null=True)
    })

    bookmark_response = api.inherit('BookmarkResponse', bookmark_model, {
        'id': fields.Integer(description='The bookmark ID')
    })

    hierarchical_topic_model = api.model('HierarchicalTopic', {
        'Parent_ID': fields.String(description='Parent topic ID'),
        'Parent_Name': fields.String(description='Parent topic name'),
        'Topics': fields.List(fields.Integer, description='List of child topic IDs'),
        'Child_Left_ID': fields.String(description='Left child topic ID'),
        'Child_Left_Name': fields.String(description='Left child topic name'),
        'Child_Right_ID': fields.String(description='Right child topic ID'),
        'Child_Right_Name': fields.String(description='Right child topic name'),
        'Distance': fields.Float(description='Distance between child topics')
    })

    return bookmark_model, bookmark_response, hierarchical_topic_model