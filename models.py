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


    bookmark_in_tree = api.model('BookmarkInTree', {
        'id': fields.Integer(description='The bookmark ID'),
        'title': fields.String(description='The bookmark title'),
        'url': fields.String(description='The bookmark URL'),
        'tags': fields.List(fields.String, description='List of tags'),
        'topic': fields.Integer(description='Topic ID')
    })

    topic_in_tree = api.model('TopicInTree', {
        'id': fields.Integer(description='Topic ID'),
        'name': fields.String(description='Topic name'),
        'subtopics': fields.Nested('TopicInTree', description='List of subtopics'),
        'bookmarks': fields.List(fields.Nested(bookmark_in_tree), description='List of bookmarks in this topic'),
        'bookmark_count': fields.Integer(description='Number of bookmarks in this topic'),
        'subtopic_count': fields.Integer(description='Number of subtopics in this topic')
    })

    topic_tree = api.model('TopicTree', {
        'root': fields.Nested(topic_in_tree)
    })

    return bookmark_model, bookmark_response, hierarchical_topic_model, topic_tree