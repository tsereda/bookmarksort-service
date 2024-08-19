from flask_restx import fields, Model

def create_models(api):
    bookmark_model = api.model('Bookmark', {
        'title': fields.String(required=True, description='The bookmark title'),
        'url': fields.String(required=True, description='The bookmark URL'),
        'tags': fields.List(fields.String, description='List of tags'),
        'topic': fields.Integer(description='Topic ID', allow_null=True)
    })

    bookmark_response = api.inherit('BookmarkResponse', bookmark_model, {
        'id': fields.Integer(description='The bookmark ID')
    })

    topic_word_model = api.model('TopicWord', {
        'word': fields.String(description='Word in the topic'),
        'score': fields.Float(description='Score of the word in the topic')
    })

    topic_model = api.model('Topic', {
        'id': fields.Integer(description='Topic ID'),
        'name': fields.String(description='Topic name'),
        'count': fields.Integer(description='Number of documents in the topic'),
        'representation': fields.List(fields.Nested(topic_word_model), description='Top words and their scores for this topic')
    })

    scatter_plot_point = api.model('ScatterPlotPoint', {
        'id': fields.Integer(description='Bookmark ID'),
        'x': fields.Float(description='X coordinate'),
        'y': fields.Float(description='Y coordinate'),
        'topic': fields.Integer(description='Topic ID')
    })

    embedding_request = api.model('EmbeddingRequest', {
        'embedding_model': fields.String(description='Name of the embedding model to use',
                                         example='all-MiniLM-L6-v2')
    })

    return bookmark_model, bookmark_response, topic_model, scatter_plot_point, embedding_request