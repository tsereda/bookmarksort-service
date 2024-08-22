from flask import jsonify
from flask_restx import Namespace, Resource
from ..models.api_models import create_models
import asyncio

topics_ns = Namespace('topics', description='Topic operations')

_, _, topic_model, _, _ = create_models(topics_ns)

@topics_ns.route('/')
class Topics(Resource):
    @topics_ns.doc('create_topics',
        description='Create topics for all bookmarks using the generated embeddings.',
        responses={
            200: 'Topics created successfully.',
            400: 'Bad request. Embeddings may not have been generated.',
            500: 'Server error. An error occurred while creating topics.'
        })
    def post(self):
        """Create topics for all bookmarks"""
        try:
            result = topics_ns.topic_service.create_topics()
            if "error" in result:
                return result, 400
            return result, 200
        except Exception as e:
            return {'message': 'An error occurred while creating topics', 'error': str(e)}, 500

@topics_ns.route('/<int:topic_id>')
class Topic(Resource):
    @topics_ns.doc('get_topic',
        description='Get the representation of a specific topic.',
        params={'topic_id': 'The ID of the topic to retrieve'},
        responses={
            200: 'Success. Returns the topic representation.',
            404: 'Topic not found.',
            500: 'Server error. An error occurred while fetching the topic.'
        })
    @topics_ns.marshal_with(topic_model)
    def get(self, topic_id):
        """Get representation of a specific topic"""
        try:
            topic = topics_ns.topic_service.get_topic_representation(topic_id)
            if topic:
                return topic, 200
            return {'message': 'Topic not found'}, 404
        except ValueError as e:
            topics_ns.logger.error(f"Error fetching topic: {str(e)}")
            return {'message': str(e)}, 400
        except Exception as e:
            topics_ns.logger.error(f"Error fetching topic: {str(e)}")
            return {'message': f'An error occurred while fetching the topic: {str(e)}'}, 500

@topics_ns.route('/tree')
class TopicTree(Resource):
    @topics_ns.doc('get_topic_tree',
        description='Get the hierarchical topic tree structure.',
        responses={
            200: 'Success. Returns the topic tree structure.',
            400: 'Bad request. Topics may not have been created.',
            500: 'Server error. An error occurred while fetching the topic tree.'
        })
    def get(self):
        """Get the topic tree structure"""
        try:
            tree = topics_ns.topic_service.get_topic_tree()
            return jsonify({"tree": tree})
        except ValueError as e:
            return {'message': str(e)}, 400
        except Exception as e:
            return {'message': 'An error occurred while fetching the topic tree', 'error': str(e)}, 500

@topics_ns.route('/tree_json')
class TopicTreeJSON(Resource):
    @topics_ns.doc('get_topic_tree_json',
        description='Get the hierarchical topic tree structure as JSON.',
        responses={
            200: 'Success. Returns the topic tree structure as JSON.',
            400: 'Bad request. Topics may not have been created.',
            500: 'Server error. An error occurred while fetching the topic tree JSON.'
        })
    def get(self):
        """Get the topic tree structure as JSON"""
        try:
            tree_json = topics_ns.topic_service.get_tree_json()
            return jsonify(tree_json)
        except ValueError as e:
            return {'message': str(e)}, 400
        except Exception as e:
            return {'message': 'An error occurred while fetching the topic tree JSON', 'error': str(e)}, 500

@topics_ns.route('/regenerate-names')
class RegenerateTopicNames(Resource):
    @topics_ns.doc('regenerate_topic_names',
        description='Regenerate names for all topics based on their hierarchical structure.',
        responses={
            200: 'Success. Returns the updated topic tree.',
            500: 'Server error. An error occurred while regenerating topic names.'
        })
    def post(self):
        """Regenerate names for all topics"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(topics_ns.topic_service.regenerate_topic_names())
            loop.close()
            return result, 200
        except Exception as e:
            return {'message': 'An error occurred while regenerating topic names', 'error': str(e)}, 500