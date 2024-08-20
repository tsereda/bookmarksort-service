from flask import request, jsonify
from flask_restx import Namespace, Resource, Api
from models import create_models
import asyncio

def setup_routes(api: Api, bookmark_organizer):
    # Define namespaces
    ns_bookmarks = Namespace('bookmarks', description='Bookmark operations')
    ns_topics = Namespace('topics', description='Topic operations')
    ns_visualization = Namespace('visualization', description='Visualization operations')
    ns_search = Namespace('search', description='Search operations')

    bookmark_model, bookmark_response, topic_model, scatter_plot_point, embedding_request = create_models(api)

    @ns_bookmarks.route('/')
    class BookmarkList(Resource):
        @ns_bookmarks.doc('list_bookmarks',
            description='Retrieve a list of all bookmarks with their associated topics.',
            responses={
                200: 'Success. Returns a list of all bookmarks.',
                500: 'Server error. An error occurred while fetching bookmarks.'
            })
        @ns_bookmarks.marshal_list_with(bookmark_response)
        def get(self):
            """List all bookmarks with their topics"""
            try:
                bookmarks = bookmark_organizer.get_bookmarks()
                if not bookmarks:
                    return [], 200
                return bookmarks, 200
            except Exception as e:
                ns_bookmarks.logger.error(f"Error fetching bookmarks: {str(e)}")
                return {'message': 'An error occurred while fetching bookmarks', 'error': str(e)}, 500

        @ns_bookmarks.doc('add_bookmark',
            description='Add a new bookmark to the database.',
            responses={
                201: 'Bookmark added successfully.',
                500: 'Server error. An error occurred while adding the bookmark.'
            })
        @ns_bookmarks.expect(bookmark_model)
        def post(self):
            """Add a new bookmark"""
            try:
                data = request.json
                result = bookmark_organizer.add_bookmark(data)
                return result, 201
            except Exception as e:
                return {'message': 'An error occurred while adding the bookmark', 'error': str(e)}, 500

    @ns_topics.route('/')
    class Topics(Resource):
        @ns_topics.doc('create_topics',
            description='Create topics for all bookmarks using the generated embeddings.',
            responses={
                200: 'Topics created successfully.',
                400: 'Bad request. Embeddings may not have been generated.',
                500: 'Server error. An error occurred while creating topics.'
            })
        def post(self):
            """Create topics for all bookmarks"""
            try:
                result = bookmark_organizer.create_topics()
                if "error" in result:
                    return result, 400
                return result, 200
            except Exception as e:
                return {'message': 'An error occurred while creating topics', 'error': str(e)}, 500

    @ns_topics.route('/<int:topic_id>')
    class Topic(Resource):
        @ns_topics.doc('get_topic',
            description='Get the representation of a specific topic.',
            params={'topic_id': 'The ID of the topic to retrieve'},
            responses={
                200: 'Success. Returns the topic representation.',
                404: 'Topic not found.',
                500: 'Server error. An error occurred while fetching the topic.'
            })
        @ns_topics.marshal_with(topic_model)
        def get(self, topic_id):
            """Get representation of a specific topic"""
            try:
                topic = bookmark_organizer.get_topic_representation(topic_id)
                if topic:
                    return topic, 200
                return {'message': 'Topic not found'}, 404
            except ValueError as e:
                ns_topics.logger.error(f"Error fetching topic: {str(e)}")
                return {'message': str(e)}, 400
            except Exception as e:
                ns_topics.logger.error(f"Error fetching topic: {str(e)}")
                return {'message': f'An error occurred while fetching the topic: {str(e)}'}, 500

    @ns_bookmarks.route('/batch_tag')
    class BatchTagBookmarks(Resource):
        @ns_bookmarks.doc('batch_tag_bookmarks',
            description='Tag all untagged bookmarks concurrently with progress updates.',
            responses={
                200: 'Batch tagging process completed successfully.',
                400: 'Bad request. Invalid parameters.',
                500: 'Server error. An error occurred while batch tagging bookmarks.'
            })
        def post(self):
            """Tag all untagged bookmarks concurrently with progress updates"""
            try:
                if request.is_json:
                    data = request.get_json()
                elif request.form:
                    data = request.form
                else:
                    data = {}

                max_concurrent = int(data.get('max_concurrent', 5))
                max_tags = int(data.get('max_tags', 10))
                batch_size = int(data.get('batch_size', 20))

                if max_concurrent <= 0 or max_tags <= 0 or batch_size <= 0:
                    return {'message': 'Invalid max_concurrent, max_tags, or batch_size. All must be positive integers.'}, 400

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(bookmark_organizer.batch_tag_all_untagged_bookmarks(max_concurrent, max_tags, batch_size))
                loop.close()

                return result, 200
            except ValueError as ve:
                return {'message': f'Invalid parameter: {str(ve)}'}, 400
            except Exception as e:
                return {'message': 'An error occurred while batch tagging bookmarks', 'error': str(e)}, 500

        @ns_visualization.route('/scatter_plot')
        class ScatterPlotVisualization(Resource):
            @ns_visualization.doc('get_scatter_plot_data',
                description='Get data for scatter plot visualization of bookmarks and their topics.',
                responses={
                    200: 'Success. Returns scatter plot data.',
                    400: 'Bad request. Topics may not have been created.',
                    500: 'Server error. An error occurred while fetching scatter plot data.'
                })
            def get(self):
                """Get data for scatter plot visualization"""
                try:
                    scatter_data = bookmark_organizer.get_scatter_plot_data()
                    return jsonify(scatter_data)
                except ValueError as e:
                    ns_visualization.logger.error(f"Error fetching scatter plot data: {str(e)}")
                    return {'message': str(e)}, 400
                except Exception as e:
                    ns_visualization.logger.error(f"Error fetching scatter plot data: {str(e)}")
                    return {'message': f'An error occurred while fetching scatter plot data: {str(e)}'}, 500

        @ns_topics.route('/tree')
        class TopicTree(Resource):
            @ns_topics.doc('get_topic_tree',
                description='Get the hierarchical topic tree structure.',
                responses={
                    200: 'Success. Returns the topic tree structure.',
                    400: 'Bad request. Topics may not have been created.',
                    500: 'Server error. An error occurred while fetching the topic tree.'
                })
            def get(self):
                """Get the topic tree structure"""
                try:
                    tree = bookmark_organizer.get_topic_tree()
                    return jsonify({"tree": tree})
                except ValueError as e:
                    return {'message': str(e)}, 400
                except Exception as e:
                    return {'message': 'An error occurred while fetching the topic tree', 'error': str(e)}, 500

        @ns_topics.route('/tree_json')
        class TopicTreeJSON(Resource):
            @ns_topics.doc('get_topic_tree_json',
                description='Get the hierarchical topic tree structure as JSON.',
                responses={
                    200: 'Success. Returns the topic tree structure as JSON.',
                    400: 'Bad request. Topics may not have been created.',
                    500: 'Server error. An error occurred while fetching the topic tree JSON.'
                })
            def get(self):
                """Get the topic tree structure as JSON"""
                try:
                    tree_json = bookmark_organizer.get_tree_json()
                    return jsonify(tree_json)
                except ValueError as e:
                    return {'message': str(e)}, 400
                except Exception as e:
                    return {'message': 'An error occurred while fetching the topic tree JSON', 'error': str(e)}, 500

    @ns_visualization.route('/sunburst')
    class SunburstVisualization(Resource):
        @ns_visualization.doc('get_sunburst_data',
            description='Get data for sunburst visualization of topics and their hierarchies.',
            responses={
                200: 'Success. Returns sunburst data.',
                400: 'Bad request. Topics may not have been created.',
                500: 'Server error. An error occurred while fetching sunburst data.'
            })
        def get(self):
            """Get data for sunburst visualization"""
            try:
                sunburst_data = bookmark_organizer.get_sunburst_data()
                return jsonify(sunburst_data)
            except ValueError as e:
                return {'message': str(e)}, 400
            except Exception as e:
                return {'message': 'An error occurred while fetching sunburst data', 'error': str(e)}, 500

    @ns_search.route('/')
    class Search(Resource):
        @ns_search.doc('search_bookmarks',
            description='Search bookmarks based on a query string.',
            params={'q': 'The search query string'},
            responses={
                200: 'Success. Returns search results.',
                500: 'Server error. An error occurred while searching bookmarks.'
            })
        def get(self):
            """Search bookmarks"""
            try:
                query = request.args.get('q', '')
                results = bookmark_organizer.search_bookmarks(query)
                return jsonify(results)
            except Exception as e:
                return {'message': 'An error occurred while searching bookmarks', 'error': str(e)}, 500

    # Add namespaces to the API
    api.add_namespace(ns_bookmarks)
    api.add_namespace(ns_topics)
    api.add_namespace(ns_visualization)
    api.add_namespace(ns_search)