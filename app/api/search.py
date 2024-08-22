# app/api/search.py

from flask import request, jsonify
from flask_restx import Namespace, Resource

search_ns = Namespace('search', description='Search operations')

@search_ns.route('/')
class Search(Resource):
    @search_ns.doc('search_bookmarks',
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
            results = search_ns.bookmark_service.search_bookmarks(query)
            return jsonify(results)
        except Exception as e:
            return {'message': 'An error occurred while searching bookmarks', 'error': str(e)}, 500

# Don't forget to update __init__.py to include this new namespace