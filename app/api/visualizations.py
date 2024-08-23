from flask import jsonify
from flask_restx import Namespace, Resource
from ..models.api_models import create_models
import logging

visualizations_ns = Namespace('visualizations', description='Visualization operations')

_, _, _, scatter_plot_point, _ = create_models(visualizations_ns)
logger = logging.getLogger(__name__)

@visualizations_ns.route('/scatter_plot')
class ScatterPlotVisualization(Resource):
    @visualizations_ns.doc('get_scatter_plot_data',
        description='Get data for scatter plot visualization of bookmarks and their topics.',
        responses={
            200: 'Success. Returns scatter plot data.',
            400: 'Bad request. Topics may not have been created.',
            500: 'Server error. An error occurred while fetching scatter plot data.'
        })
    @visualizations_ns.marshal_list_with(scatter_plot_point)
    def get(self):
        """Get data for scatter plot visualization"""
        try:
            scatter_data = visualizations_ns.visualization_service.get_scatter_plot_data()
            return scatter_data, 200
        except ValueError as e:
            logger.error(f"Error fetching scatter plot data: {str(e)}")
            return {'message': str(e)}, 400
        except Exception as e:
            logger.error(f"Unexpected error in get_scatter_plot_data: {str(e)}", exc_info=True)
            return {'message': f'An error occurred while fetching scatter plot data: {str(e)}'}, 500

@visualizations_ns.route('/sunburst')
class SunburstVisualization(Resource):
    @visualizations_ns.doc('get_sunburst_data',
        description='Get data for sunburst visualization of topics and their hierarchies.',
        responses={
            200: 'Success. Returns sunburst data.',
            400: 'Bad request. Topics may not have been created.',
            500: 'Server error. An error occurred while fetching sunburst data.'
        })
    def get(self):
        """Get data for sunburst visualization"""
        try:
            logger.info("Attempting to fetch sunburst data")
            sunburst_data = visualizations_ns.visualization_service.get_sunburst_data()
            logger.info("Successfully fetched sunburst data")
            return jsonify(sunburst_data)
        except ValueError as e:
            logger.error(f"ValueError in get_sunburst_data: {str(e)}")
            return {'message': str(e)}, 400
        except Exception as e:
            logger.error(f"Unexpected error in get_sunburst_data: {str(e)}", exc_info=True)
            return {'message': 'An error occurred while fetching sunburst data', 'error': str(e)}, 500