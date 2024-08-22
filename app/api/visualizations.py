from flask import jsonify
from flask_restx import Namespace, Resource
from ..models.api_models import create_models

visualizations_ns = Namespace('visualizations', description='Visualization operations')

_, _, _, scatter_plot_point, _ = create_models(visualizations_ns)

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
            visualizations_ns.logger.error(f"Error fetching scatter plot data: {str(e)}")
            return {'message': str(e)}, 400
        except Exception as e:
            visualizations_ns.logger.error(f"Error fetching scatter plot data: {str(e)}")
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
            sunburst_data = visualizations_ns.visualization_service.get_sunburst_data()
            if "error" in sunburst_data:
                return {'message': sunburst_data["error"]}, 400
            return jsonify(sunburst_data)
        except Exception as e:
            return {'message': 'An error occurred while fetching sunburst data', 'error': str(e)}, 500