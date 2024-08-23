from typing import Dict, List, Any
from .color_service import color_service
import logging
import math

class VisualizationService:
    def __init__(self, topic_service: Any, bookmark_service: Any):
        self.topic_service = topic_service
        self.bookmark_service = bookmark_service
        self.logger = logging.getLogger(__name__)

    def get_scatter_plot_data(self) -> List[Dict[str, Any]]:
        bookmarks = self.bookmark_service.get_bookmarks()
        embeddings = self.topic_service.get_embeddings()
        reduced_embeddings = self.topic_service.reduce_embeddings(embeddings)
        topic_names = self.topic_service.get_topic_names()

        scatter_data = []
        for i, bookmark in enumerate(bookmarks):
            topic_id = bookmark['topic']
            scatter_data.append({
                'id': bookmark['id'],
                'x': float(reduced_embeddings[i, 0]),
                'y': float(reduced_embeddings[i, 1]),
                'topic': topic_id,
                'topicName': topic_names.get(topic_id, f"Topic {topic_id}"),
                'title': bookmark['title'],
                'url': bookmark['url'],
                'tags': bookmark['tags'],
                'color': color_service.get_color(topic_id)
            })
        return scatter_data

    def get_sunburst_data(self) -> Dict[str, Any]:
        if self.topic_service.hierarchical_topics is None:
            raise ValueError("Hierarchical topics have not been created. Call create_topics() first.")

        tree_json = self.topic_service.get_tree_json()
        topic_names = self.topic_service.get_topic_names()

        root_node = {
            "name": "Bookmarks",
            "children": tree_json.get("children", [])
        }

        def add_values_and_colors(node: Dict[str, Any], topic_id: int = 0, depth: int = 0, start_angle: float = 0, end_angle: float = 2 * math.pi) -> None:
            if not node.get("children"):
                node["value"] = 1
            else:
                total_value = sum(child.get("value", 0) for child in node["children"])
                if total_value > 0:
                    current_angle = start_angle
                    for i, child in enumerate(node["children"]):
                        child_id = (topic_id * 10 + i + 1) if topic_id != 0 else i + 1
                        child_value = child.get("value", 0)
                        child_angle = (end_angle - start_angle) * (child_value / total_value)
                        add_values_and_colors(child, child_id, depth + 1, current_angle, current_angle + child_angle)
                        current_angle += child_angle
                else:
                    angle_step = (end_angle - start_angle) / len(node["children"])
                    for i, child in enumerate(node["children"]):
                        child_id = (topic_id * 10 + i + 1) if topic_id != 0 else i + 1
                        child_start = start_angle + i * angle_step
                        child_end = child_start + angle_step
                        add_values_and_colors(child, child_id, depth + 1, child_start, child_end)
                node["value"] = max(total_value, 1)  # Ensure the value is at least 1

            mid_angle = (start_angle + end_angle) / 2
            node["color"] = color_service.get_color(topic_id, depth, mid_angle)
            if "name" in node and depth > 0:
                node["name"] = topic_names.get(topic_id, node["name"])

        add_values_and_colors(root_node)
        return root_node