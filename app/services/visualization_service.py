from .color_service import color_service

class VisualizationService:
    def __init__(self, topic_service, bookmark_service):
        self.topic_service = topic_service
        self.bookmark_service = bookmark_service

    def get_scatter_plot_data(self):
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

    def get_sunburst_data(self):
        if self.topic_service.hierarchical_topics is None:
            raise ValueError("Hierarchical topics have not been created. Call create_topics() first.")
        
        tree_json = self.topic_service.get_tree_json()
        topic_names = self.topic_service.get_topic_names()
        
        def add_values_and_colors(node, topic_id=0):
            if not node.get("children"):
                node["value"] = 1
                node["color"] = color_service.get_color(topic_id)
            else:
                for i, child in enumerate(node["children"]):
                    child_id = topic_id * 10 + i + 1
                    add_values_and_colors(child, child_id)
                node["value"] = sum(child["value"] for child in node["children"])
            
            if "name" in node:
                node["name"] = topic_names.get(topic_id, node["name"])
        
        add_values_and_colors(tree_json)
        return tree_json