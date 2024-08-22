from .color_service import ColorService

class VisualizationService:
    def __init__(self, topic_service, bookmark_service):
        self.topic_service = topic_service
        self.bookmark_service = bookmark_service
        self.color_service = ColorService()

    def get_scatter_plot_data(self):
        bookmarks = self.bookmark_service.get_bookmarks()
        embeddings = self.topic_service.get_embeddings()
        reduced_embeddings = self.topic_service.reduce_embeddings(embeddings)

        scatter_data = []
        for i, bookmark in enumerate(bookmarks):
            scatter_data.append({
                'id': bookmark['id'],
                'x': float(reduced_embeddings[i, 0]),
                'y': float(reduced_embeddings[i, 1]),
                'topic': bookmark['topic'],
                'title': bookmark['title'],
                'url': bookmark['url'],
                'tags': bookmark['tags'],
                'color': self.color_service.get_color(bookmark['topic'])
            })
        return scatter_data

    def get_sunburst_data(self):
        if self.topic_service.hierarchical_topics is None:
            raise ValueError("Hierarchical topics have not been created. Call create_topics() first.")
        
        tree_json = self.topic_service.get_tree_json()
        
        def add_values(node):
            if not node["children"]:
                node["value"] = 1
            else:
                for child in node["children"]:
                    add_values(child)
                node["value"] = sum(child["value"] for child in node["children"])
        
        add_values(tree_json)
        return tree_json