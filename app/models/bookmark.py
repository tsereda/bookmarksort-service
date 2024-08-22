class Bookmark:
    def __init__(self, id, title, url, tags=None, topic=None):
        self.id = id
        self.title = title
        self.url = url
        self.tags = tags or []
        self.topic = topic