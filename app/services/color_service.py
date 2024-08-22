import colorsys

class ColorService:
    def __init__(self):
        self.color_map = {}
        self.golden_ratio_conjugate = 0.618033988749895

    def get_color(self, topic_id):
        if topic_id not in self.color_map:
            hue = (topic_id * self.golden_ratio_conjugate) % 1
            r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.5, 0.95)]
            self.color_map[topic_id] = f"#{r:02x}{g:02x}{b:02x}"
        return self.color_map[topic_id]

    def get_all_colors(self):
        return self.color_map

color_service = ColorService()