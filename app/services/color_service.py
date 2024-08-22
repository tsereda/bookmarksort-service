import colorsys
import random

class ColorService:
    def __init__(self):
        self.color_map = {}

    def get_color(self, topic_id):
        if topic_id not in self.color_map:
            # Generate a new color
            hue = random.random()
            saturation = 0.5 + random.random() * 0.5  # 0.5 to 1.0
            lightness = 0.4 + random.random() * 0.3   # 0.4 to 0.7
            r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(hue, lightness, saturation)]
            self.color_map[topic_id] = f"#{r:02x}{g:02x}{b:02x}"
        return self.color_map[topic_id]

    def get_all_colors(self):
        return self.color_map

color_service = ColorService()