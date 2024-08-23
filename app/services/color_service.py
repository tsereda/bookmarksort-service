import colorsys
from typing import Dict, Optional
import math

class ColorService:
    def __init__(self):
        self.color_map: Dict[int, str] = {}
        self.golden_ratio_conjugate = 0.618033988749895

    def get_color(self, topic_id: int, depth: int = 0, angle: Optional[float] = None) -> str:
        if topic_id in self.color_map:
            return self.color_map[topic_id]

        if angle is not None:
            # Use angle for hierarchical visualizations (e.g., sunburst)
            hue = (angle / (2 * math.pi)) % 1.0
        else:
            # Use topic_id for non-hierarchical visualizations (e.g., scatter plot)
            hue = (topic_id * self.golden_ratio_conjugate) % 1.0

        # Adjust saturation and lightness based on depth
        saturation = max(0.5, 1 - (depth * 0.1))
        lightness = min(0.6, 0.4 + (depth * 0.1))

        color = self._hsl_to_hex(hue, saturation, lightness)
        self.color_map[topic_id] = color
        return color

    @staticmethod
    def _hsl_to_hex(h: float, s: float, l: float) -> str:
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))

color_service = ColorService()