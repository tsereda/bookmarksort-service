import colorsys
from typing import Dict, Tuple
import math

class ColorService:
    def __init__(self):
        self.color_map: Dict[int, str] = {}
        self.golden_ratio_conjugate = 0.618033988749895

    def get_color(self, identifier: int, angle: float = None, depth: int = 0) -> str:
        if identifier == 0:  # Root node
            return "rgba(255, 255, 255, 0)"  # Transparent

        if angle is not None:
            # Sunburst visualization
            hue = angle / (2 * math.pi)
        else:
            # Scatter plot
            hue = (identifier * self.golden_ratio_conjugate) % 1.0

        saturation = 0.7  # Fixed saturation for more vibrant colors
        lightness = max(0.3, min(0.7, 0.5 + depth * 0.05))  # Adjust lightness based on depth

        return self._hsl_to_hex(hue, saturation, lightness)

    @staticmethod
    def _hsl_to_hex(h: float, s: float, l: float) -> str:
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))

color_service = ColorService()