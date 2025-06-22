"""Basic swatch widget that displays a single color."""

from datautils.types import Color
from datautils.types.convert import color_to_tuple, color_to_hex, color_with_alpha

import ipywidgets as W
from io import BytesIO
from PIL import Image as PILImage
from traitlets import Any, observe, validate, TraitError

class Swatch(W.Image):
    color = Any(default_value=(0, 0, 0, 255), help="(R, G, B, A) color").tag(sync=True)

    def __init__(self, color : Color, size : tuple[int, int] = (20, 20)):
        color = color_with_alpha(color_to_tuple(color), 255)
        super().__init__(
        value=Swatch._bytes(color),
        format="png",
        layout=W.Layout(
            width=f"{size[0]}px", height="{size[1]}px",
            border="1px solid #888888",
            object_fit="fill" # stretch the 1×1 to fill the box
        ))
        self.color = color  # triggers initial .value

    @observe("color")
    def _value_from_color(self, change):
        """Update .value (PNG) when .color changes."""
        self.value = self._bytes(change["new"])
    
    @validate("color")
    def _validate_color(self, proposal):
        """Convert color to RGBA."""
        try:
            return color_with_alpha(color_to_tuple(proposal['value']), 255)
        except Exception as e:
            raise TraitError(f"Invalid colour: {proposal['value']}") from e
    
    @classmethod
    def _bytes(cls, color : tuple[int,int,int,int]):
        img = PILImage.new("RGBA", (1, 1), color)
        buf = BytesIO()
        img.save(buf, format="png")
        return buf.getvalue()
