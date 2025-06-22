from typing import Any, Sequence
import torch
import numpy
import PIL as pillow


def color_to_tuple(color: Any) -> tuple[int, ...]:
    """Convert color to a int tuple with range [0-255]."""
    if isinstance(color, str):
        return pillow.ImageColor.getrgb(color)
    elif isinstance(color, (torch.Tensor, numpy.ndarray)):
        return color_to_tuple(color.tolist())
    elif isinstance(color, Sequence) and len(color) in (3, 4):
        if all(isinstance(x, int) for x in color):
            assert all(
                x <= 255 and x >= 0 for x in color
            ), f"Color has bad range [{min(color)}-{max(color)}]"
            return tuple(color)
        elif all(isinstance(x, float) for x in color):  # assume [0-1]
            assert all(
                x <= 1.0 and x >= 0.0 for x in color
            ), f"Color has bad range [{min(color)}-{max(color)}]"
            return tuple(int(x * 255) for x in color)
    raise ValueError(f"Argument: `color` is not a valid Color got {color}")


def color_to_hex(color: Any) -> str:
    """Convert color to a rgb(a) hex string"""
    color = color_to_tuple(color)
    if len(color) == 3:
        return "#{:02X}{:02X}{:02X}".format(*color)
    else:
        return "#{:02X}{:02X}{:02X}{:02X}".format(*color)
