from typing import Any, Sequence
import torch
import numpy
import PIL as pillow

import distinctipy


def get_colors(
    n_colors: int, pastel_factor: float = 0.7, alpha: float = 1.0
) -> torch.Tensor:
    """
    Generate n distinct colors suitable for class visualization.

    Args:
        n_colors (int): Number of distinct colors to generate.
        pastel_factor (float): How pastel the colors should be (0-1). Defaults to 0.7.
        alpha (float): Alpha value for the colors. Defaults to 1.0.

    Returns:
        torch.Tensor: Tensor of shape (n_colors, 4) with RGBA values.
    """
    colors = distinctipy.get_colors(n_colors, pastel_factor=pastel_factor)
    return (
        torch.tensor(
            [(r, g, b, alpha) for r, g, b in colors],
            dtype=torch.float32,
        )
        * 255.0
    ).type(torch.uint8)


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
