from typing import Any, Sequence
import torch
import numpy as np
import PIL as pillow

import distinctipy

from traitlets import TraitType


def demo_image(
    width: int = 32,
    height: int = 32,
    square_size: int = 4,
    color1: tuple[int, int, int, int] = (255, 255, 255, 255),
    color2: tuple[int, int, int, int] = (0, 0, 0, 255),
) -> np.ndarray:
    """Create a checkerboard pattern image.

    Args:
        width (int): Image width in pixels. Defaults to 32.
        height (int): Image height in pixels. Defaults to 32.
        square_size (int): Size of each square in pixels. Defaults to 4.
        color1 (tuple[int, int, int, int]): RGBA color for first squares. Defaults to (255, 255, 255, 255).
        color2 (tuple[int, int, int, int]): RGBA color for second squares. Defaults to (0, 0, 0, 255).

    Returns:
        np.ndarray: RGBA image array with shape (height, width, 4).
    """
    image = np.zeros((height, width, 4), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Determine which square we're in
            square_x = x // square_size
            square_y = y // square_size

            # Alternate colors based on square position
            if (square_x + square_y) % 2 == 0:
                image[y, x] = color1
            else:
                image[y, x] = color2

    return image


class TensorTrait(TraitType):
    """A trait for torch tensors that supports custom validation."""

    info_text = "a torch.Tensor or None"

    def validate(self, obj, value):
        if value is None:
            return value
        if not isinstance(value, torch.Tensor):
            self.error(obj, value)
        return value


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
    elif isinstance(color, (torch.Tensor, np.ndarray)):
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


def resolve_colors(labels: torch.Tensor, cmap: torch.Tensor):
    """Resolve the colors for the classes.

    Args:
        labels (torch.Tensor[N]): Tensor containing label indicies.
        cmap (torch.Tensor[C,4]): Tensor of colors (RGBA).

    Returns:
        torch.Tensor[N,4]: Colors (RGBA) for each label in `labels`.
    """
    if cmap.ndim != 2:
        raise ValueError(
            f"Argument: `cmap` expected rgba colour tensor of shape [M,4] but got shape: {cmap.shape}"
        )
    if cmap.shape[-1] == 3:  # add alpha channel if it doesnt exist
        cmap = torch.cat([cmap, torch.ones_like(cmap[..., :1]) * 255], dim=-1)
    if cmap.shape[-1] != 4:
        raise ValueError(
            f"Argument: `cmap` expected rgba colour tensor of shape [M,4] but got shape: {cmap.shape}"
        )

    if labels.numel() == 0:  # no actual classes were provided
        return cmap[:1].expand(labels.shape[0], -1)
    else:
        return cmap[labels.long()]


def resolve_display_size(
    image_size: tuple[int, int], display_size: int | tuple[int, int]
):
    """Get the display size for the image.

    Args:
        image_size (tuple[int, int]): The size of the image.
        display_size (int | tuple[int, int]): The size of the display.

    Returns:
        tuple[int, int]: The size of the display.
    """
    if isinstance(display_size, int):
        scale = display_size / image_size[0]
        return (display_size, int(image_size[1] * scale))
    elif isinstance(display_size, (tuple, list)):
        return tuple(display_size)
    else:
        raise ValueError(
            f"Argument: `display_size` expected int or tuple[int,int] but got {type(display_size)}"
        )
