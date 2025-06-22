"""Utilities for standardizing and converting between types (see `datautils.types.hints`).

Typically this means converting to torch.Tensor.
"""

from collections.abc import Sequence
from typing import Optional
from functools import lru_cache

import pathlib
import PIL as pillow
import PIL.Image as _PILImage
import torch

import torchvision.transforms.v2 as T

from datautils.types.hints import Color, Image, ImageMode

# =================================================================================== #
# ================================ Image Conversions ================================ #
# =================================================================================== #

# some image tranforms that may be reused
_TENSOR_TO_PIL = T.ToPILImage()
_IMAGE_INT_TO_FLOAT = T.ToDtype(torch.float32, scale=True)
_PIL_TO_TENSOR = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
)

def image_to_pil(image: Image) -> pillow.Image.Image:
    """Convert an `Image` to a PIL image.
    
    See `datautils.types.Image` for details on the `image` type.

    Args:
        image (torch.Tensor): The image to convert to PIL.

    Returns:
        pillow.Image.Image: The converted image.
    """
    if isinstance(image, str):
        return _PILImage.open(pathlib.Path(image).expanduser().resolve().as_posix())
    elif isinstance(image, pathlib.Path):
        return _PILImage.open(image.expanduser().resolve().as_posix())
    elif isinstance(image, torch.Tensor):
        # squeeze the initial dims up to [CHW]
        if image.ndim > 3 and image.shape[0] == 1:
            return image # batch of 1 is ok, try again
        return _TENSOR_TO_PIL(
            image
        )  # let torchvision handle the conversion + validation
    elif isinstance(image, _PILImage.Image):
        return image
    else:
        raise TypeError(
            f"Argument: `image` expected type `str`, `pathlib.Path`, `torch.Tensor` or `pillow.Image.Image` but got {type(image)}"
        )

def image_to_tensor(image: Image, mode: Optional[str] = None) -> torch.Tensor:
    """Convert an image that is in another format (e.g. pillow, path or URL) to a torch.Tensor.

    Args:
        image (Image): image to convert to tensor.
        mode (str, optional): image mode (L, LA, RGB, RGBA).

    Returns:
        torch.Tensor: image tensor in CHW float32 format.
    """
    if isinstance(image, str):
        image = _PILImage.open(image)
        if mode:
            image = image.convert(mode)
        return _PIL_TO_TENSOR(image)
    elif isinstance(image, pathlib.Path):
        image = _PILImage.open(image.expanduser().resolve().as_posix())
        if mode:
            image = image.convert(mode)
        return _PIL_TO_TENSOR(image)
    elif isinstance(image, torch.Tensor):
        # squeeze the initial dims up to [CHW]
        if image.ndim > 3 and image.shape[0] == 1:
            return image_to_tensor(image)  # batch of 1 is ok, try again
        elif image.ndim == 2:
            image = image.unsqueeze(0)  # [1HW] assume grey scale
        # validate the input tensor
        # TODO what about the `mode` argument? should it convert an image tensor as well?
        if image.shape[0] > 4: # not an image tensor?
            raise ValueError(
                f"Argument: `image` expected channel dimension [0-4] but got {image.shape[0]}"
            )
        if torch.is_floating_point(image):
            _min, _max = image.min(), image.max()
            if _min < 0.0 or _max > 1.0:
                # TODO we chould check against a small tolerance
                # sometimes tensors go slighly out of range, we should allow this but clamp them.
                raise ValueError(
                    f"Argument: `image` expected range [0-1] but got [{image.min():.2f}-{image.max():.2f}]"
                )
        else:
            _min, _max = image.min(), image.max()
            if _min < 0 or _max > 255:
                image = _IMAGE_INT_TO_FLOAT(image)
        return image
    elif isinstance(image, _PILImage.Image):
        if mode:
            image = image.convert(mode)
        return _PIL_TO_TENSOR(image)
    else:
        raise TypeError(
            f"Argument: `image` expected type `str`, `pathlib.Path`, `torch.Tensor` or `pillow.Image.Image` but got {type(image)}"
        )

# =================================================================================== #
# ================================ Color Conversions ================================ #
# =================================================================================== #

def color_with_alpha(color : Color, alpha : int = 255) -> tuple[int,int,int,int]:
    color = color_to_tuple(color)
    if len(color) == 3:
        return color + (alpha,)
    else:
        return color

def color_to_tuple(color : Color) -> tuple[int,...]:
    """Convert color to a int tuple with range [0-255]."""
    if isinstance(color, str):
        return pillow.ImageColor.getrgb(color)
    elif isinstance(color, Sequence) and len(color) in (3,4):
        if all(isinstance(x, int) for x in color):
            assert all(x <= 255 and x >= 0 for x in color), f"Color has bad range [{min(color)}-{max(color)}]"
            return tuple(color)
        elif all(isinstance(x, float) for x in color): # assume [0-1]
            assert all(x <= 1.0 and x >= 0.0 for x in color), f"Color has bad range [{min(color)}-{max(color)}]"
            return tuple(int(x * 255) for x in color)
    raise ValueError(f"Argument: `color` is not a valid Color got {color}")
    
def color_to_hex(color : Color) -> str:
    """Convert color to a rgb(a) hex string"""
    color = color_to_tuple(color)
    if len(color) == 3:
        return "#{:02X}{:02X}{:02X}".format(*color)
    else:
        return "#{:02X}{:02X}{:02X}{:02X}".format(*color)

def color_to_tensor(color : Color) -> torch.Tensor:
    """Convert the `color` to a color tensor with range [0-1]"""
    return torch.tensor(color_to_tuple(color)) / 255.0
