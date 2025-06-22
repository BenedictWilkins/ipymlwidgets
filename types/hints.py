"""A bunch of useful type hints."""

from typing import TypeAlias, Union, Sequence, Literal

import pathlib
import torch
import PIL.Image as _PILImage

# type alias (type hint) for torch device
Device: TypeAlias = Union[str, torch.device]
# type alias (type hint) for a single Image
Image: TypeAlias = Union[
    str, pathlib.Path, torch.Tensor, _PILImage.Image
]  # single image
# type alias (type hint) for a batch of images
ImageBatch: TypeAlias = Union[Image, list[Image]]
# type alias (type hint) for image size(s) typically (width, height)
Size: TypeAlias = Union[
    tuple[int, int],  # is [2]
    list[int],  # must be [2]
    list[list[int]],  # must be [N,2]
    list[tuple[int, int]],  # must be [N, 2]
    torch.Tensor,  # must be [..., 2]
]

Color : TypeAlias = Union[
    str,
    Sequence[int],
    Sequence[float],
]

ImageMode : TypeAlias = Literal["L", "LA", "RGB", "RGBA"]

__all__ = ("Device", "Image", "ImageBatch", "Size", "Color", "ImageMode")
