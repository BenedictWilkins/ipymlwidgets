"""Module containing some useful utility types and type hints.

See:
- `datautils.types.Namespace`
- `datautils.types.hints` - for various useful type hints/aliases
"""

from .namespace import Namespace
from .hints import Device, Image, ImageBatch, ImageMode, Color, Size

__all__ = (
    "Namespace",
    "Device",
    "Image",
    "ImageBatch",
    "ImageMode",
    "Size",
    "Color",
)
