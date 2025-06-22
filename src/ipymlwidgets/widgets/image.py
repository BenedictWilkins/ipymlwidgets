"""Image widget that displays PyTorch tensors using ipycanvas."""

import numpy as np
import torch
from typing import Optional, Any
from traitlets import observe, Instance
import ipycanvas
import ipywidgets as W
import torchvision.transforms.v2.functional as F
from PIL import Image as PILImage
from IPython.display import HTML, display


_NEAREST_STYLE = """
    <style>
    .nearest_interpolation {
        image-rendering: pixelated !important;
        image-rendering: crisp-edges !important;
    }
    </style>
    """
_DEFAULT_SIZE = 320


class Image(W.HBox):
    """A widget that displays PyTorch tensors as images using ipycanvas."""

    image = Instance(torch.Tensor, allow_none=True).tag(sync=False)

    def __init__(
        self,
        image: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self._canvas = ipycanvas.Canvas(
            width=640,
            height=640,
            layout=W.Layout(
                width="100%",
                height="auto",
                border="1px solid black",
            ),
        )
        self._canvas.add_class("nearest_interpolation")
        super().__init__(
            [W.HTML(_NEAREST_STYLE), self._canvas],
            layout=W.Layout(
                background="red",
                width=f"{_DEFAULT_SIZE}px",
                height="auto",  # let height follow aspect
                aspect_ratio="auto",  # updated dynamically
                overflow="hidden",
                position="relative",
            ),
            **kwargs,
        )
        if image is not None:
            self.image = image

    def __getitem__(self, index):
        """Access pixel data directly from the underlying image tensor."""
        return self.image[index]

    def __setitem__(self, index, value):
        """Set pixel(s) in-place and refresh display directly."""
        self.image[index] = value

    @observe("image")
    def _on_image_change(self, change):
        """Handle image changes by updating the canvas."""
        _, h, w = change["new"].shape
        # Set logical resolution of canvas (important for pixelated scaling)
        self._canvas.width = w
        self._canvas.height = h
        # Dynamically update container's aspect ratio
        self.layout.aspect_ratio = f"{h} / {w}"
        self.refresh(change)

    def _to_canvas_data(
        self,
        change: Optional[dict] = None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Convert the tensor to canvas data - set directly with `self._canvas.set_image_data`"""
        if change is None:
            return np.array(F.to_pil_image(self.image)), (0, 0)
        else:
            return np.array(F.to_pil_image(change["new"])), (0, 0)
        # _, h, w = self.image.shape

        # canvas_w, canvas_h = self._canvas.width, self._canvas.height
        # # Fit to canvas while maintaining aspect ratio
        # scale = min(canvas_w / w, canvas_h / h)
        # size = (int(w * scale), int(h * scale))

        # if change is None:
        #     return np.array(F.to_pil_image(self.image)), (0, 0)
        # else:
        #     # the position and the size are in the original `image` coordinate space.
        #     _, h, w = self.image.shape
        #     (x1, y1, x2, y2) = change.get("box", (0, 0, w, h))
        #     x = int(x1 * self._canvas.width / w)
        #     y = int(y1 * self._canvas.height / h)
        #     w = int((x2 - x1) * self._canvas.width / w)
        #     h = int((y2 - y1) * self._canvas.height / h)
        #     return (
        #         np.array(
        #             F.to_pil_image(change["new"]).resize(
        #                 (w, h),
        #                 mode,
        #             )
        #         ),
        #         (x, y),
        #     )

    def refresh(self, change: Optional[dict] = None):
        """Refresh the canvas."""
        data, (x, y) = self._to_canvas_data(change)
        self._canvas.put_image_data(data, x=x, y=y)
