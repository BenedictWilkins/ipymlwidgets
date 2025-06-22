"""Image widget that displays PyTorch tensors using ipycanvas."""

import numpy as np
import torch
from typing import Optional
from traitlets import observe, Instance, Dict as TDict
import ipycanvas
import ipywidgets as W
import ipyevents as E
import torchvision.transforms.v2.functional as F

_NEAREST_STYLE = """
    <style>
    .nearest_interpolation {
        image-rendering: pixelated !important;
        image-rendering: crisp-edges !important;
    }
    </style>
    """
_DEFAULT_SIZE = 320


class Image(W.Box):
    """A widget that displays PyTorch tensors as images using ipycanvas."""

    image = Instance(torch.Tensor, allow_none=True).tag(sync=False)
    # when this value is set by the instance, any observers will be notified
    click = TDict(
        allow_none=False, default_value=dict(x=0, y=0, width=0, height=0)
    ).tag(sync=False)

    def __init__(
        self,
        image: Optional[torch.Tensor] = None,
        layers: int = 1,
        **kwargs,
    ):
        self._canvas = ipycanvas.MultiCanvas(
            n_canvases=layers,
            width=640,
            height=640,
            layout=W.Layout(
                width="100%",
                height="auto",
                border="1px solid black",
            ),
        )
        self._canvas.add_class("nearest_interpolation")
        # Set up ipyevents for the canvas
        self._click_event = E.Event(
            source=self._canvas,
            watched_events=["click"],
            prevent_default_action=True,
        )
        self._click_event.on_dom_event(self._on_canvas_click)

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

    @property
    def layers(self) -> int:
        return len(self._canvas._canvases)

    @property
    def size(self) -> tuple[int, int]:
        if self.image is None:
            return (0, 0)
        return self.image.shape[-1], self.image.shape[-2]

    def _on_canvas_click(self, event: dict):
        w_dom, h_dom = event["boundingRectWidth"], event["boundingRectHeight"]
        x_dom, y_dom = event["relativeX"], event["relativeY"]
        w_orig, h_orig = self.size
        x_orig = int(round(x_dom / w_dom * w_orig)) if w_orig > 0 else 0
        y_orig = int(round(y_dom / h_dom * h_orig)) if h_orig > 0 else 0
        # trigger callbacks via traitlets observe
        self.click = dict(x=x_orig, y=y_orig, width=w_orig, height=h_orig)

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

    def refresh(self, change: Optional[dict] = None):
        """Refresh the canvas."""
        layer = change.get("layer", 0)
        data, (x, y) = self._to_canvas_data(change)
        self._canvas[layer].put_image_data(data, x=x, y=y)

    def __repr__(self):
        return f"Image({[self.layers, *self.size]})"

    def __str__(self):
        return self.__repr__()
