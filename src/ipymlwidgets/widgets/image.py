"""Image widget that displays PyTorch tensors using ipycanvas."""

import numpy as np
from typing import Optional, Callable
from traitlets import observe, Instance, Dict as TDict
import ipycanvas
import ipywidgets as W
import ipyevents as E
import torchvision.transforms.v2.functional as F
from ipymlwidgets.traits import Tensor, SupportedTensor

_NEAREST_STYLE = """
    <style>
    .nearest_interpolation {
        image-rendering: pixelated !important;
        image-rendering: crisp-edges !important;
    }
    </style>
"""
DRAG_THRESHOLD = 3  # client space threshold for drag start

_STYLE_HTML_LAYOUT = W.Layout(
    width="0px", height="0px", margin="0px", padding="0px", border="none"
)


class Image(W.Box):
    """A widget that displays PyTorch tensors as images using ipycanvas."""

    # always convert the incoming tensor value to numpy
    image = Tensor(allow_none=True).tag(sync=False)
    # events
    mouse_click = TDict(allow_none=False, default_value=dict()).tag(sync=False)
    mouse_drag = TDict(allow_none=False, default_value=dict()).tag(sync=False)
    mouse_move = TDict(allow_none=False, default_value=dict()).tag(sync=False)
    mouse_down = TDict(allow_none=False, default_value=dict()).tag(sync=False)
    mouse_up = TDict(allow_none=False, default_value=dict()).tag(sync=False)

    def __init__(
        self,
        image: Optional[SupportedTensor] = None,
        layers: int = 1,
        **kwargs,
    ):

        self._canvas = ipycanvas.MultiCanvas(
            n_canvases=layers,
            width=1,
            height=1,
            layout=W.Layout(
                width="100%",
                height="auto",
                border="1px solid black",
                # visibility="visible",
            ),
        )
        self._canvas.add_class("nearest_interpolation")

        self._mouse_down_event = E.Event(
            source=self._canvas,
            watched_events=["mousedown"],
            prevent_default_action=True,
        )
        self._mouse_down_event.on_dom_event(self._on_canvas_mousedown)

        self._mouse_up_event = E.Event(
            source=self._canvas,
            watched_events=["mouseup"],
            prevent_default_action=True,
        )
        self._mouse_up_event.on_dom_event(self._on_canvas_mouseup)

        # Add drag event handling
        self._mouse_move_event = E.Event(
            source=self._canvas,
            watched_events=["mousemove"],
            prevent_default_action=True,
        )
        self._mouse_move_event.on_dom_event(self._on_canvas_mousemove)

        # event state - internal use to handle combined events like drag, click, etc.
        self._mouse_down_position = None  # used by other mouse event handlers

        super().__init__(
            [W.HTML(_NEAREST_STYLE, layout=_STYLE_HTML_LAYOUT), self._canvas],
            layout=W.Layout(
                width="100%",
                height="auto",  # let height follow aspect
                aspect_ratio="auto",  # updated dynamically
                overflow="hidden",
                position="relative",
            ),
            **kwargs,
        )
        self.image = image

    def hide(self):
        self._canvas.layout.visibility = "hidden"

    def show(self):
        self._canvas.layout.visibility = "visible"

    @property
    def canvas(self) -> Optional[ipycanvas.MultiCanvas]:
        return self._canvas

    def get_canvas(self, layer: int = 0) -> ipycanvas.Canvas:
        return self._canvas[layer]

    def clear_canvas(self, layer: int = 0):
        self._canvas[layer].clear()

    def observe_mouse_click(self, callback: Callable):
        self.observe(callback, "mouse_click")

    def observe_mouse_drag(self, callback: Callable):
        self.observe(callback, "mouse_drag")

    def observe_mouse_move(self, callback: Callable):
        self.observe(callback, "mouse_move")

    def observe_mouse_down(self, callback: Callable):
        self.observe(callback, "mouse_down")

    def observe_mouse_up(self, callback: Callable):
        self.observe(callback, "mouse_up")

    @property
    def layers(self) -> int:
        return len(self._canvas._canvases)

    @property
    def size(self) -> tuple[int, int]:
        if self.image is None:
            return (0, 0)
        return self.image.shape[-1], self.image.shape[-2]

    def on_click(self, event: dict):
        """Called when the user clicks on the image, the `self.click` traitlet should be updated here (always call super().on_click if you override this.)"""
        self.mouse_click = event

    def on_drag(self, event: dict):
        """Called when the user drags on the image, the `self.drag` traitlet should be updated here (always call super().on_drag if you override this.)"""
        self.mouse_drag = event

    def on_mouse_move(self, event: dict):
        pass  # TODO

    def on_mouse_down(self, event: dict):
        pass  # TODO

    def on_mouse_up(self, event: dict):
        pass  # TODO

    def _on_canvas_mousedown(self, raw_event: dict):
        # dom position of the mouse down event, used by click and drag event handlers
        self._mouse_down_position = (raw_event["relativeX"], raw_event["relativeY"])

    def _on_canvas_mouseup(self, raw_event: dict):
        if self._mouse_down_position is None:
            return  # something weird happened, ignore the event

        x_dom, y_dom = raw_event["relativeX"], raw_event["relativeY"]
        if (x_dom, y_dom) == self._mouse_down_position:
            # trigger a click event
            self.on_click(self._click_data(raw_event))
        elif len(self.mouse_drag) > 0:  # a drag is currently in progress
            # trigger end drag event
            self.on_drag(self._drag_end_data(raw_event))

        # clear all the event data
        self.mouse_drag.clear()  # drag has ended
        self.mouse_click.clear()  # click has ended
        self._mouse_down_position = None

    def _on_canvas_mousemove(self, raw_event: dict):
        if self._mouse_down_position is None:  # is the mouse down?
            return

        # mouse is down, drag?
        if len(self.mouse_drag) > 0:  # continue the drag
            self.on_drag(self._drag_data(raw_event))
        else:  # start a drag?
            x_dom, y_dom = raw_event["relativeX"], raw_event["relativeY"]
            dx = (x_dom - self._mouse_down_position[0]) ** 2
            dy = (y_dom - self._mouse_down_position[1]) ** 2
            if (dx + dy) ** 0.5 > DRAG_THRESHOLD:

                self.on_drag(self._drag_start_data(raw_event))

    def _mouse_data(self, raw_event: dict):
        x_dom, y_dom = raw_event["relativeX"], raw_event["relativeY"]
        w_dom, h_dom = raw_event["boundingRectWidth"], raw_event["boundingRectHeight"]
        w_orig, h_orig = self.size
        return dict(
            # image space
            x=int(x_dom / w_dom * w_orig) if w_orig > 0 else 0,
            y=int(y_dom / h_dom * h_orig) if h_orig > 0 else 0,
            w=w_orig,
            h=h_orig,
            # client space
            x_client=x_dom,
            y_client=y_dom,
            w_client=w_dom,
            h_client=h_dom,
        )

    def _click_data(self, raw_event: dict):
        return self._mouse_data(raw_event)

    def _drag_data(self, raw_event: dict):
        m = self._mouse_data(raw_event)
        if len(self.mouse_drag) > 0:
            x_start, y_start = self.mouse_drag["x_start"], self.mouse_drag["y_start"]
        else:  # new start position
            x_start, y_start = self._mouse_down_position
            x_start = int(x_start / m["w_client"] * m["w"]) if m["w"] > 0 else 0
            y_start = int(y_start / m["h_client"] * m["h"]) if m["h"] > 0 else 0

        return dict(
            **m,
            x_start=x_start,
            y_start=y_start,
            is_start=False,
            is_end=False,
        )

    def _drag_end_data(self, raw_event: dict):
        data = self._drag_data(raw_event)
        data["is_start"] = False
        data["is_end"] = True
        return data

    def _drag_start_data(self, raw_event: dict):
        data = self._drag_data(raw_event)
        data["is_start"] = True
        data["is_end"] = False
        return data

    def __getitem__(self, index):
        """Access pixel data directly from the underlying image tensor."""
        return self.image[index]

    def __setitem__(self, index, value):
        """Set pixel(s) in-place and refresh display directly."""
        self.image[index] = value

    @observe("image")
    def _on_image_change(self, change):
        """Handle image changes by updating the canvas."""
        if change["new"] is None:
            # self.hide()  # the image has been removed
            return self.resize((0, 0))
        _, h, w = change["new"].shape
        self.resize((w, h))

    def resize(self, size: tuple[int, int]):
        # Set logical resolution of canvas
        # must be positive to avoid a ipycanvas bug (downstream HTML issue)
        size = (max(size[0], 1), max(size[1], 1))
        self._canvas.width = size[0]
        self._canvas.height = size[1]
        # Dynamically update container's aspect ratio
        self.layout.aspect_ratio = f"{size[0]} / {size[1]}"
        self.refresh()

    def _to_canvas_data(
        self,
        change: Optional[dict] = None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Convert the tensor to canvas data - set directly with `self._canvas.set_image_data`"""
        if change is None and self.image is None:
            return np.empty((3, 0, 0)), (0, 0)
        elif change is None:
            return np.array(F.to_pil_image(self.image)), (0, 0)
        else:
            return np.array(F.to_pil_image(change["new"])), (
                change.get("x", 0),
                change.get("y", 0),
            )

    def refresh(self, change: Optional[dict] = None):
        """Refresh the canvas.

        Default keys that may be provided in `change`:
        - `new`: image data to display, expects an image tensor of shape (C, H, W).
        - `x`: x-coordinate of the top-left corner of the image (defaults to 0).
        - `y`: y-coordinate of the top-left corner of the image (defaults to 0).
        - `layer`: canvas layer to refresh (defaults to 0).

        All other keys are ignored, but may be used by subclass specific implementations of `refresh`.

        Args:
            change (Optional[dict], optional): dictionary of change information. Defaults to None.
        """
        layer = change.get("layer", 0) if change is not None else 0
        data, (x, y) = self._to_canvas_data(change)
        if data.size > 0:
            self.get_canvas(layer).put_image_data(data, x=x, y=y)

    def hold(self, canvas: ipycanvas.Canvas):
        """Batch draw on the given canvas using a context manager - this haults immediate mode repaints and will batch repaint whent the context manager exits.

        Example:

        ```python
        canvas = image.get_canvas(0)
        with image.hold(canvas):
            canvas.clear()
            canvas.draw_circle(100, 100, 10)
        ```

        Args:
            canvas (ipycanvas.Canvas): canvas to draw on.

        Returns:
            Any: batch draw context manager.
        """
        return ipycanvas.hold_canvas(canvas)

    def __repr__(self):
        return f"Image({[self.layers, *self.size]})"

    def __str__(self):
        return self.__repr__()
