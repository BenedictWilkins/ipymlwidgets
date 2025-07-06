from typing import Optional, Any
from contextlib import contextmanager
import pathlib
import functools

import anywidget
import traitlets
import numpy as np

import time


class Canvas(anywidget.AnyWidget):
    """A multi-layer canvas widget that displays multiple image layers stacked on top of each other."""

    # Canvas dimensions (pixel data)
    width = traitlets.Int(8).tag(sync=True)
    height = traitlets.Int(8).tag(sync=True)

    # CSS layout properties
    css_width = traitlets.Unicode("auto").tag(sync=True)
    css_height = traitlets.Unicode("auto").tag(sync=True)

    # Client-side rendered size (actual display size)
    client_size = traitlets.Tuple(
        traitlets.Int(), traitlets.Int(), default_value=(0, 0)
    ).tag(sync=True)

    # Number of layers
    layers = traitlets.Int(1).tag(sync=True)

    # Buffered patches for when widget is not ready
    _buffer = traitlets.List([]).tag(sync=True)
    # used by the front end to ack that the render was completed
    _buffer_ack = traitlets.Int(0).tag(sync=True)
    # used by the backend to notify that the buffer has been changed/flushed
    _buffer_syn = traitlets.Int(0).tag(sync=True)
    # used to buffer patches when hold is on
    _buffer_hold = traitlets.List([]).tag(sync=False)

    # Mouse events
    mouse_move = traitlets.Dict().tag(sync=True)
    mouse_down = traitlets.Dict().tag(sync=True)
    mouse_up = traitlets.Dict().tag(sync=True)
    mouse_click = traitlets.Dict().tag(sync=True)
    mouse_drag = traitlets.Dict().tag(sync=True)
    mouse_leave = traitlets.Dict().tag(sync=True)
    mouse_enter = traitlets.Dict().tag(sync=True)

    # Built-in anywidget CSS property
    _css = """
    .multicanvas-wrapper {
        display: grid;
        width: 100%;
        height: 100%;
        max-width: 100%;
        max-height: none;
    }
    .multicanvas-canvas {
        grid-area: 1 / 1;
        width: 100%;
        height: 100%;
        display: block;
        image-rendering: pixelated;
        border: 1px solid rgba(0,0,0,0.1);
        background: transparent;
    }
    """

    # Javascript
    _esm = pathlib.Path(__file__).parent / "canvas.js"

    @property
    def stroke_width(self) -> int:
        """Write-only property. Setting this enqueues a set command for ctx.lineWidth. Value is not stored in Python."""
        raise AttributeError("stroke_width is write-only.")

    @stroke_width.setter
    def stroke_width(self, value: int) -> None:
        self.add_draw_command(
            {"type": "set", "name": "lineWidth", "value": value, "layer": self._layer}
        )

    @property
    def stroke_color(self) -> str:
        """Write-only property. Setting this enqueues a set command for ctx.strokeStyle. Value is not stored in Python."""
        raise AttributeError("stroke_color is write-only.")

    @stroke_color.setter
    def stroke_color(self, value: str) -> None:
        self.add_draw_command(
            {"type": "set", "name": "strokeStyle", "value": value, "layer": self._layer}
        )

    @property
    def fill_color(self) -> str:
        """Write-only property. Setting this enqueues a set command for ctx.fillStyle. Value is not stored in Python."""
        raise AttributeError("fill_color is write-only.")

    @fill_color.setter
    def fill_color(self, value: str) -> None:
        self.add_draw_command(
            {"type": "set", "name": "fillStyle", "value": value, "layer": self._layer}
        )

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        layers: int = 1,
        **kwargs,
    ) -> None:
        """Initialize the multi-layer canvas widget.

        Args:
            width (int): Canvas width in pixels. Defaults to 8.
            height (int): Canvas height in pixels. Defaults to 8.
            layers (int): Number of canvas layers. Defaults to 1.
            **kwargs: Additional keyword arguments passed to parent.
        """
        super().__init__(
            width=width,
            height=height,
            layers=layers,
            **kwargs,
        )
        self._hold = 0
        self._layer = 0

    def _flush_buffer(self):
        """Flushes the currently draw command buffer for rendering in the front end.

        The draw will only happen (buffer flushed) if:
            - a batch draw hold is not currently active
            - the draw command buffer is not empty
        """
        if not self._hold and not self._buffer and self._buffer_hold:
            buffer_hold = self._buffer_hold
            self._buffer_hold = []
            self._buffer = buffer_hold  # + [{"type": "debug", "value": nonce}]
            self._buffer_syn = self._buffer_syn + 1  # trigger render

    def add_draw_command(self, command):
        """Add a draw command to the the command buffer.

        The command will be immediate if there is not batch draw hold active.

        Args:
            command (dict[str,Any]): The draw command.
        """
        self._buffer_hold.append(command)
        self._flush_buffer()

    @traitlets.observe("_buffer_ack")
    def _on_render_complete(self, _):
        self._buffer.clear()
        self._flush_buffer()

    def redraw(self) -> None:
        """Manually triggered a repaint of the canvas."""
        self._flush_buffer()  # will only flush if no hold repaint is active.

    def set_image(
        self, image_data: Optional[bytes | np.ndarray], layer: Optional[int] = None
    ) -> None:
        """Set the entire image data for a specific layer using a full-size patch.

        Args:
            image_data (bytes | np.ndarray): Raw RGBA image data as bytes or numpy array.
            layer (int): Layer index to update. Defaults to 0.
        """
        layer = layer if layer is not None else self._layer
        if image_data is None:
            return self.clear(layer)
        else:
            return self.set_patch(0, 0, self.width, self.height, image_data, layer)

    def set_patch(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data: bytes | np.ndarray,
        layer: Optional[int] = None,
    ) -> None:
        """Set image data at a specific location using a patch.

        Args:
            x (int): X coordinate of the patch (left edge).
            y (int): Y coordinate of the patch (top edge).
            width (int): Width of the patch in pixels.
            height (int): Height of the patch in pixels.
            data (bytes | np.ndarray): Raw RGBA image data for the patch.
            layer (int): Layer index to update. Defaults to 0.
        """
        data = asbytes(data, width, height)
        patch_dict = {
            "type": "patch",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "data": data,
            "layer": layer,
        }
        self.add_draw_command(patch_dict)

    def draw_rect(
        self,
        xyxy: tuple[int, int, int, int] | np.ndarray,
        layer: Optional[int] = None,
    ) -> None:
        """Draw one or more rectangles on the specified layer using current style traits.

        Args:
            xyxy (tuple[int, int, int, int] | numpy.ndarray):
                - Single rectangle: (x0, y0, x1, y1) coordinates.
                - Batched: numpy array of shape [N, 4] (int), each row is (x0, y0, x1, y1).
            layer (int): Layer index to draw on. Defaults to 0.

        Note:
            The rectangle(s) appearance is controlled by by following attributes:
                - stroke_width (int): Outline thickness in pixels.
                - stroke_color (str): Outline color (CSS color string).
                - fill_color (str): Fill color (CSS color string, empty for no fill).
            Set these attributes directly on the Canvas instance before calling `draw_rect`.

        Raises:
            ValueError: If any coordinate is not an integer type.
        """
        layer = layer if layer is not None else self._layer
        if isinstance(xyxy, tuple):
            arr = np.array([xyxy], dtype=np.uint32)
        else:
            arr = xyxy.astype(np.int32)  # always int32
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError("Rectangle array must have shape [N, 4].")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("All rectangle coordinates must be integer type.")
        rect_patch = {
            "type": "draw",
            "shape": "rect",
            "data": arr.tobytes(),
            "count": arr.shape[0],
            "layer": layer,
        }
        self.add_draw_command(rect_patch)

    def clear(self, layer: Optional[int] = None) -> None:
        """Clear the canvas at the specified layer.

        Args:
            layer (int): Layer index to clear. Defaults to 0.
        Returns:
            None: This method does not return a value.
        """

        def clear_commands(buffer: list, layer: int):
            for c in buffer:
                if c["layer"] != layer or c["type"] in ["set"]:
                    yield c

        layer = layer if layer is not None else self._layer
        clear_patch = {
            "type": "clear",
            "layer": layer,
        }
        # we could remove all previous draw commands for the layer... TODO
        self.add_draw_command(clear_patch)

    @contextmanager
    def hold_repaint(self, layer: Optional[int] = None):
        old_layer = self._layer
        # the layer is set here, any draw calls used while this context manager is active
        # will use this layer - unless it is explicitly override in the call.
        # the layer will be restored when the context manager exits.
        # it is safe to nest hold_repaint.
        # if the layer was not specified, use the original layer
        layer = layer if layer is not None else old_layer
        self._layer = layer
        self._hold += 1
        try:
            yield self
        finally:
            self._layer = old_layer
            self._hold -= 1
            if not self._hold:
                self._flush_buffer()
        assert self._hold >= 0  # sanity check

    def __repr__(self):
        return f"MultiCanvas(width={self.width}, height={self.height}, layers={self.layers})"

    def __str__(self):
        return self.__repr__()


@staticmethod
def hold_repaint(func):
    """Decorator to hold repaint operations."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not isinstance(self, Canvas):
            raise ValueError(
                "`hold_repaint` can only be used as a decorator on `Canvas` methods"
            )
        with self.hold_repaint():
            return func(self, *args, **kwargs)

    return wrapper


def asbytes(image_data: Any, width: int, height: int) -> bytes:
    """Convert image data to bytes format.

    Args:
        image_data (Any): Image data as numpy array or bytes.
        width (int): Expected width of the image.
        height (int): Expected height of the image.

    Returns:
        bytes: Image data as bytes.

    Raises:
        ValueError: If image_data is not the expected type or size.
    """
    if isinstance(image_data, np.ndarray):
        if tuple(image_data.shape) != (height, width, 4):
            raise ValueError(
                f"Argument: `image_data` expected shape [{height}, {width}, 4] got {list(image_data.shape)}"
            )
        return image_data.tobytes()
    elif isinstance(image_data, bytes):
        if len(image_data) != height * width * 4:
            raise ValueError(
                f"Argument: `image_data` expected {height * width * 4} bytes, got {len(image_data)}"
            )
        return image_data
    else:
        raise ValueError(
            f"Argument: `image_data` expected numpy array or bytes got {type(image_data)}"
        )
