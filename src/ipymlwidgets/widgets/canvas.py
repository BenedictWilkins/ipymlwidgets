import anywidget
import traitlets
import numpy as np
from typing import Optional, Any
from contextlib import contextmanager
import pathlib


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
        buffer = list(self._buffer)
        buffer.append({"type": "set", "name": "lineWidth", "value": value})
        self._buffer = buffer

    @property
    def stroke_color(self) -> str:
        """Write-only property. Setting this enqueues a set command for ctx.strokeStyle. Value is not stored in Python."""
        raise AttributeError("stroke_color is write-only.")

    @stroke_color.setter
    def stroke_color(self, value: str) -> None:
        buffer = list(self._buffer)
        buffer.append({"type": "set", "name": "strokeStyle", "value": value})
        self._buffer = buffer

    @property
    def fill_color(self) -> str:
        """Write-only property. Setting this enqueues a set command for ctx.fillStyle. Value is not stored in Python."""
        raise AttributeError("fill_color is write-only.")

    @fill_color.setter
    def fill_color(self, value: str) -> None:
        buffer = list(self._buffer)
        buffer.append({"type": "set", "name": "fillStyle", "value": value})
        self._buffer = buffer

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

    def repaint(self) -> None:
        """Manually triggered a repaint of the canvas."""
        if self._hold > 0:
            pass  # wait until the hold is released the repaint will be triggered then
        else:
            buffer = list(self._buffer)
            self._buffer = []  # clear and reset the buffer to trigger the change
            self._buffer = buffer

    def set_image(
        self, image_data: Optional[bytes | np.ndarray], layer: int = 0
    ) -> None:
        """Set the entire image data for a specific layer using a full-size patch.

        Args:
            image_data (bytes | np.ndarray): Raw RGBA image data as bytes or numpy array.
            layer (int): Layer index to update. Defaults to 0.
        """
        if image_data is None:
            with self.hold_trait_notifications():
                # Clear patches from the specified layer
                self._buffer = [
                    patch for patch in self._buffer if patch.get("layer") != layer
                ]
                self.width = 0
                self.height = 0
            return
        else:
            self.set_patch(0, 0, self.width, self.height, image_data, layer)

    def set_patch(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data: bytes | np.ndarray,
        layer: int = 0,
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

        if self._hold > 0:
            self._buffer_hold.append(patch_dict)
        else:
            buffer = list(self._buffer)
            buffer.append(patch_dict)
            self._buffer = buffer

    @contextmanager
    def hold_repaint(self):
        self._hold += 1
        try:
            yield self
        finally:
            self._hold -= 1
            if self._hold == 0:
                self._buffer = self._buffer + self._buffer_hold
                self._buffer_hold = []

    def __repr__(self):
        return f"MultiCanvas(width={self.width}, height={self.height}, layers={self.layers})"

    def __str__(self):
        return self.__repr__()

    def draw_rect(
        self,
        xyxy: tuple[int, int, int, int] | np.ndarray,
        layer: int = 0,
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
        if isinstance(xyxy, tuple):
            arr = np.array([xyxy], dtype=np.uint32)
        else:
            arr = xyxy.astype(np.int32)  # always int32
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError("Rectangle array must have shape [N, 4].")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("All rectangle coordinates must be integer type.")
        rect_patch = {
            "type": "rect",
            "rects": arr.tobytes(),
            "count": arr.shape[0],
            "layer": layer,
        }
        if self._hold > 0:
            self._buffer_hold.append(rect_patch)
        else:
            buffer = list(self._buffer)
            buffer.append(rect_patch)
            self._buffer = buffer

    def clear(self, layer: int = 0) -> None:
        """Clear the canvas at the specified layer.

        Args:
            layer (int): Layer index to clear. Defaults to 0.
        Returns:
            None: This method does not return a value.
        """
        if layer >= self.layers:
            raise IndexError(
                f"{layer} is out of range Canvas has {self.layers} layers."
            )
        clear_patch = {
            "type": "clear",
            "layer": layer,
        }
        if self._hold > 0:
            self._buffer_hold.append(clear_patch)
        else:
            buffer = list(self._buffer)
            buffer.append(clear_patch)
            self._buffer = buffer


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
