import anywidget
import traitlets
import numpy as np
from typing import Optional, Any, List, Dict
import time
from contextlib import contextmanager


class Canvas(anywidget.AnyWidget):
    """A generic canvas widget that displays image data with CSS layout controls.

    This is a base class that provides core canvas functionality without tensor dependencies.
    Subclasses can extend this to add specific image handling and manipulation features.
    """

    # Canvas dimensions (pixel data)
    width = traitlets.Int(8).tag(sync=True)
    height = traitlets.Int(8).tag(sync=True)

    # CSS layout properties
    css_width = traitlets.Unicode("auto").tag(sync=True)
    css_height = traitlets.Unicode("auto").tag(sync=True)
    css_max_width = traitlets.Unicode("100%").tag(sync=True)
    css_max_height = traitlets.Unicode("none").tag(sync=True)

    # Client-side rendered size (actual display size)
    client_size = traitlets.Tuple(
        traitlets.Int(), traitlets.Int(), default_value=(0, 0)
    ).tag(sync=True)

    # Buffered patches for when widget is not ready
    _buffer = traitlets.List([]).tag(sync=True)
    # used to buffer patches when hold is on
    _buffer_hold = traitlets.List([]).tag(sync=False)

    _esm = """
    function render({ model, el }) {
        console.log("JS: Render function called");
        console.log("JS: Model:", model);
        console.log("JS: Model attributes:", model.attributes);
        console.log("JS: Model traits:", model.attributes ? Object.keys(model.attributes) : "No attributes");
        
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        
        // Base canvas styles
        canvas.style.imageRendering = "pixelated";
        canvas.style.border = "1px solid #ccc";
        canvas.style.display = "block";
        
        // ResizeObserver to track actual rendered size
        let resizeObserver;
        
        function updateClientSize() {
            const rect = canvas.getBoundingClientRect();
            const clientWidth = Math.round(rect.width);
            const clientHeight = Math.round(rect.height);
            
            // Only update if size actually changed
            const currentSize = model.get("client_size");
            if (currentSize[0] !== clientWidth || currentSize[1] !== clientHeight) {
                model.set("client_size", [clientWidth, clientHeight]);
                model.save_changes();
                console.log("Client size updated:", clientWidth, clientHeight);
            }
        }
        
        function updateCanvas() {
            const width = model.get("width");
            const height = model.get("height");
            console.log("JS: Canvas update:", width, height);
            // Set canvas pixel dimensions
            canvas.width = width;
            canvas.height = height;
        }
        
        function drawPatch(patchData) {
            if (patchData && patchData.data) {
                const { x, y, width, height, data } = patchData;
                
                console.log("JS: Patch update:", x, y, width, height, data.length);
                console.log("JS: Data type:", typeof data);
                console.log("JS: Data constructor:", data.constructor.name);
                
                // Create ImageData for the patch
                const patchImageData = ctx.createImageData(width, height);
                
                // Handle the ArrayBuffer properly
                let uint8Array;
                if (data instanceof ArrayBuffer) {
                    console.log("JS: Data is ArrayBuffer");
                    uint8Array = new Uint8Array(data);
                } else if (data.buffer) {
                    console.log("JS: Data has buffer property");
                    uint8Array = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
                } else {
                    console.log("JS: Data is neither ArrayBuffer nor has buffer, trying direct conversion");
                    uint8Array = new Uint8Array(data);
                }
                
                console.log("JS: uint8Array length:", uint8Array.length);
                console.log("JS: Expected length:", width * height * 4);
                console.log("JS: First few bytes:", uint8Array.slice(0, 16));
                
                if (uint8Array.length > 0) {
                    patchImageData.data.set(uint8Array);
                    ctx.putImageData(patchImageData, x, y);
                    console.log("JS: Patch updated successfully");
                } else {
                    console.log("JS: Patch data is empty");
                }
            } else {
                console.log("JS: No patch data or data is missing");
                console.log("JS: patchData exists:", !!patchData);
                if (patchData) {
                    console.log("JS: patchData keys:", Object.keys(patchData));
                    console.log("JS: patchData.data exists:", !!patchData.data);
                }
            }
        }
        
        function drawPatches() {
            const bufferedPatches = model.get("_buffer");
            console.log("JS: Processing buffered patches:", bufferedPatches.length);
            
            for (const patch of bufferedPatches) {
                console.log("JS: Applying buffered patch:", patch);
                drawPatch(patch);
            }
            model.set("_buffer", []);
            model.set("_buffer_repaint", false);
            model.save_changes();
        }
        
        function updateStyles() {
            console.log("JS: updateStyles called");
            // Apply CSS layout properties
            canvas.style.width = model.get("css_width");
            canvas.style.height = model.get("css_height");
            canvas.style.maxWidth = model.get("css_max_width");
            canvas.style.maxHeight = model.get("css_max_height");
        }
        
        // Set up ResizeObserver if available
        if (typeof ResizeObserver !== 'undefined') {
            resizeObserver = new ResizeObserver(entries => {
                for (let entry of entries) {
                    if (entry.target === canvas) {
                        updateClientSize();
                    }
                }
            });
            resizeObserver.observe(canvas);
        }
        
        console.log("JS: Setting up event listeners...");
        
        // Listen for changes - use individual listeners for better debugging
        model.on("change:width", () => {
            console.log("JS: width changed");
            updateCanvas();
        });
        model.on("change:height", () => {
            console.log("JS: height changed");
            updateCanvas();
        });
        model.on("change:_buffer", () => {
            console.log("JS: _buffer changed");
            drawPatches();
        });
        model.on("change:css_width change:css_height change:css_max_width change:css_max_height", () => {
            console.log("JS: CSS properties changed");
            updateStyles();
        });

        console.log("JS: Event listeners set up successfully");
        
        // Initial render - just set up the canvas dimensions
        console.log("JS: Performing initial render");
        updateCanvas();
        updateStyles();
        
        // Process any patches that were buffered
        const bufferedPatches = model.get("_buffer");
        console.log("JS: Processing patches buffered before ready", bufferedPatches.length);

        if (bufferedPatches && bufferedPatches.length > 0) {
            console.log("JS: Processing patches buffered before ready");
            drawPatches();
        }

        el.appendChild(canvas);
        
        // Return cleanup function
        return () => {
            if (resizeObserver) {
                resizeObserver.disconnect();
            }
        };
    }
    export default { render };
    """

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        **kwargs,
    ) -> None:
        """Initialize the canvas widget.

        Args:
            width (int): Canvas width in pixels. Defaults to 8.
            height (int): Canvas height in pixels. Defaults to 8.
            **kwargs: Additional keyword arguments passed to parent.
        """
        super().__init__(
            width=width,
            height=height,
            **kwargs,
        )
        self._hold = False

    def set_image(self, image_data: bytes | np.ndarray) -> None:
        """Set the entire image data for the canvas using a full-size patch.

        Args:
            image_data (bytes | np.ndarray): Raw RGBA image data as bytes or numpy array.
        """
        self.set_patch(0, 0, self.width, self.height, image_data)

    def set_patch(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data: bytes | np.ndarray,
    ) -> None:
        """Set image data at a specific location using a patch.

        Args:
            x (int): X coordinate of the patch (left edge).
            y (int): Y coordinate of the patch (top edge).
            width (int): Width of the patch in pixels.
            height (int): Height of the patch in pixels.
            data (bytes | np.ndarray): Raw RGBA image data for the patch.
        """
        data = asbytes(data, width, height)
        patch_dict = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "data": data,
        }
        if self._hold:
            self._buffer_hold.append(patch_dict)
        else:
            buffer = list(self._buffer)
            buffer.append(patch_dict)
            self._buffer = buffer

        if not self._hold:
            self._buffer_repaint = True

    def clear(self) -> None:
        """Clear the canvas by setting an empty patch."""
        self.patch_data = {}

    @contextmanager
    def hold(self):
        self._hold = True
        try:
            yield self
        finally:
            self._hold = False
            self._buffer = self._buffer + self._buffer_hold
            self._buffer_hold = []


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
        if image_data.shape != (height, width, 4):
            raise ValueError(
                f"Argument: `image_data` expected shape (H, W, 4) got {list(image_data.shape)}"
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


def create_demo_canvas() -> Canvas:
    """Create a demo canvas with a basic checkerboard pattern.

    Returns:
        Canvas: A canvas widget with a checkerboard pattern.
    """
    # Create a 64x64 checkerboard
    width, height = 64, 64
    canvas = Canvas(width=width, height=height)

    # Create checkerboard pattern
    image_data = np.zeros((height, width, 4), dtype=np.uint8)

    # Fill with checkerboard pattern
    for y in range(height):
        for x in range(width):
            if (x // 8 + y // 8) % 2 == 0:
                image_data[y, x] = [200, 200, 200, 255]  # Light gray
            else:
                image_data[y, x] = [100, 100, 100, 255]  # Dark gray

    # Set the initial image using patch
    canvas.set_image(image_data)

    return canvas


def demo_patch_updates(canvas: Canvas) -> None:
    """Demonstrate patch updates on a canvas.

    Args:
        canvas (Canvas): The canvas to update with patches.
    """
    # Create a red square patch (16x16)
    patch_size = 16
    red_patch = np.full((patch_size, patch_size, 4), [255, 0, 0, 255], dtype=np.uint8)
    canvas.set_patch(10, 10, patch_size, patch_size, red_patch)

    # Create a blue square patch (16x16)
    blue_patch = np.full((patch_size, patch_size, 4), [0, 0, 255, 255], dtype=np.uint8)
    canvas.set_patch(30, 30, patch_size, patch_size, blue_patch)

    # Create a green rectangle patch (32x8)
    green_patch = np.full((8, 32, 4), [0, 255, 0, 255], dtype=np.uint8)
    canvas.set_patch(16, 50, 32, 8, green_patch)
