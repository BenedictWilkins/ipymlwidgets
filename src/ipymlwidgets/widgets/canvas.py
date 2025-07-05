import anywidget
import traitlets
import numpy as np
from typing import Optional, Any
from contextlib import contextmanager


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

    _esm = """
    function render({ model, el }) {
        // Apply initial CSS sizing
        function updateStyles() {
            wrapper.style.width = model.get("css_width");
            wrapper.style.height = model.get("css_height");
        }
        
        // Create inner wrapper for canvas positioning
        const wrapper = document.createElement("div");
        wrapper.classList.add("multicanvas-wrapper");
        
        // Array to hold all canvas elements and contexts
        const canvases = [];
        const contexts = [];
        
        // Get number of layers
        const numLayers = model.get("layers") || 1;
        
        // Create canvas elements for each layer
        for (let i = 0; i < numLayers; i++) {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            
            // Add CSS class to canvas
            canvas.classList.add("multicanvas-canvas");
            // No need for z-index or absolute positioning
            
            // Set up crisp scaling
            ctx.imageSmoothingEnabled = false;
            
            // Add to arrays
            canvases.push(canvas);
            contexts.push(ctx);
            
            // Add to wrapper
            wrapper.appendChild(canvas);
        }
        
        el.appendChild(wrapper);
        
        // ResizeObserver to track actual rendered size
        let resizeObserver;
        
        function updateClientSize() {
            const rect = wrapper.getBoundingClientRect();
            const clientWidth = Math.round(rect.width);
            const clientHeight = Math.round(rect.height);
            
            // Only update if size actually changed
            const currentSize = model.get("client_size");
            if (currentSize[0] !== clientWidth || currentSize[1] !== clientHeight) {
                model.set("client_size", [clientWidth, clientHeight]);
                model.save_changes();
            }
        }
        
        function updateCanvas() {
            const width = model.get("width");
            const height = model.get("height");
            
            // Update all canvases with the new dimensions
            for (let i = 0; i < canvases.length; i++) {
                const canvas = canvases[i];
                canvas.width = width;
                canvas.height = height;
            }
        }
        
        function drawPatch(patchData, layerIndex) {
            if (patchData && patchData.data && layerIndex < contexts.length) {
                const { x, y, width, height, data } = patchData;
                const ctx = contexts[layerIndex];
                
                // Create ImageData for the patch
                const patchImageData = ctx.createImageData(width, height);
                
                // Handle the ArrayBuffer properly
                let uint8Array;
                if (data instanceof ArrayBuffer) {
                    uint8Array = new Uint8Array(data);
                } else if (data.buffer) {
                    uint8Array = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
                } else {
                    uint8Array = new Uint8Array(data);
                }
                
                if (uint8Array.length > 0) {
                    patchImageData.data.set(uint8Array);
                    ctx.putImageData(patchImageData, x, y);
                }
            }
        }
        
        // Draw a rectangle (box) on the specified layer.
        // boxData: {
        //   xyxy: [x0, y0, x1, y1],
        //   color: CSS color string or [r,g,b,a],
        //   thickness: int (optional, default 1)
        // }
        function drawBox(boxData, layerIndex) {
            if (!boxData || layerIndex >= contexts.length) return;
            const ctx = contexts[layerIndex];
            const { xyxy, color, thickness } = boxData;
            if (!xyxy || xyxy.length !== 4) return;
            const [x0, y0, x1, y1] = xyxy;
            ctx.save();
            ctx.strokeStyle = Array.isArray(color)
                ? `rgba(${color[0]},${color[1]},${color[2]},${color.length > 3 ? color[3] / 255 : 1})`
                : (color || 'red');
            ctx.lineWidth = thickness || 1;
            ctx.beginPath();
            ctx.rect(x0, y0, x1 - x0, y1 - y0);
            ctx.stroke();
            ctx.restore();
        }
        
        function drawPatches() {
            const bufferedPatches = model.get("_buffer");
            for (const patch of bufferedPatches) {
                const layerIndex = patch.layer || 0;
                if (patch.type === 'box') {
                    drawBox(patch, layerIndex);
                } else {
                    drawPatch(patch, layerIndex);
                }
            }
            model.set("_buffer", []);
            model.save_changes();
        }
        
        // Set up ResizeObserver if available
        if (typeof ResizeObserver !== 'undefined') {
            resizeObserver = new ResizeObserver(entries => {
                for (let entry of entries) {
                    if (entry.target === wrapper) {
                        updateClientSize();
                    }
                }
            });
            resizeObserver.observe(wrapper);
        }
        
        // Listen for changes - use individual listeners for better debugging
        model.on("change:width", () => {
            updateCanvas();
        });
        model.on("change:height", () => {
            updateCanvas();
        });
        model.on("change:_buffer", () => {
            drawPatches();
        });
        model.on("change:css_width change:css_height change:css_max_width change:css_max_height", () => {
            updateStyles();
        });
      
        // Mouse event handling
        let isMouseDown = false;
        let dragStartPos = null;
        let dragThreshold = 3; // pixels
        
        function getMouseData(event) {
            const rect = wrapper.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            // Convert to canvas coordinates using the first canvas as reference
            const canvasX = Math.floor((x / rect.width) * canvases[0].width);
            const canvasY = Math.floor((y / rect.height) * canvases[0].height);
            
            return {
                x: canvasX,
                y: canvasY,
                x_client: x,
                y_client: y,
                w_client: rect.width,
                h_client: rect.height,
                w: canvases[0].width,
                h: canvases[0].height
            };
        }
        
        function handleMouseDown(event) {
            isMouseDown = true;
            const mouseData = getMouseData(event);
            dragStartPos = { x: mouseData.x, y: mouseData.y, clientX: event.clientX, clientY: event.clientY };
            
            model.set("mouse_down", mouseData);
            model.save_changes();
        }
        
        function handleMouseUp(event) {
            const mouseData = getMouseData(event);
            
            if (isMouseDown && dragStartPos) {
                // Check if it was a click (no significant movement)
                const dx = event.clientX - dragStartPos.clientX;
                const dy = event.clientY - dragStartPos.clientY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < dragThreshold) {
                    // It's a click
                    model.set("mouse_click", mouseData);
                    model.save_changes();
                } else {
                    // It was a drag that ended
                    const dragData = { 
                        ...mouseData, 
                        x_start: dragStartPos.x,
                        y_start: dragStartPos.y
                    };
                    model.set("mouse_drag", dragData);
                    model.save_changes();
                }
                
                model.set("mouse_up", mouseData);
                model.save_changes();
            }
            
            isMouseDown = false;
            dragStartPos = null;
        }
        
        function handleMouseMove(event) {
            const mouseData = getMouseData(event);
            model.set("mouse_move", mouseData);
            model.save_changes();
            
            if (isMouseDown && dragStartPos) {
                const dx = event.clientX - dragStartPos.clientX;
                const dy = event.clientY - dragStartPos.clientY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance >= dragThreshold) {
                    // It's a drag
                    const dragData = { 
                        ...mouseData, 
                        x_start: dragStartPos.x,
                        y_start: dragStartPos.y
                    };
                    model.set("mouse_drag", dragData);
                    model.save_changes();
                }
            }
        }
        
        function handleMouseEnter(event) {
            const mouseData = getMouseData(event);
            model.set("mouse_enter", mouseData);
            model.save_changes();
        }
        
        function handleMouseLeave(event) {
            const mouseData = getMouseData(event);
            model.set("mouse_leave", mouseData);
            
            // Reset mouse state when leaving canvas
            isMouseDown = false;
            dragStartPos = null;
        }
        
        // Add mouse event listeners to wrapper
        wrapper.addEventListener('mousedown', handleMouseDown);
        wrapper.addEventListener('mouseup', handleMouseUp);
        wrapper.addEventListener('mousemove', handleMouseMove);
        wrapper.addEventListener('mouseenter', handleMouseEnter);
        wrapper.addEventListener('mouseleave', handleMouseLeave);

        // Initial render - just set up the canvas dimensions
        updateCanvas();
        updateStyles();
        
        // Process any patches that were buffered
        const bufferedPatches = model.get("_buffer");

        if (bufferedPatches && bufferedPatches.length > 0) {
            drawPatches();
        }

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
        self._hold = False

    def repaint(self) -> None:
        """Manually triggered a repaint of the canvas."""
        if self._hold:
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
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "data": data,
            "layer": layer,
        }

        if self._hold:
            self._buffer_hold.append(patch_dict)
        else:
            buffer = list(self._buffer)
            buffer.append(patch_dict)
            self._buffer = buffer

    @contextmanager
    def hold(self):
        self._hold = True
        try:
            yield self
        finally:
            self._hold = False
            self._buffer = self._buffer + self._buffer_hold
            self._buffer_hold = []

    def __repr__(self):
        return f"MultiCanvas(width={self.width}, height={self.height}, layers={self.layers})"

    def __str__(self):
        return self.__repr__()

    def draw_rect(
        self,
        xyxy: tuple[int, int, int, int],
        color: str | list[int],
        thickness: int = 1,
        layer: int = 0,
    ) -> None:
        """Draw a rectangle (box) on the canvas at the specified layer.

        Args:
            xyxy (tuple[int, int, int, int]): (x0, y0, x1, y1) coordinates of the rectangle.
            color (str | list[int]): CSS color string or [r, g, b, a] list (0-255).
            thickness (int): Border thickness in pixels. Defaults to 1.
            layer (int): Layer index to draw on. Defaults to 0.
        Returns:
            None: This method does not return a value.
        """
        box_patch = {
            "type": "box",
            "xyxy": xyxy,
            "color": color,
            "thickness": thickness,
            "layer": layer,
        }
        if self._hold:
            self._buffer_hold.append(box_patch)
        else:
            buffer = list(self._buffer)
            buffer.append(box_patch)
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
