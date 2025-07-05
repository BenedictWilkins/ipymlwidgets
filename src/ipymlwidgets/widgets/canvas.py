import anywidget
import traitlets
import numpy as np
from typing import Optional


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

    # Image data as bytes - synced to frontend
    _image_data = traitlets.Bytes().tag(sync=True)

    _esm = """
    function render({ model, el }) {
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
            const rawData = model.get("_image_data");
            
            console.log("Canvas update:", width, height, rawData ? rawData.byteLength : 0);
            
            // Set canvas pixel dimensions
            canvas.width = width;
            canvas.height = height;
            
            if (rawData && rawData.byteLength > 0) {
                const imageData = ctx.createImageData(width, height);
                
                // Handle the ArrayBuffer properly
                let uint8Array;
                if (rawData instanceof ArrayBuffer) {
                    uint8Array = new Uint8Array(rawData);
                } else if (rawData.buffer) {
                    // Handle typed arrays or DataView
                    uint8Array = new Uint8Array(rawData.buffer, rawData.byteOffset, rawData.byteLength);
                } else {
                    // Fallback: try to create directly
                    uint8Array = new Uint8Array(rawData);
                }
                
                console.log("Expected bytes:", width * height * 4, "Got:", uint8Array.length);
                
                if (uint8Array.length > 0) {
                    imageData.data.set(uint8Array);
                    ctx.putImageData(imageData, 0, 0);
                    console.log("Canvas updated successfully");
                } else {
                    console.log("Uint8Array is empty");
                }
            } else {
                console.log("No image data to display");
            }
            
            // Update client size after canvas update
            setTimeout(updateClientSize, 0);
        }
        
        function updateStyles() {
            // Apply CSS layout properties
            canvas.style.width = model.get("css_width");
            canvas.style.height = model.get("css_height");
            canvas.style.maxWidth = model.get("css_max_width");
            canvas.style.maxHeight = model.get("css_max_height");
            
            // Update client size after style changes
            setTimeout(updateClientSize, 0);
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
        
        // Listen for changes
        model.on("change:_image_data change:width change:height", updateCanvas);
        model.on("change:css_width change:css_height change:css_max_width change:css_max_height", updateStyles);
        
        // Initial render
        updateCanvas();
        updateStyles();
        
        // Cleanup function
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

    def set_image_data(self, image_data: bytes) -> None:
        """Set the raw image data for the canvas.

        Args:
            image_data (bytes): Raw RGBA image data as bytes.
        """
        self._image_data = image_data

    def clear(self) -> None:
        """Clear the canvas by setting empty image data."""
        self._image_data = b""
