import anywidget
import traitlets
import numpy as np
from typing import Optional, Any
from contextlib import contextmanager

from ipymlwidgets.traits.tensor import (
    Tensor as TTensor,
    OptionalDependency,
    SupportedTensor,
)


class Image(anywidget.AnyWidget):
    """A canvas widget that displays image data with CSS layout controls."""

    # Canvas dimensions (pixel data)
    width = traitlets.Int(8).tag(sync=True)
    height = traitlets.Int(8).tag(sync=True)

    # CSS layout properties
    css_width = traitlets.Unicode("auto").tag(sync=True)
    css_height = traitlets.Unicode("auto").tag(sync=True)
    css_max_width = traitlets.Unicode("100%").tag(sync=True)
    css_max_height = traitlets.Unicode("none").tag(sync=True)

    # Backing image field - not synced, uses Tensor trait
    image = TTensor(allow_none=True).tag(sync=False)

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
                
                // The key fix: handle the ArrayBuffer properly
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
        }
        
        function updateStyles() {
            // Apply CSS layout properties
            canvas.style.width = model.get("css_width");
            canvas.style.height = model.get("css_height");
            canvas.style.maxWidth = model.get("css_max_width");
            canvas.style.maxHeight = model.get("css_max_height");
        }
        
        // Listen for changes
        model.on("change:_image_data change:width change:height", updateCanvas);
        model.on("change:css_width change:css_height change:css_max_width change:css_max_height", updateStyles);
        
        // Initial render
        updateCanvas();
        updateStyles();
        
        el.appendChild(canvas);
    }
    export default { render };
    """

    def __init__(
        self,
        image: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Initialize the canvas widget.

        Args:
            image (Optional[np.ndarray]): Initial image array with shape (H, W, 3) or (H, W, 4). Defaults to None.
            **kwargs: Additional keyword arguments passed to parent.
        """
        # Initialize width and height from image if provided
        if image is not None:
            height, width = image.shape[:2]
        else:
            width, height = 8, 8  # Default size
        self._hold = False
        super().__init__(
            width=width,
            height=height,
            **kwargs,
        )

        self.observe(self._refresh_internal, names=["image"])
        if image is not None:
            self.image = image

    def _on_image_change(self, change: dict) -> None:
        """Handle changes to the backing image field.

        Args:
            change (dict): The change dictionary from traitlets.
        """
        new_image = change["new"]
        if new_image is not None:
            self._convert_image_to_bytes(new_image)

    def _convert_image_to_bytes(self, tensor: SupportedTensor) -> bytes:
        """Convert image array to bytes and update synced fields."""
        image_trait: TTensor = self.traits()["image"]
        dependency: OptionalDependency = image_trait.get_dependency(self, tensor)
        array = dependency.to_numpy_image(tensor)
        assert array.ndim == 3
        assert array.shape[2] == 4  # HWC format RGBA
        assert array.dtype == np.uint8
        self.width = array.shape[1]
        self.height = array.shape[0]
        return array.tobytes()

    def _refresh_internal(self, _) -> None:
        """Refresh the image display."""
        if self._hold:  # wait until hold is released
            return
        if self.image is None:
            pass  # TODO handle this..
        self._image_data = self._convert_image_to_bytes(self.image)

    def refresh(self) -> None:
        """Refresh the image display."""
        self._refresh_internal(None)  # not used?

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the shape of the image."""
        return self.image.shape

    def __setitem__(self, index: Any, value: SupportedTensor) -> None:
        """Set pixels via tensor indexing - any operation that is supported by the ML framework in use will work.

        Args:
            index (Any): The index to set the pixel at.
            value (SupportedTensor): The value to set the pixel to.
        """
        self.image[index] = value
        self._refresh_internal(None)

    def __getitem__(self, key):
        """Get pixels using array indexing syntax.

        Args:
            key: Index or slice for pixel selection

        Returns:
            np.ndarray: Selected pixel data
        """
        if self.image is None:
            raise ValueError("No image data available")

        return self.image[key]

    @contextmanager
    def hold(self):
        """Context manager to suppress immediate repaint.

        Multiple pixel operations can be performed without triggering immediate updates to the display. The display is updated once when exiting the context.

        Example:
        ```python
            with image.hold():
                image[10:20, 10:20] = [255, 0, 0]  # Red square
                image[30:40, 30:40] = [0, 255, 0]  # Green square
                image[50:60, 50:60] = [0, 0, 255]  # Blue square
            # Display updates here with all changes at once
        ```
        """
        self._hold = True
        try:
            yield self
        finally:
            self._hold = False
            # Trigger a single update with the current image state
            if self.image is not None:
                self.refresh()
