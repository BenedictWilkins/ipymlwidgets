import numpy as np
from typing import Optional, Any

from ipymlwidgets.widgets.canvas import Canvas
from ipymlwidgets.traits.tensor import (
    Tensor as TTensor,
    OptionalDependency,
    SupportedTensor,
)


class Image(Canvas):
    """A canvas widget that displays image data with CSS layout controls."""

    # Backing image field - not synced, uses Tensor trait
    image = TTensor(allow_none=True).tag(sync=False)

    def __init__(
        self,
        image: Optional[SupportedTensor] = None,
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
        self.observe(self._repaint_image, names=["image"])
        self.image = image

    def _convert_image(self, tensor: Optional[SupportedTensor]) -> Optional[bytes]:
        """Convert image array to bytes and update synced fields.

        Args:
            tensor (SupportedTensor): Image tensor to convert.

        Returns:
            Optional[bytes]: Converted image data as bytes or None if tensor is None.
        """
        if tensor is None:
            return None
        image_trait: TTensor = self.traits()["image"]
        dependency: OptionalDependency = image_trait.get_dependency(self, tensor)
        array = dependency.to_numpy_image(tensor)
        assert array.ndim == 3
        assert array.shape[2] == 4  # HWC format RGBA
        assert array.dtype == np.uint8
        return array

    def _repaint_image(self, change: Optional[dict[str, Any]] = None) -> None:
        """Internal call back to repaint the image."""
        if change is None:
            self.set_image(self._convert_image(self.image))
        else:
            image = change["new"]
            with self.hold_trait_notifications():
                image = self._convert_image(image)  # HWC
                self.width = image.shape[1]
                self.height = image.shape[0]
                self.set_image(image)

    def repaint(self) -> None:
        """Manually repaint the image, e.g. after direct pixel operations."""
        self._repaint_image(None)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the shape of the image.

        Returns:
            tuple[int, int, int]: Shape of the image as (height, width, channels).
        """
        return self.image.shape

    def __setitem__(self, index: Any, value: SupportedTensor) -> None:
        """Set pixels via tensor indexing - any operation that is supported by the ML framework in use will work.

        Args:
            index (Any): The index to set the pixel at.
            value (SupportedTensor): The value to set the pixel to.
        """
        self.image[index] = value
        self._repaint_image(None)

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
