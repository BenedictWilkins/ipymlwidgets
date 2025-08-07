import numpy as np
from typing import Optional, Any
from traitlets import Tuple as TTuple

from ipymlwidgets.widgets.canvas import Canvas
from ipymlwidgets.traits.tensor import (
    Tensor as TTensor,
    OptionalDependency,
    SupportedTensor,
)


class Image(Canvas):
    """A canvas widget that displays image data."""

    # Backing image field - not synced
    image = TTensor(allow_none=True).tag(sync=False)
    
    def __init__(
        self,
        image: Optional[SupportedTensor] = None,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            image (Optional[SupportedTensor]): image to display. Defaults to None.
        """
        # Initialize width and height from image if provided
        if image is not None:
            height, width = image.shape[:2]
        else:
            width, height = 8, 8  # Default size
        self._hold = False
        super().__init__(
            size=(width, height),
            **kwargs,
        )
        self.observe(self._repaint_image, names=["image"])
        self.image = image

    def _to_numpy_image(
        self, tensor: Optional[SupportedTensor]
    ) -> Optional[np.ndarray]:
        """Convert image array to numpy array.

        Args:
            tensor (SupportedTensor): Image tensor to convert.

        Returns:
            Optional[np.ndarray]: Converted image data as numpy array or None if tensor is None.
        """
        if tensor is None:
            return None
        image_trait: TTensor = self.traits()["image"]
        dependency: OptionalDependency = image_trait.get_dependency(tensor, obj=self)
        array = dependency.to_numpy_image(tensor)
        assert array.ndim == 3
        assert array.shape[2] == 4  # HWC format RGBA
        assert array.dtype == np.uint8
        return array

    def _to_numpy(self, tensor: SupportedTensor) -> np.ndarray:
        """Convert a supported tensor to a numpy array."""
        # assume that we are being consistent with tensor type usage...
        image_trait: TTensor = self.traits()["image"]
        dependency: OptionalDependency = image_trait.get_dependency(tensor, obj=self)
        return dependency.to_numpy(tensor)

    def _repaint_image(self, change: Optional[dict[str, Any]] = None) -> None:
        """Internal call back to repaint the image."""
        if change is None:
            self.set_image(self._to_numpy_image(self.image))
        else:
            image = change["new"]
            with self.hold_trait_notifications():
                image = self._to_numpy_image(image)  # HWC
                self.size = (image.shape[1], image.shape[0])
                self.set_image(image)

    def repaint(self) -> None:
        """Manually repaint the image, e.g. after direct pixel operations."""
        self._repaint_image(None)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the shape of the image, see also `size`.

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
