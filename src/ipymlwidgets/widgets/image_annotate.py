from ipymlwidgets.widgets.image import Image

from typing import Optional

from ipymlwidgets.traits.tensor import (
    Tensor as TTensor,
    SupportedTensor,
)


class ImageAnnotate(Image):
    """A widget that displays an image and allows for annotation."""

    boxes = TTensor(allow_none=True).tag(sync=True)

    def __init__(
        self,
        image: Optional[SupportedTensor] = None,
        boxes: Optional[SupportedTensor] = None,
        **kwargs,
    ) -> None:
        super().__init__(image=image, layers=2, **kwargs)
        self.boxes = boxes
