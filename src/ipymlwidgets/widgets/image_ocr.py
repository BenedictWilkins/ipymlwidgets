from typing import Optional, Any
import numpy as np

from ipymlwidgets.widgets.image_annotate import ImageAnnotated
from ipymlwidgets.traits.tensor import (
    Tensor as TTensor,
    SupportedTensor,
)
from traitlets import Instance, observe, List as TList
import ipywidgets as W

from PIL import Image
import io


def array_to_png_bytes(arr):
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="png")
    return buf.getvalue()


class ImageOCR(W.HBox):

    texts = TList(allow_none=True).tag(sync=False)
    boxes = TTensor(convert_to="np", allow_none=True).tag(sync=False)

    def __init__(
        self,
        image: Optional[SupportedTensor] = None,
        boxes: Optional[SupportedTensor] = None,
        texts: Optional[list[str]] = None,
    ) -> None:
        self.texts = texts
        self.boxes = boxes
        self.image_widget = ImageAnnotated(image=image, boxes=boxes)

        crops = self.image_widget.crop_boxes()
        self.ocr_list = []
        for crop, text in zip(crops, texts):
            self.ocr_list.append(
                W.VBox(
                    [
                        W.Label(value=text),
                        W.Image(value=array_to_png_bytes(crop), format="png"),
                    ]
                )
            )
        super().__init__(
            children=[
                W.Box(
                    children=[self.image_widget],
                    layout=W.Layout(width="50%", height="auto", display="block"),
                ),
                W.VBox(
                    children=self.ocr_list, layout=W.Layout(width="50%", height="auto")
                ),
            ],
            layout=W.Layout(width="100%", height="auto", display="flex"),
        )

        self.image_widget.observe(self._on_selection_change, names=["selection"])

    def _on_selection_change(self, change: dict) -> None:
        if change["new"] is None:
            return
        index = change["new"].index
        item = self.ocr_list[index]

        crops = self.image_widget.crop_boxes()
        item.children = [
            W.Label(value=self.texts[index]),
            W.Image(value=array_to_png_bytes(crops[index]), format="png"),
        ]
