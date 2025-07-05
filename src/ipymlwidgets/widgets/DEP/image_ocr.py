import ipywidgets as W
import ipycanvas as C

from ipymlwidgets.widgets.image_annotated import ImageAnnotated, Image

from traitlets import Instance, observe, validate, Dict as TDict
from typing import Optional
import torch


class ImageOCR(ImageAnnotated):

    def __init__(
        self,
        image: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        texts: Optional[str] = None,
        display_size: tuple[int, int] = (320, 320),
    ):
        self.texts = texts if texts is not None else [""] * boxes.shape[0]
        super().__init__(
            image,
            boxes=boxes,
            display_size=display_size,
        )
        # self._text_widget = W.Text(
        #     value="", disabled=True, layout=W.Layout(width=f"100%")
        # )
        # self._text_widget.on_submit(self._on_text_widget_submit)
        # self.add_child(self._text_widget)

        self._ocr_list = OCRList()
        for text, box in zip(self.texts, self.boxes):
            box = box.long()
            image = image[:, box[1] : box[3], box[0] : box[2]]
            self._ocr_list.add_item(image, text)

        self.add_child(self._ocr_list)

    @observe("boxes")
    def on_boxes_change(self, _):
        # if a new box is added, add empty text to texts
        if len(self.texts) == self.boxes.shape[0]:
            return  # just the inner values were updated...
        if len(self.texts) < self.boxes.shape[0]:
            self.texts = self.texts + [""] * (self.boxes.shape[0] - len(self.texts))
        else:
            assert False, "a box was removed?"  # TODO

    # @observe("selection")
    # def on_selection_change(self, _):
    #     if self.selection is not None:
    #         self._text_widget.disabled = False
    #         # update texts to include the newly selected box
    #         assert self.selection.index < self.boxes.shape[0]
    #         index = self.selection.index % self.boxes.shape[0]
    #         # print(self.selection.index, index, self.boxes.shape[0], len(self.texts))
    #         self._text_widget.value = self.texts[index]
    #         self._text_widget.focus()
    #     else:
    #         self._text_widget.value = ""
    #         self._text_widget.disabled = True

    # def _on_text_widget_submit(self, widget):
    #     self.texts[self.selection.index] = widget.value


class OCRList(W.VBox):
    """A VBox widget that displays a vertical list of image and text widget pairs."""

    def __init__(
        self,
        **kwargs,
    ):
        if "layout" not in kwargs:
            kwargs["layout"] = W.Layout(width="100%")
        super().__init__(children=[], **kwargs)
        self._item_containers: list[W.VBox] = []

    def add_item(self, image: torch.Tensor, text: str = "") -> int:
        """Add a new canvas-text pair to the list.

        Args:
            image (torch.Tensor): The image to add.
            text (str): Initial text value. Defaults to "".

        Returns:
            int: Index of the added item.
        """
        text_widget = W.Text(value=text, layout=W.Layout(width="100%"))
        image_widget = Image(image)

        # image_widget = W.HBox(
        #     children=[image_widget],

        # )
        item_container = W.VBox(
            children=[image_widget, text_widget],
            layout=W.Layout(
                margin="0 0 10px 0",
                padding="5px",
                border="1px solid #ddd",
                border_radius="5px",
                height="64px",
            ),
        )
        self._item_containers.append(item_container)
        self.children = tuple(self._item_containers)

    def remove_item(self, index: int) -> None:
        """Remove an item at the specified index.

        Args:
            index (int): Index of the item to remove.
        """

        self._item_containers.pop(index)
        self.children = tuple(self._item_containers)

    def clear(self) -> None:
        """Remove all items from the list."""
        self.children = ()
        self._item_containers.clear()

    def get_text(self, index: int) -> str:
        """Get the text value at the specified index.

        Args:
            index (int): Index of the text to retrieve.

        Returns:
            str: The text value at the specified index.
        """
        return self.get_text_widget(index).value

    def set_text(self, index: int, text: str) -> None:
        """Set the text value at the specified index.

        Args:
            index (int): Index of the text widget to update.
            text (str): New text value.
        """
        self.get_text_widget(index).value = text

    def get_text_widget(self, index: int) -> W.Text:
        """Get the text widget at the specified index.

        Args:
            index (int): Index of the text widget to retrieve.

        Returns:
            widgets.Text: The text widget at the specified index.
        """
        return self._item_containers[index][1]

    def get_image(self, index: int) -> C.Canvas:
        """Get the canvas widget at the specified index.

        Args:
            index (int): Index of the canvas to retrieve.

        Returns:
            Canvas: The canvas widget at the specified index
        """
        return self._item_containers[index][0]

    @property
    def count(self) -> int:
        """Get the number of items in the list.

        Returns:
            int: Number of canvas-text pairs in the list.
        """
        return len(self._item_containers)
