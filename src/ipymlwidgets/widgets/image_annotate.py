from typing import Optional, Any
import numpy as np

from ipymlwidgets.widgets.image import Image
from ipymlwidgets.traits.tensor import (
    Tensor as TTensor,
    SupportedTensor,
)
from traitlets import Instance, observe

LAYER_IMAGE = 0
LAYER_BOXES = 1
LAYER_SELECTION = 2

# Box node constants for selection/drag
BOX_NODE_INSIDE = 0
BOX_NODE_LEFT = 1
BOX_NODE_RIGHT = 2
BOX_NODE_TOP = 3
BOX_NODE_BOTTOM = 4
BOX_NODE_TOP_LEFT = 5
BOX_NODE_TOP_RIGHT = 6
BOX_NODE_BOTTOM_LEFT = 7
BOX_NODE_BOTTOM_RIGHT = 8

SELECT_NODE_SIZE = 8


class BoxSelection:
    """Represents a selected box and node for interaction."""

    def __init__(self, box: np.ndarray, node: int, index: int):
        self.box = box  # shape (4,)
        self.node = node
        self.index = index

    def __repr__(self):
        return f"BoxSelection(box={self.box.tolist()}@{self.index} node={self.node})"


class ImageAnnotated(Image):
    """A widget that displays an image and allows for annotation with interactive boxes."""

    boxes = TTensor(convert_to="np", allow_none=True).tag(sync=False)
    selection = Instance(BoxSelection, allow_none=True).tag(sync=False)

    def __init__(
        self,
        image: Optional[SupportedTensor] = None,
        boxes: Optional[SupportedTensor] = None,
        **kwargs,
    ) -> None:
        super().__init__(image=image, layers=3, **kwargs)
        self.observe(self._repaint_boxes, names="boxes")
        self.boxes = boxes
        self.selection: Optional[BoxSelection] = None
        self._node_size = SELECT_NODE_SIZE

    def _repaint_boxes(self, _) -> None:
        with self.hold_repaint():
            if self.boxes is None or len(self.boxes) == 0:
                self.clear(layer=LAYER_BOXES)
                self.clear(layer=LAYER_SELECTION)
                return
            self.clear(layer=LAYER_BOXES)
            self.draw_rect(self.boxes[:, :4], layer=LAYER_BOXES)
            self._repaint_selection()

    def _repaint_selection(self) -> None:
        with self.hold_repaint():
            self.clear(layer=LAYER_SELECTION)
            if self.selection is not None:
                self.draw_rect(self.selection.box[:4][None, :], layer=LAYER_SELECTION)

    @observe("mouse_click")
    def _on_click(self, event: dict) -> None:
        event = event["new"]
        """Handle mouse click event for box selection."""
        self.selection = self._select_box(event["x"], event["y"])
        self._repaint_selection()

    def _select_box(self, x: int, y: int) -> Optional[BoxSelection]:
        """Select a box and node based on mouse event coordinates, supporting node/corner/edge/inside selection (like box_overlay.py).

        Args:
            x (int): X coordinate of the mouse event.
            y (int): Y coordinate of the mouse event.
        Returns:
            Optional[BoxSelection]: The selected box and node, or None if no box is selected.
        """
        if self.boxes is None or len(self.boxes) == 0:
            return None
        boxes = self.boxes[:, :4]
        left_diff = x - boxes[:, 0]  # [N,]
        right_diff = boxes[:, 2] - x  # [N,]
        top_diff = y - boxes[:, 1]  # [N,]
        bottom_diff = boxes[:, 3] - y  # [N,]
        select_node_size = SELECT_NODE_SIZE
        select_node_size2 = select_node_size / 2
        # is the mouse inside any of the boxes (including the expanded edge)
        select_inside = (
            (left_diff + select_node_size2 > 0)
            & (right_diff + select_node_size2 > 0)
            & (top_diff + select_node_size2 > 0)
            & (bottom_diff + select_node_size2 > 0)
        )
        if not select_inside.any():
            return None  # no boxes are selected
        left_sel = np.abs(left_diff) < select_node_size
        right_sel = np.abs(right_diff) < select_node_size
        top_sel = np.abs(top_diff) < select_node_size
        bottom_sel = np.abs(bottom_diff) < select_node_size
        select_top_left = left_sel & top_sel
        select_top_right = right_sel & top_sel
        select_bottom_right = right_sel & bottom_sel
        select_bottom_left = left_sel & bottom_sel
        select_corner = np.stack(
            [select_top_left, select_top_right, select_bottom_right, select_bottom_left]
        )
        select_corner = select_corner & select_inside[np.newaxis, :]
        if select_corner.any():
            corner, selected = np.nonzero(select_corner)
            corner, selected = corner[-1].item(), selected[-1].item()
            return BoxSelection(self.boxes[selected].copy(), 5 + corner, selected)
        # edge selection takes priority over inside selection
        edge = np.stack([left_sel, right_sel, top_sel, bottom_sel])
        select_edge = edge & select_inside[np.newaxis, :]
        if select_edge.any():
            edge, selected = np.nonzero(select_edge)
            edge, selected = edge[-1].item() + 1, selected[-1].item()
            return BoxSelection(self.boxes[selected].copy(), edge, selected)
        selected = np.nonzero(select_inside)[0][-1].item()
        return BoxSelection(self.boxes[selected].copy(), BOX_NODE_INSIDE, selected)

    # def _on_drag(self, event: dict) -> None:
    #     """Handle drag event for moving/resizing boxes."""
    #     if event.get("is_start", False):
    #         self.selection = self._select_box(event, use_start=True)
    #         self._repaint_selection()
    #     elif event.get("is_end", False):
    #         if self.selection is not None:
    #             self._drag_selection(event)
    #             # Update the box in the boxes array
    #             self.boxes[self.selection.index, :4] = self.selection.box[:4]
    #             self.selection = None
    #             self._repaint_boxes(None)
    #     else:
    #         if self.selection is not None:
    #             self._drag_selection(event)
    #             self._repaint_selection()

    # def _drag_selection(self, event: dict) -> None:
    #     """Update the selected box during drag."""
    #     if self.selection is None:
    #         return
    #     x = event["x"]
    #     y = event["y"]
    #     x_start = event.get("x_start", x)
    #     y_start = event.get("y_start", y)
    #     dx = int(x - x_start)
    #     dy = int(y - y_start)
    #     box = self.selection.box.copy()
    #     # Only support moving the whole box for now
    #     if self.selection.node == BOX_NODE_INSIDE:
    #         box[0] += dx
    #         box[1] += dy
    #         box[2] += dx
    #         box[3] += dy
    #     self.selection.box = box
