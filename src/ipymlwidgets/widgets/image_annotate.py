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


BOX_NODE_CORNER = (
    BOX_NODE_TOP_LEFT,
    BOX_NODE_TOP_RIGHT,
    BOX_NODE_BOTTOM_RIGHT,
    BOX_NODE_BOTTOM_LEFT,
)

BOX_NODE_EDGE = (
    BOX_NODE_LEFT,
    BOX_NODE_RIGHT,
    BOX_NODE_TOP,
    BOX_NODE_BOTTOM,
)

SELECT_NODE_SIZE = 1


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
        self.observe(self._repaint_selection, names="selection")
        self.boxes = boxes
        self.selection: Optional[BoxSelection] = None
        self._node_size = SELECT_NODE_SIZE
        self._was_dragging = False

    def _repaint_boxes(self, _) -> None:
        with self.hold_repaint():
            if self.boxes is None or len(self.boxes) == 0:
                self.clear(layer=LAYER_BOXES)
                return
            self.clear(layer=LAYER_BOXES)
            self.draw_rect(self.boxes[:, :4], layer=LAYER_BOXES)

    def _repaint_selection(self, _) -> None:
        with self.hold_repaint():
            self.clear(layer=LAYER_SELECTION)
            if self.selection is not None:
                self.draw_rect(self.selection.box[:4][None, :], layer=LAYER_SELECTION)

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

    def _drag_selection(
        self, x: int, y: int, x_start: int, y_start: int
    ) -> BoxSelection:
        dx = int(x - x_start)
        dy = int(y - y_start)
        # select the box and update its position
        box = self.boxes[self.selection.index, :].copy()
        if self.selection.node == BOX_NODE_INSIDE:
            # drag the entire box around
            box[:4] += np.array([dx, dy, dx, dy], dtype=box.dtype)
        elif self.selection.node in BOX_NODE_EDGE:
            if self.selection.node == BOX_NODE_LEFT:
                box[0] += dx
            elif self.selection.node == BOX_NODE_RIGHT:
                box[2] += dx
            elif self.selection.node == BOX_NODE_TOP:
                box[1] += dy
            elif self.selection.node == BOX_NODE_BOTTOM:
                box[3] += dy
        elif self.selection.node in BOX_NODE_CORNER:
            if self.selection.node == BOX_NODE_TOP_LEFT:
                box[0] += dx
                box[1] += dy
            elif self.selection.node == BOX_NODE_TOP_RIGHT:
                box[2] += dx
                box[1] += dy
            elif self.selection.node == BOX_NODE_BOTTOM_RIGHT:
                box[2] += dx
                box[3] += dy
            elif self.selection.node == BOX_NODE_BOTTOM_LEFT:
                box[0] += dx
                box[3] += dy
        else:
            raise ValueError(f"Invalid box node: {self.selection.node}")
        self.selection.box = box
        return self.selection

    @observe("mouse_down")
    def _on_mouse_down(self, event: dict) -> None:
        """Handle mouse down event for box selection."""
        event = event["new"]
        # repaint happens automatically
        self.selection = self._select_box(event["x"], event["y"])

    @observe("mouse_drag")
    def _on_mouse_drag(self, event: dict) -> None:
        """Handle drag event for moving/resizing boxes."""
        event = event["new"]
        if self.selection is not None:
            self._drag_selection(
                event["x"], event["y"], event["x_start"], event["y_start"]
            )
            # manual because internal mutation of self.selection
            self._repaint_selection(None)
            self._was_dragging = True
            # # Update the box in the boxes array
            # self.boxes[self.selection.index, :4] = self.selection.box[:4]
            # self.selection = None
            # self._repaint_boxes(None)

    def normalize_box(self, box: np.ndarray) -> np.ndarray:
        box = box.copy()
        box[0] = min(box[0], box[2])
        box[1] = min(box[1], box[3])
        box[2] = max(box[0], box[2])
        box[3] = max(box[1], box[3])
        return box

    @observe("mouse_up")
    def _on_mouse_up(self, event: dict) -> None:
        """Handle mouse up event for box selection."""
        event = event["new"]
        if self._was_dragging:
            self._was_dragging = False
            with self.hold_repaint():
                box = self.normalize_box(self.selection.box)
                self.boxes[self.selection.index, :] = box
                self.selection = None
                self._repaint_boxes(None)

        # self.selection = None
        # self._repaint_boxes(None)
        # self._repaint_selection()
