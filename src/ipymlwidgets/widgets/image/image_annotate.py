from typing import Optional, Tuple
import pathlib
import numpy as np
import anywidget
import math

from ipymlwidgets.widgets.image import Image
from ipymlwidgets.widgets.canvas import hold_repaint, color_to_hex

from ipymlwidgets.traits import (
    Tensor as TTensor,
    SupportedTensor,
)

from traitlets import Instance, observe, Tuple as TTuple, Int as TInt

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

    def __init__(self, box: np.ndarray, node: int, index: int, clamp : tuple[int,int]):
        self._clamp = clamp # size of the image
        self._box = box.astype(np.int32)
        # update this to change the box using drag, use get_box to get the final box.
        self.box_offset = np.array([0, 0, 0, 0], dtype=np.int32)
        self.node = node
        self.index = index

    def __repr__(self):
        return (
            f"BoxSelection(box={self.get_box().tolist()}@{self.index} node={self.node}, index={self.index})"
        )

    def get_box(self, dtype: np.dtype = np.uint32) -> np.ndarray:
        """Get the box in xyxy format."""
        _box = self._box + self.box_offset
        _box = np.clip(_box, [0, 0, 0, 0], [self._clamp[0], self._clamp[1], self._clamp[0], self._clamp[1]])
        _box = np.array([
            min(_box[0], _box[2]), min(_box[1], _box[3]),  # x1, y1
            max(_box[0], _box[2]), max(_box[1], _box[3])   # x2, y2
        ], dtype=dtype)
        return _box


class ImageAnnotated(Image):
    """A widget that displays an image and allows for annotation with interactive boxes."""

    boxes = TTensor(convert_to="np", allow_none=True).tag(sync=False)
    selection = Instance(BoxSelection, allow_none=True).tag(sync=False)

    # updated when the selection box changes, is always in xyxy format
    _box_selection = TTuple(allow_none=True).tag(sync=False)

    def __init__(
        self,
        image: Optional[SupportedTensor] = None,
        boxes: Optional[SupportedTensor] = None,
        client_node_size: int = 8, # pixels in the client for selection boxes
        **kwargs,
    ) -> None:
        self.boxes = boxes if boxes is not None else np.array([], dtype=np.int32).reshape(0, 4)
        self.selection: Optional[BoxSelection] = None
        self._client_node_size = client_node_size
        self._node_size = 1 # initial value will be updated immediately on resize
        self._dragging = False

        # colours for drawing boxes
        self._box_fill_color = color_to_hex((255, 182, 193, 50))  # Light pink
        self._box_stroke_color = color_to_hex((255, 182, 193, 200))
        self._selection_fill_color = color_to_hex((144, 238, 144, 50)) # Light green
        self._selection_stroke_color = color_to_hex((144, 238, 144, 200))
    
        super().__init__(image=image, layers=3, **kwargs)
        self.observe(self._repaint_boxes, names="boxes")
        self.observe(self._repaint_selection, names="selection")
    
    
    @observe("client_size", "size")
    def _on_resize(self, _: dict) -> None:
        """Update stroke width based on client size."""
        # width, height = self.client_size
        # w, h = self.size
        # # TODO
        # # Example: scale stroke width to be 1% of the smaller dimension, min 1, max 20
        # self._node_size = max(1, min(12, int(min(width, height) * 0.05)))
        # #self._node_size = 2
        width, height = self.client_size
        w, h = self.size

        # Compute scale factors for width and height
        scale_x = w / width if width else 1
        scale_y = h / height if height else 1

        # Use the minimum scale to ensure the stroke is not too thick
        scale = min(scale_x, scale_y)
        # Set node size in canvas pixels so it appears as NODE_SIZE on the client
        self._node_size = int(math.ceil(self._client_node_size * scale))
        with self.hold_repaint(layer=LAYER_SELECTION):
            self.stroke_width = max(self._node_size // 4, 1)
        with self.hold_repaint(layer=LAYER_BOXES):
            self.stroke_width = max(self._node_size // 4, 1)
        
        self._repaint_boxes(None)
        self._repaint_selection(None)

    def _repaint_boxes(self, _) -> None:
        with self.hold_repaint(layer=LAYER_BOXES):
            self.clear()
            self.fill_color = self._box_fill_color
            self.stroke_color = self._box_stroke_color
            if self.boxes is not None and len(self.boxes) > 0:
                self.draw_rect(self.boxes[:, :4])

    def _repaint_selection(self, _) -> None:
        with self.hold_repaint(layer=LAYER_SELECTION):
            self.clear()
            if self.selection is not None:
                self.fill_color = self._selection_fill_color
                self.stroke_color = self._selection_stroke_color
                self.draw_rect(self.selection.get_box()[None, :])

    def _select_box(self, x: int, y: int) -> Optional[BoxSelection]:
        """Select a box and node based on mouse event coordinates, supporting node/corner/edge/inside selection.

        Args:
            x (int): X coordinate of the mouse event.
            y (int): Y coordinate of the mouse event.
        Returns:
            Optional[BoxSelection]: The selected box and node, or None if no box is selected.
        """
        if self.boxes is None or len(self.boxes) == 0:
            return None
        boxes = self.boxes[:, :4]
        #print(boxes[0], x, y)
        left_diff = x - boxes[:, 0]  # [N,]
        right_diff = boxes[:, 2] - x  # [N,]
        top_diff = y - boxes[:, 1]  # [N,]
        bottom_diff = boxes[:, 3] - y  # [N,]
        select_node_size = self._node_size
        #print(left_diff, right_diff, top_diff, bottom_diff)
        # is the mouse inside any of the boxes (including the expanded edge)
        select_inside = (
            (left_diff >= 0)
            & (right_diff >= 0)
            & (top_diff >= 0)
            & (bottom_diff >= 0)
        )
        #print("inside", select_inside)
        if not select_inside.any():
            return None  # no boxes are selected
        left_sel = np.abs(left_diff) <= select_node_size
        right_sel = np.abs(right_diff) <= select_node_size
        top_sel = np.abs(top_diff) <= select_node_size
        bottom_sel = np.abs(bottom_diff) <= select_node_size
        #print(left_sel, right_sel, top_sel, bottom_sel)
        select_top_left = left_sel & top_sel
        select_top_right = right_sel & top_sel
        select_bottom_right = right_sel & bottom_sel
        select_bottom_left = left_sel & bottom_sel
        select_corner = np.stack(
            [select_top_left, select_top_right, select_bottom_left, select_bottom_right]
        )
        select_corner = select_corner & select_inside[np.newaxis, :]
        if select_corner.any():
            corner, selected = np.nonzero(select_corner)
            corner, selected = corner[-1].item(), selected[-1].item()
            return BoxSelection(self.boxes[selected], 5 + corner, selected, self.size)
        # edge selection takes priority over inside selection
        edge = np.stack([left_sel, right_sel, top_sel, bottom_sel])
        select_edge = edge & select_inside[np.newaxis, :]
        if select_edge.any():
            edge, selected = np.nonzero(select_edge)
            edge, selected = edge[-1].item() + 1, selected[-1].item()
            return BoxSelection(self.boxes[selected], edge, selected, self.size)
        selected = np.nonzero(select_inside)[0][-1].item()
        return BoxSelection(self.boxes[selected], BOX_NODE_INSIDE, selected, self.size)

    def _drag_selection(
        self, x: int, y: int, x_start: int, y_start: int
    ) -> BoxSelection:
        """Drag the selected box to a new position or update its size."""
        dx = int(x - x_start)
        dy = int(y - y_start)
        self.selection.box_offset[:] = 0
        offset = self.selection.box_offset
        if self.selection.node == BOX_NODE_INSIDE:
            # drag the entire box around
            offset += np.array([dx, dy, dx, dy], dtype=offset.dtype)
        elif self.selection.node in BOX_NODE_EDGE:
            if self.selection.node == BOX_NODE_LEFT:
                offset[0] += dx
            elif self.selection.node == BOX_NODE_RIGHT:
                offset[2] += dx
            elif self.selection.node == BOX_NODE_TOP:
                offset[1] += dy
            elif self.selection.node == BOX_NODE_BOTTOM:
                offset[3] += dy
        elif self.selection.node in BOX_NODE_CORNER:
            if self.selection.node == BOX_NODE_TOP_LEFT:
                offset[0] += dx
                offset[1] += dy
            elif self.selection.node == BOX_NODE_TOP_RIGHT:
                offset[2] += dx
                offset[1] += dy
            elif self.selection.node == BOX_NODE_BOTTOM_RIGHT:
                offset[2] += dx
                offset[3] += dy
            elif self.selection.node == BOX_NODE_BOTTOM_LEFT:
                offset[0] += dx
                offset[3] += dy
        else:
            raise ValueError(f"Invalid box node: {self.selection.node}")
        return self.selection

    

    @observe("key_press")
    def _on_key_press(self, event: dict) -> None:
        """Handle key press event for box selection."""
        event = event["new"]
        if event["key"] == "Delete":
            if self.selection is not None:
                self.remove_box(self.selection.index)

    @observe("mouse_leave")
    def _on_mouse_leave(self, event: dict) -> None:
        """Handle mouse exit event for box selection."""
        if self._dragging:
            self._drag_end(event)
            
    @observe("mouse_down")
    def _on_mouse_down(self, event: dict) -> None:
        """Handle mouse down event for box selection."""
        event = event["new"]
        # repaint happens automatically
        self.selection = self._select_box(event["x"], event["y"])

    @observe("mouse_up")
    def _on_mouse_up(self, event: dict) -> None:
        """Handle mouse up event for box selection."""
        if self._dragging:
            self._drag_end(event)

    @observe("mouse_drag")
    def _on_mouse_drag(self, event: dict) -> None:
        """Handle drag event for moving/resizing boxes."""
        if self._dragging:
            self._drag_continue(event)
        else:
            self._drag_start(event)

    @hold_repaint
    def _drag_end(self, _: dict) -> None:
        if not self._dragging:
            return # TODO warning? this shouldn't happen...
        box = self.selection.get_box()  # get xyxy box
        if self.selection.index >= 0:
            _new = self.boxes.copy()  # trigger change notification
            _new[self.selection.index, :] = box
            self.boxes = _new
        else:
            self.boxes = np.concatenate([self.boxes, box[None, :]], axis=0)
        self._dragging = False


    @hold_repaint
    def _drag_start(self, event: dict) -> None:
        self._dragging = True
        if self.selection is None:
            # start a new box!
            e = event["new"]
            xyxy = np.array(
                [e["x_start"], e["y_start"], e["x_start"], e["y_start"]],
                dtype=np.int32,
            )
            self.selection = BoxSelection(
                box=xyxy,
                node=BOX_NODE_BOTTOM_RIGHT,
                index=-1,  # negative index indicates that it is a new box
                clamp=self.size,
            )
        self._drag_continue(event)

    @hold_repaint
    def _drag_continue(self, event: dict) -> None:
        """Continue the drag of an already selected box."""
        event = event["new"]
        # this should have been set already on mouse down
        if self.selection is None:
            # something weird happened cancel the drag
            # TODO warning?
            self._dragging = False
            return

        # continue the drag
        self._drag_selection(event["x"], event["y"], event["x_start"], event["y_start"])
        # manual because internal mutation of self.selection
        self._repaint_selection(None)

    def remove_box(self, index: int) -> None:
        """Remove a box from the image."""
        index = int(index) % len(self.boxes)
        s_index = self.selection.index % len(self.boxes) if self.selection is not None else None
        self.boxes = np.delete(self.boxes, index, axis=0)

        display((index, s_index))
        if s_index == index:
            self.selection = None
            self._repaint_selection(None)

    def set_boxes(self, boxes: SupportedTensor) -> None:
        """Set the boxes to be displayed."""
        if boxes is None:
            self.boxes = np.array([], dtype=np.int32).reshape(0, 4)
        else:
            self.boxes = np.array(boxes, dtype=np.int32).reshape(-1, 4)
        self.selection = None

    def crop_selection(self) -> np.ndarray:
        """Crop out the selection box from the image."""
        if self.selection is None:
            return None
        return self.crop([self.selection.index])[0]

    def crop(self, indicies : Optional[list[int]] = None) -> list[np.ndarray]:
        """Crop out each box from the image."""
        if indicies is None:
            boxes = self.boxes[:, :4]
        else:
            boxes = self.boxes[indicies, :4]
        crops = []
        for box in boxes:
            # self.image is already a numpy array, it is automatically converted to HWC uint8
            crop = self.image[box[1] : box[3], box[0] : box[2]]
            crops.append(crop)
        return crops
