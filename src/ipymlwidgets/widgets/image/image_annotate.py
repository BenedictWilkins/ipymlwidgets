from typing import Optional, Tuple
import pathlib
import numpy as np
import anywidget
import math
import traceback

from ipymlwidgets.widgets.image import Image
from ipymlwidgets.widgets.canvas import hold_repaint, color_to_hex

from ipymlwidgets.traits import (
    Tensor as TTensor,
    SupportedTensor,
)

from traitlets import Instance, observe, Tuple as TTuple, Int as TInt


def debug_exception(func):
    def _wrap(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise e
    return _wrap

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

    # TODO move these to __init__
    _boxes = TTensor(convert_to="np", allow_none=True).tag(sync=False)
    _selection = Instance(BoxSelection, allow_none=True).tag(sync=False)

    def __init__(
        self,
        image: Optional[SupportedTensor] = None,
        boxes: Optional[SupportedTensor] = None,
        client_node_size: int = 8, # pixels in the client for selection boxes
        delete_modifier : Optional[str] = "ctrl",
        hide_modifier : Optional[str] = None,
        **kwargs,
    ) -> None:
        self._boxes : np.ndarray = boxes if boxes is not None else np.array([], dtype=np.int32).reshape(0, 4)
        self._selection: Optional[BoxSelection] = None
        self._dragging = False

        self._box_mask = np.ones_like(self._boxes[:,0], dtype=bool)
        self.delete_modifier = delete_modifier  # modifier key for delete action
        self.hide_modifier = hide_modifier  # modifier key for hide action

        # Colors for drawing boxes
        self._box_fill_color = color_to_hex((255, 182, 193, 50))  # Light pink
        self._box_stroke_color = color_to_hex((255, 182, 193, 200))
        self._selection_fill_color = color_to_hex((144, 238, 144, 50)) # Light green
        self._selection_stroke_color = color_to_hex((144, 238, 144, 200))

        # UI config
        self._client_node_size = client_node_size
        self._node_size = 1 # initial value will be updated immediately on resize

        super().__init__(image=image, layers=3, **kwargs)
       
    @debug_exception
    def _insert_box(self, box: np.ndarray, index : int) -> None:
        if box.ndim == 1:
            box = box.reshape(1, -1)
        if box.ndim != 2:
            raise ValueError(f"Box must be a 2D array, got {box.ndim}D")
        if index < 0:
            index = index % (len(self._boxes) + 1)
        else:
            index = index + 1
        self._box_mask = np.insert(self._box_mask, index, np.ones(box.shape[0], dtype=bool), axis=0)
        self._boxes = np.insert(self._boxes, index, box, axis=0)
        #print(f"[DEBUG] insert box: {box} {index} {self._boxes} {self._box_mask}")

    @hold_repaint
    def set_boxes(self, boxes : np.ndarray) -> None:
        """Set box annotations - replaces existing annotations."""
        self._selection = None
        self._repaint_selection(None)
        self._box_mask = np.ones_like(boxes[:,0], dtype=bool)
        self._boxes = boxes
        self._repaint_boxes(None)

    @debug_exception
    def _set_box(self,  box: np.ndarray, index : int) -> None:
        if index > len(self._boxes):
            raise IndexError(f"Argument `index` {index} out of bounds for box array of shape {list(self._boxes.shape)}")
        index = int(index) % len(self._boxes)
        self._boxes[index] = box
    
    @debug_exception
    def _remove_box(self, index: int) -> None:
        if index > len(self._boxes):
            raise IndexError(f"Argument `index` {index} out of bounds for box array of shape {list(self._boxes.shape)}")
        n = len(self._boxes)
        index = int(index) % n
        self._box_mask = np.delete(self._box_mask, index, axis=0)
        self._boxes = np.delete(self._boxes, index, axis=0) # auto-repaint
        if self._selection is not None:
            if self._selection.index < 0:
                selection_index = self._selection.index % (n + 1)
            else:
                selection_index = self._selection.index
            if selection_index == index:
                self._selection = None # auto-repaint

    @debug_exception
    def _toggle_box(self, index: int) -> None:
        self._box_mask[index] = not self._box_mask[index]

    # does not repaint by default!
    @debug_exception
    def _show_box(self, index: int | slice) -> None:
        self._box_mask[index] = True

    # does not repaint by default!
    @debug_exception
    def _hide_box(self, index: int | slice) -> None:
        self._box_mask[index] = False

    @debug_exception
    def _repaint_boxes(self, _) -> None:
        with self.hold_repaint(layer=LAYER_BOXES):
            self.clear()
            boxes = self._boxes[self._box_mask, :4]  # only use the first 4 columns (x1, y1, x2, y2)
            self.fill_color = self._box_fill_color
            self.stroke_color = self._box_stroke_color
            if boxes is not None and boxes.size > 0:
                self.draw_rect(boxes)
        #print(f"[DEBUG] - repaint: {boxes} {self._boxes} {self._box_mask}")

    @debug_exception
    def _repaint_selection(self, _) -> None:
        with self.hold_repaint(layer=LAYER_SELECTION):
            self.clear()
            if self._selection is not None:
                self.fill_color = self._selection_fill_color
                self.stroke_color = self._selection_stroke_color
                self.draw_rect(self._selection.get_box()[None, :])

    @debug_exception
    def _select_box(self, x: int, y: int) -> Optional[BoxSelection]:
        """Select a box and node based on mouse event coordinates, supporting node/corner/edge/inside selection.

        Args:
            x (int): X coordinate of the mouse event.
            y (int): Y coordinate of the mouse event.
        Returns:
            Optional[BoxSelection]: The selected box and node, or None if no box is selected.
        """
        if self._boxes is None or len(self._boxes) == 0:
            return None
        boxes = self._boxes[:, :4]
        left_diff = x - boxes[:, 0]  # [N,]
        right_diff = boxes[:, 2] - x  # [N,]
        top_diff = y - boxes[:, 1]  # [N,]
        bottom_diff = boxes[:, 3] - y  # [N,]
        select_node_size = self._node_size
        # is the mouse inside any of the boxes (including the expanded edge)
        select_inside = (
            (left_diff >= 0)
            & (right_diff >= 0)
            & (top_diff >= 0)
            & (bottom_diff >= 0)
        )
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
            return BoxSelection(self._boxes[selected], 5 + corner, selected, self.size)
        # edge selection takes priority over inside selection
        edge = np.stack([left_sel, right_sel, top_sel, bottom_sel])
        select_edge = edge & select_inside[np.newaxis, :]
        if select_edge.any():
            edge, selected = np.nonzero(select_edge)
            edge, selected = edge[-1].item() + 1, selected[-1].item()
            return BoxSelection(self._boxes[selected], edge, selected, self.size)
        selected = np.nonzero(select_inside)[0][-1].item()
        return BoxSelection(self._boxes[selected], BOX_NODE_INSIDE, selected, self.size)

    @debug_exception
    def _drag_selection(
        self, x: int, y: int, x_start: int, y_start: int
    ) -> BoxSelection:
        """Drag the selected box to a new position or update its size."""
        dx = int(x - x_start)
        dy = int(y - y_start)
        self._selection.box_offset[:] = 0
        offset = self._selection.box_offset
        if self._selection.node == BOX_NODE_INSIDE:
            # drag the entire box around
            offset += np.array([dx, dy, dx, dy], dtype=offset.dtype)
        elif self._selection.node in BOX_NODE_EDGE:
            if self._selection.node == BOX_NODE_LEFT:
                offset[0] += dx
            elif self._selection.node == BOX_NODE_RIGHT:
                offset[2] += dx
            elif self._selection.node == BOX_NODE_TOP:
                offset[1] += dy
            elif self._selection.node == BOX_NODE_BOTTOM:
                offset[3] += dy
        elif self._selection.node in BOX_NODE_CORNER:
            if self._selection.node == BOX_NODE_TOP_LEFT:
                offset[0] += dx
                offset[1] += dy
            elif self._selection.node == BOX_NODE_TOP_RIGHT:
                offset[2] += dx
                offset[1] += dy
            elif self._selection.node == BOX_NODE_BOTTOM_RIGHT:
                offset[2] += dx
                offset[3] += dy
            elif self._selection.node == BOX_NODE_BOTTOM_LEFT:
                offset[0] += dx
                offset[3] += dy
        else:
            raise ValueError(f"Invalid box node: {self._selection.node}")
        return self._selection

     # TODO it is possible for a box of size zero to be added... dont allow this!
    
    @debug_exception
    def _drag_end(self, _: dict) -> None:
        if not self._dragging:
            return # TODO warning? this shouldn't happen...
        box = self._selection.get_box()  # get xyxy box
        if self._selection.index >= 0:
            self._set_box(box, self._selection.index)
        else:
            self._insert_box(box, self._selection.index)
        self._dragging = False
        #print(["[DEBUG] drag end: ", self._selection, self._boxes, self._box_mask])
        self._repaint_boxes(None)

    @debug_exception
    def _drag_start(self, event: dict) -> None:
        self._dragging = True
        if self._selection is None:
            # start a new box!
            e = event["new"]
            xyxy = np.array(
                [e["x_start"], e["y_start"], e["x_start"], e["y_start"]],
                dtype=np.int32,
            )
            self._selection = BoxSelection(
                box=xyxy,
                node=BOX_NODE_BOTTOM_RIGHT,
                index=-1,  # negative index indicates that it is a new box
                clamp=self.size,
            )
        self._drag_continue(event) # repaints already

    @debug_exception
    def _drag_continue(self, event: dict) -> None:
        """Continue the drag of an already selected box."""
        event = event["new"]
        # this should have been set already on mouse down
        if self._selection is None:
            # something weird happened cancel the drag
            # TODO warning?
            self._dragging = False
            return
        # continue the drag
        self._drag_selection(event["x"], event["y"], event["x_start"], event["y_start"])
        self._repaint_selection(None)

    @observe("client_size", "size")
    @debug_exception
    def _on_resize(self, _: dict) -> None:
        """Update stroke width based on client size."""
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

    @observe("mouse_leave")
    @debug_exception
    def _on_mouse_leave(self, event: dict) -> None:
        """Handle mouse exit event for box selection."""
        if self._dragging:
            self._drag_end(event)
            
    @observe("mouse_down")
    @debug_exception
    def _on_mouse_down(self, event: dict) -> None:
        """Handle mouse down event for box selection."""
        event = event["new"]
        #print(f"[DEBUG] - MOUSE DOWN  {self.delete_modifier} {event.keys()}")
        selection = self._select_box(event["x"], event["y"])
        if selection is None:
            self._selection = None
        else:
            if event.get(self.delete_modifier, False):
                # this will set self._selection if necessary
                self._remove_box(selection.index) 
                self._repaint_boxes(None)
            elif event.get(self.hide_modifier, False):
                self._toggle_box(selection.index)
                self._repaint_boxes(None)
            else:
                self._selection = selection
                
        #print(f"[DEBUG] - selection: {self._selection}")
        self._repaint_selection(None)

    @observe("mouse_up")
    @debug_exception
    def _on_mouse_up(self, event: dict) -> None:
        """Handle mouse up event for box selection."""
        if self._dragging:
            self._drag_end(event)

    @observe("mouse_drag")
    @debug_exception
    def _on_mouse_drag(self, event: dict) -> None:
        """Handle drag event for moving/resizing boxes."""
        if self._dragging:
            self._drag_continue(event)
        else:
            self._drag_start(event)

    @debug_exception
    def crop_selection(self) -> list[np.ndarray]:
        """Crop out the selection box from the image."""
        if self._selection is None:
            return []
        return self.crop([self._selection.index])

    @debug_exception
    def crop_visible(self) -> list[np.ndarray]:
        """Crop out the visible boxes from the image."""
        indices = self._box_mask.nonzero()[0]
        if indices.size == 0:
            return []
        return self.crop(indices)

    @debug_exception
    def crop(self, indices: Optional[list[int]] = None) -> list[np.ndarray]:
        """Crop out each box from the image."""
        if indices is None:
            boxes = self._boxes[:, :4]
        else:
            boxes = self._boxes[indices, :4]
        crops = []
        for box in boxes:
            crop = self.image[box[1] : box[3], box[0] : box[2]]
            crops.append(crop)
        return crops
