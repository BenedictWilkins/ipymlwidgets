from traitlets import Instance, Undefined, validate, observe
from typing import Optional

import torch
import ipyevents as E

from ipymlwidgets.utils import color_to_hex, get_colors, resolve_colors
from ipymlwidgets.widgets.image import Image
from ipymlwidgets.traits.tensor import Tensor


DEFAULT_STROKE_COLOR = (255, 0, 0, 255)
DEFAULT_STROKE_WIDTH = 2
DEFAULT_SELECTED_CMAP = torch.tensor(DEFAULT_STROKE_COLOR).unsqueeze(0)

SELECT_NODE_SIZE = 8  # 5 pixels in the client

LAYER_BOXES = 0
LAYER_SELECTED = 1

BOX_DIM = 5  # x1,y1,x2,y2,class

BOX_NODE_TOP_LEFT = 0
BOX_NODE_TOP_RIGHT = 1
BOX_NODE_BOTTOM_RIGHT = 2
BOX_NODE_BOTTOM_LEFT = 3
BOX_NODE_LEFT = 4
BOX_NODE_RIGHT = 5
BOX_NODE_TOP = 6
BOX_NODE_BOTTOM = 7
BOX_NODE_INSIDE = 8

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


class BoxSelection:

    def __init__(self, box: torch.Tensor, node: int, index: int):
        self.box = box
        self.node = node
        self.index = index

    def __repr__(self):
        return f"BoxSelection(box={self.box.detach().cpu().tolist()}@{self.index} node={self.node})"


class BoxOverlay(Image):

    boxes = Tensor(allow_none=True).tag(sync=False)
    cmap = Tensor(allow_none=True).tag(sync=False)
    selection = Instance(BoxSelection, allow_none=True).tag(sync=False)

    def __init__(
        self,
        boxes: Optional[torch.Tensor] = None,
        cmap: Optional[torch.Tensor] = None,
        overlay_size: tuple[int, int] = (320, 320),
        **kwargs,
    ):

        # 2 layers - 0: boxes and 1: selected box
        super().__init__(
            None,
            layers=2,
            **kwargs,
        )
        # fake the size of this image - super().image trailet is always None
        self._overlay_size = overlay_size
        self.resize(self._overlay_size)

        # TODO cmap will break if more classes are added...
        boxes = BoxOverlay.validate_boxes(boxes, self.size)
        self.cmap = self.validate_cmap(cmap, boxes=boxes)
        self.boxes = boxes
        # the size of the selectable node in image space
        self._node_size = 0

    @observe("client_size")
    def _on_client_size_change(self, _):
        # they should match in aspect ratio... no weird distortions please!
        scale = max(self.size) / max(self.client_size)
        self._node_size = max(1, scale * SELECT_NODE_SIZE)

    @observe("boxes")
    def _on_boxes_change(self, _):
        if self.boxes is None:
            return
        self.draw_boxes()

    @property
    def size(self):
        return self._overlay_size

    @validate("boxes")
    def _validate_boxes(self, proposal):
        return self.validate_boxes(proposal["value"], self.size)

    @classmethod
    def validate_boxes(cls, value: Optional[torch.Tensor], size: tuple[int, int]):
        if value is None or value.numel() == 0:
            return torch.empty(0, BOX_DIM, dtype=torch.float32)
        if value.ndim != 2:
            raise ValueError(
                f"Argument: `boxes` expected tensor of shape [N,{BOX_DIM}] but got shape: {value.shape}"
            )
        boxes = torch.zeros(value.shape[0], BOX_DIM)
        boxes[:, : value.shape[1]] = value.float()
        # check that the boxes are valid
        x1, y1, x2, y2, c = boxes.unbind(dim=1)
        if (x1 > x2).any() or (y1 > y2).any():
            raise ValueError(f"Argument: `boxes` must have x1 <= x2 and y1 <= y2")
        if (x1 < 0).any() or (y1 < 0).any() or (x2 < 0).any() or (y2 < 0).any():
            raise ValueError(f"Argument: `boxes` must have non-negative values")
        if (x2 >= size[0]).any() or (y2 >= size[1]).any():
            raise ValueError(
                f"Argument: `boxes` must have values less than the overlay size: {size}"
            )
        return boxes

    @validate("cmap")
    def _validate_cmap(self, proposal):
        return self.validate_cmap(proposal["value"])

    def validate_cmap(self, value, boxes: Optional[torch.Tensor] = None):
        if value is None:
            boxes = boxes if boxes is not None else self.boxes
            n = len(boxes[:, 4].long().unique())
            return get_colors(max(n, 1), pastel_factor=0.5)
        return value

    def select_box(self, mouse_event: dict) -> BoxSelection:
        x, y = mouse_event["x"], mouse_event["y"]
        x1, y1, x2, y2, *_ = self.boxes.unbind(dim=1)
        # check if the mouse is inside any of the boxes
        left_diff = x - x1  # [N,]
        right_diff = x2 - x  # [N,]
        top_diff = y - y1  # [N,]
        bottom_diff = y2 - y  # [N,]

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

        left_sel = left_diff.abs() < select_node_size
        right_sel = right_diff.abs() < select_node_size
        top_sel = top_diff.abs() < select_node_size
        bottom_sel = bottom_diff.abs() < select_node_size

        select_top_left = left_sel & top_sel
        select_top_right = right_sel & top_sel
        select_bottom_right = right_sel & bottom_sel
        select_bottom_left = left_sel & bottom_sel
        select_corner = torch.stack(
            [select_top_left, select_top_right, select_bottom_right, select_bottom_left]
        )
        select_corner = select_corner & select_inside.unsqueeze(0)
        if select_corner.any():
            corner, selected = torch.nonzero(select_corner, as_tuple=True)
            corner, selected = corner[-1].item(), selected[-1].item()
            return BoxSelection(
                self.boxes[selected, :].clone(),
                corner,  # [0,1,2,3] -> [TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT]
                selected,
            )
        # edge selection takes priority over inside selection
        edge = torch.stack([left_sel, right_sel, top_sel, bottom_sel])
        select_edge = edge & select_inside.unsqueeze(0)
        if select_edge.any():
            edge, selected = torch.nonzero(select_edge, as_tuple=True)
            edge, selected = edge[-1].item() + 4, selected[-1].item()
            return BoxSelection(
                self.boxes[selected, :].clone(),
                edge,  # [4,5,6,7] -> [LEFT, RIGHT, TOP, BOTTOM]
                selected,
            )
        selected = torch.nonzero(select_inside, as_tuple=True)[0][-1].item()
        return BoxSelection(
            self.boxes[selected, :].clone(),
            BOX_NODE_INSIDE,  # [8] -> [INSIDE]
            selected,
        )

    def drag_selection(
        self, x: int, y: int, x_start: int, y_start: int
    ) -> BoxSelection:
        dx = int(x - x_start)
        dy = int(y - y_start)
        # select the box and update its position
        box = self.boxes[self.selection.index, :].clone()
        if self.selection.node == BOX_NODE_INSIDE:
            # drag the entire box around
            box[:4] += torch.tensor([dx, dy, dx, dy], dtype=torch.float32)
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

    # raw mouse click handler
    def on_click(self, event: dict):
        # repaint for the select layer is called by traitlet observer
        self.selection = self.select_box(event["x"], event["y"])
        self.draw_boxes()
        self.draw_selection()
        super().on_click(event)

    def on_drag(self, event: dict):
        if event["is_start"]:
            # the drag has started, select the box at the position of the drag start
            select = self.select_box(event["x_start"], event["y_start"])
            # if select is None, we are going to create a new box
            if select is None:
                new_box = torch.zeros(BOX_DIM)
                index = self.boxes.shape[0]
                new_box[0] = event["x_start"]
                new_box[1] = event["y_start"]
                new_box[2] = event["x_start"]
                new_box[3] = event["y_start"]
                self.boxes = torch.cat([self.boxes, new_box.unsqueeze(0)], dim=0)
                self.selection = BoxSelection(
                    self.boxes[index, :].clone(),
                    BOX_NODE_BOTTOM_RIGHT,
                    index,
                )
            else:
                self.selection = select
            self.draw_boxes()
            self.draw_selection()

        if self.selection is not None:  # the box is already being dragged (probably...)
            if not event["is_start"] and not event["is_end"]:
                # is it possible there is a drag happening but no box was selected
                if self.selection is not None:
                    # the drag has already started, move the box
                    self.drag_selection(
                        event["x"], event["y"], event["x_start"], event["y_start"]
                    )
            elif event["is_end"]:
                # the drag has ended, ensure the box is rendered in the final position
                select = self.drag_selection(
                    event["x"], event["y"], event["x_start"], event["y_start"]
                )
                # update the box when the change has completed
                self.boxes[select.index, :] = select.box
                box = select.box.clone()
                # ensure that the box has the correct format...
                box[0] = min(select.box[0], select.box[2])
                box[1] = min(select.box[1], select.box[3])
                box[2] = max(select.box[0], select.box[2])
                box[3] = max(select.box[1], select.box[3])
                self.boxes[select.index, :] = box

            self.draw_boxes()
            self.draw_selection()

        super().on_drag(event)

    def draw_selection(self):
        if self.selection is None:
            self.clear_canvas(LAYER_SELECTED)
        else:
            boxes = self.selection.box.unsqueeze(0)
            self._draw_boxes(
                boxes,
                DEFAULT_SELECTED_CMAP,
                layer=LAYER_SELECTED,
            )

    def draw_boxes(self):
        self._draw_boxes(
            layer=LAYER_BOXES,
            boxes=self.boxes,
            cmap=self.cmap,
        )

    def _draw_boxes(
        self,
        boxes: torch.Tensor,
        cmap: torch.Tensor,
        layer: int = LAYER_BOXES,
    ):
        color = resolve_colors(boxes[:, 4].long(), cmap)
        canvas = self.get_canvas(layer)
        with self.hold(canvas):  # batch the draw
            canvas.save()
            canvas.clear()
            canvas.line_width = self._node_size
            for i in range(boxes.shape[0]):
                # draw in the center of the pixels
                x1, y1, x2, y2 = (boxes[i, :4] + 0.5).tolist()
                c = tuple(color[i].tolist())
                canvas.stroke_style = color_to_hex(c)
                canvas.fill_style = color_to_hex(c[:3] + (90,))
                canvas.begin_path()
                canvas.rect(x1, y1, x2 - x1, y2 - y1)
                canvas.fill()
                canvas.stroke()
            canvas.restore()
