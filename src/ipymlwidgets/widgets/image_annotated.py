import ipywidgets as W

from ipymlwidgets.widgets.image import Image, _STYLE_HTML_LAYOUT
from traitlets import Instance, observe, validate
from typing import Optional
from copy import deepcopy
import torch
import functools
import time

DEFAULT_STROKE_COLOR = (255, 0, 0, 255)
DEFAULT_STROKE_WIDTH = 2
DEFAULT_SELECTED_CMAP = torch.tensor((255, 0, 0, 255)).unsqueeze(0)

SELECT_NODE_SIZE = 5  # 5 pixels in the client...?


LAYER_BOXES = 0
LAYER_KEYPOINTS = 1
LAYER_MASK = 2
LAYER_SELECTED = 3


from ipymlwidgets.utils import color_to_hex, get_colors


class BoxSelection:

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

    def __init__(self, box: torch.Tensor, node: int, index: int):
        self.box = box
        self.node = node
        self.index = index

    def __repr__(self):
        return f"BoxSelection(box={self.box.detach().cpu().tolist()}@{self.index} node={self.node})"


_OVERLAY_STYLE = """
<style>
.overlay-image {
    position: absolute;
    top: 0;
    left: 0;
}
</style>
"""


def debounce(delay: float = 0.016):
    """Debounce function calls to reduce the number of times a function is called."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, _debounce: bool = True, **kwargs):
            if not _debounce:
                return func(self, *args, **kwargs)

            # Get or create last call time for this function
            func_name = func.__name__
            last_call_attr = f"_last_{func_name}_call"

            current_time = time.time()
            last_call_time = getattr(self, last_call_attr, 0)

            # Only call if enough time has passed
            if current_time - last_call_time >= delay:
                setattr(self, last_call_attr, current_time)
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


class ImageAnnotated(W.Box):

    BOX_DIM = 6  # x1,y1,x2,y2,angle,class

    boxes = Instance(torch.Tensor, allow_none=True).tag(sync=False)
    keypoints = Instance(torch.Tensor, allow_none=True).tag(sync=False)
    mask = Instance(torch.Tensor, allow_none=True).tag(sync=False)

    # the currently selected annotation
    select = Instance(BoxSelection, allow_none=True).tag(sync=False)

    @validate("boxes")
    def _validate_boxes(self, proposal):
        value = proposal["value"]
        if value is not None:
            if value.ndim != 2:
                raise ValueError(
                    f"Argument: `boxes` expected tensor of shape [N,{ImageAnnotated.BOX_DIM}] but got shape: {value.shape}"
                )
            _boxes = torch.zeros(value.shape[0], ImageAnnotated.BOX_DIM)
            _boxes[:, : value.shape[1]] = value.float()
        else:
            _boxes = None
        return _boxes

    def __init__(
        self,
        image,
        boxes: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        classes: Optional[list[str]] = None,
        cmap: Optional[torch.Tensor] = None,
        display_size: int | tuple[int, int] = (320, 320),
    ):
        self._image_canvas = Image(image, layers=1)
        self._image_canvas.add_class("overlay-image")
        if isinstance(display_size, int):
            scale = display_size / self._image_canvas.size[0]
            size = (display_size, int(self._image_canvas.size[1] * scale))

        elif isinstance(display_size, (tuple, list)):
            scale = (
                display_size[0] / self._image_canvas.size[0],
                display_size[1] / self._image_canvas.size[1],
            )
            # this is the display size of the image (and the overlay)
            size = (
                int(self._image_canvas.size[0] * scale[0]),
                int(self._image_canvas.size[1] * scale[1]),
            )
        else:
            raise ValueError(
                f"Argument: `display_size` expected int or tuple[int,int] but got {type(display_size)}"
            )
        # this should match the client side pixels ideally...
        overlay = torch.zeros((4, size[0], size[1]))
        self._overlay_canvas = Image(overlay, layers=4)
        self._overlay_canvas.add_class("overlay-image")
        size = (f"{size[0]}px", f"{size[1]}px")
        super().__init__(
            [
                W.HTML(_OVERLAY_STYLE, layout=_STYLE_HTML_LAYOUT),
                self._image_canvas,
                self._overlay_canvas,
            ],
            layout=W.Layout(
                width=size[0],
                height=size[1],
            ),
        )
        # ensure the boxes have the correct shape x1,y1,x2,y2,angle,class
        self.boxes = boxes
        self.keypoints = keypoints.float() if keypoints is not None else None
        self.mask = mask.float() if mask is not None else None
        self._classes = classes
        self._cmap = (
            cmap
            if cmap is not None
            else get_colors(len(classes) if classes else 1, pastel_factor=0.5)
        )

        # set up event handlers
        self._overlay_canvas.observe_mouse_click(self._on_mouse_click)
        self._overlay_canvas.observe_mouse_drag(self._on_mouse_drag)

        # simple debouncing - drawing should not happen at every mouse move event

    def _select_box(self, x: float, y: float) -> BoxSelection:
        select = None  # default nothing selected
        # overlay space scale
        x_scale = self._overlay_canvas.size[0] / self._image_canvas.size[0]
        y_scale = self._overlay_canvas.size[1] / self._image_canvas.size[1]

        # scale the boxes to the overlay space from image space to do the selection checks
        boxes = self.boxes.clone()
        boxes[:, :4] += 0.5  # paint offset (to center on pixels)
        boxes[:, :4] *= torch.tensor(
            [x_scale, y_scale, x_scale, y_scale], dtype=torch.float32
        )
        # position in the overlay space
        xy = (x, y)

        x1, y1, x2, y2, *_ = boxes.unbind(dim=1)
        # check if the mouse is inside any of the boxes
        left_diff = xy[0] - x1  # [N,]
        right_diff = x2 - xy[0]  # [N,]
        top_diff = xy[1] - y1  # [N,]
        bottom_diff = y2 - xy[1]  # [N,]

        select_node_size = SELECT_NODE_SIZE * 2
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
        select_corner = select_corner & select_inside.unsqueeze(
            0
        )  # must also be inside!
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
        # ok, but are inside any?
        if select_inside.any():
            selected = torch.nonzero(select_inside, as_tuple=True)[0][-1].item()
            return BoxSelection(
                self.boxes[selected, :].clone(),
                BoxSelection.BOX_NODE_INSIDE,  # [8] -> [INSIDE]
                selected,
            )

        return select

    def _drag_box(self, mouse_event: dict) -> BoxSelection:
        x_iscale = self._image_canvas.size[0] / self._overlay_canvas.size[0]
        y_iscale = self._image_canvas.size[1] / self._overlay_canvas.size[1]
        xy = (int(mouse_event["x"] * x_iscale), int(mouse_event["y"] * y_iscale))
        xy_start = (
            int(mouse_event["x_start"] * x_iscale),
            int(mouse_event["y_start"] * y_iscale),
        )
        dx = xy[0] - xy_start[0]
        dy = xy[1] - xy_start[1]

        box = self.boxes[self.select.index, :].clone()

        # update the box
        if self.select.node == BoxSelection.BOX_NODE_INSIDE:
            # drag the entire box around
            box[:4] += torch.tensor([dx, dy, dx, dy], dtype=torch.float32)
        elif self.select.node in BoxSelection.BOX_NODE_EDGE:
            if self.select.node == BoxSelection.BOX_NODE_LEFT:
                box[0] += dx
            elif self.select.node == BoxSelection.BOX_NODE_RIGHT:
                box[2] += dx
            elif self.select.node == BoxSelection.BOX_NODE_TOP:
                box[1] += dy
            elif self.select.node == BoxSelection.BOX_NODE_BOTTOM:
                box[3] += dy
        elif self.select.node in BoxSelection.BOX_NODE_CORNER:
            if self.select.node == BoxSelection.BOX_NODE_TOP_LEFT:
                box[0] += dx
                box[1] += dy
            elif self.select.node == BoxSelection.BOX_NODE_TOP_RIGHT:
                box[2] += dx
                box[1] += dy
            elif self.select.node == BoxSelection.BOX_NODE_BOTTOM_RIGHT:
                box[2] += dx
                box[3] += dy
            elif self.select.node == BoxSelection.BOX_NODE_BOTTOM_LEFT:
                box[0] += dx
                box[3] += dy
        else:
            raise ValueError(f"Invalid box node: {self.select.node}")

        self.select.box = box
        return self.select

    # raw mouse click handler
    def _on_mouse_click(self, event: dict):
        # repaint for the select layer is called by traitlet observer
        self.select = self._select_box(event["new"]["x"], event["new"]["y"])
        self.draw_boxes(self.boxes)

    # raw mouse drag handler
    def _on_mouse_drag(self, event: dict):
        event = event["new"]
        if event["is_start"]:
            # the drag has started, select the box at the position of the drag start
            # repaint is called by traitlet observer
            select = self._select_box(event["x_start"], event["y_start"])
            x_iscale = self._image_canvas.size[0] / self._overlay_canvas.size[0]
            y_iscale = self._image_canvas.size[1] / self._overlay_canvas.size[1]
            x_start, y_start = (
                int(event["x_start"] * x_iscale),
                int(event["y_start"] * y_iscale),
            )
            # if select is None, we are going to create a new box
            if select is None:
                new_box = torch.zeros(ImageAnnotated.BOX_DIM)
                index = self.boxes.shape[0]
                new_box[0] = x_start
                new_box[1] = y_start
                new_box[2] = x_start
                new_box[3] = y_start
                boxes = torch.cat([self.boxes, new_box.unsqueeze(0)], dim=0)
                self.select = BoxSelection(
                    boxes[-1, :].clone(),
                    BoxSelection.BOX_NODE_BOTTOM_RIGHT,
                    index,
                )
                self.boxes = boxes
            else:
                self.select = select
                self.draw_boxes(self.boxes)

        if self.select is not None:
            if not event["is_start"] and not event["is_end"]:
                # is it possible there is a drag happening but no box was selected
                if self.select is not None:
                    # the drag has already started, move the box
                    self._drag_box(event)
                    self.draw_selected_box()
            elif event["is_end"]:
                # the drag has ended, ensure the box is rendered in the final position
                select = self._drag_box(event)
                # update the box when the change has completed
                self.boxes[select.index, :] = select.box
                box = select.box.clone()
                # ensure that the box has the correct format...
                box[0] = min(select.box[0], select.box[2])
                box[1] = min(select.box[1], select.box[3])
                box[2] = max(select.box[0], select.box[2])
                box[3] = max(select.box[1], select.box[3])
                self.boxes[select.index, :] = box
                self.draw_boxes(self.boxes)
                self.draw_selected_box()

    @observe("select")
    def _on_select_change(self, _):
        # the change happens internally, we can access the value directly
        self.draw_selected_box()

    def draw_selected_box(self):
        if self.select is None:
            self._overlay_canvas.clear_canvas(LAYER_SELECTED)
        else:
            self.draw_boxes(
                self.select.box.unsqueeze(0),
                layer=LAYER_SELECTED,
                cmap=DEFAULT_SELECTED_CMAP,
            )

    def draw_boxes(
        self,
        boxes: torch.Tensor,
        cmap: Optional[torch.Tensor] = None,
        layer: int = LAYER_BOXES,
    ):
        canvas = self._overlay_canvas.get_canvas(layer)
        with self._overlay_canvas.hold(canvas):  # batch the draw
            canvas.save()
            canvas.clear()
            boxes = boxes.detach().cpu()
            cmap = cmap if cmap is not None else self._cmap
            color = _resolve_colors(boxes[:, 5:], cmap)
            assert color.ndim == 2, f"invalid color tensor: {color}"

            canvas.line_width = SELECT_NODE_SIZE
            canvas.fill_style = "transparent"  # Explicitly transparent
            # TODO scale x and y?
            scale = self._overlay_canvas.size[0] / self._image_canvas.size[0]

            for i in range(boxes.shape[0]):
                box = (boxes[i, :4] + 0.5) * scale  # TODO handle angle
                x1, y1, x2, y2 = box.tolist()
                c = color[i].clone()
                canvas.stroke_style = color_to_hex(c)
                c[-1] = 90  # reduce alpha for fill
                canvas.fill_style = color_to_hex(c)
                canvas.begin_path()
                canvas.rect(x1, y1, x2 - x1, y2 - y1)
                canvas.fill()
                canvas.stroke()

            canvas.restore()

    @observe("boxes")
    def _on_boxes_change(self, change):
        self.draw_boxes(change["new"])

    @observe("keypoints")
    def _on_keypoints_change(self, change: dict):
        pass  # self.draw_keypoints(change)

    @observe("mask")
    def _on_mask_change(self, change):
        pass  # value = change["new"]

    @property
    def overlay_size(self) -> tuple[int, int]:
        return (
            int(self._overlay_canvas.size[0]),
            int(self._overlay_canvas.size[1]),
        )

    @property
    def image_size(self) -> tuple[int, int]:
        return (
            int(self._image_canvas.size[0]),
            int(self._image_canvas.size[1]),
        )


def _resolve_colors(cls: torch.Tensor, cmap: torch.Tensor):
    if cmap.ndim != 2:
        raise ValueError(
            f"Argument: `cmap` expected rgba colour tensor of shape [M,4] but got shape: {cmap.shape}"
        )
    if cmap.shape[-1] == 3:  # add alpha channel if it doesnt exist
        cmap = torch.cat([cmap, torch.ones_like(cmap[..., :1]) * 255], dim=-1)
    if cmap.shape[-1] != 4:
        raise ValueError(
            f"Argument: `cmap` expected rgba colour tensor of shape [M,4] but got shape: {cmap.shape}"
        )

    if cls.numel() == 0:  # no actual classes were provided
        # this cannot be set in the public api, something weird happended...
        assert cls.shape[1] == 0  # expected shape[N,0]
        # hmm... map all to the first class color...?
        # TODO maybe use a default color to minimise confusion?
        return cmap[:1].expand(cls.shape[0], -1)
    else:
        # print(cmap, cls)
        return cmap[cls.squeeze(-1).long()]
