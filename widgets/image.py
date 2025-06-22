"""Image widget that works seemlessly with torch tensors and exposes custom click event handler."""

import math
import io
import torch
import PIL.Image as PILImage
import PIL as pillow

import ipywidgets as W
import ipyevents
from typing import Optional, Callable, Any, Literal

from traitlets import Tuple, Instance, observe, Int, Unicode, validate, Any as TraitAny, TraitError, Set

from datautils.image import draw_poly, to_tensor, DEFAULT_FONT_PATH
from datautils.box import point_in_box
from datautils.types import Color
from datautils.types.convert import color_to_tuple, color_to_tensor
from datautils.image import box_crop_ragged_, downscale_to, make_image_grid


def image_placeholder(
    size: tuple[int, int] = (128, 128),
    color: tuple[int, int, int] = (220, 220, 220),
    text: Optional[str] = "IMAGE MISSING",
    return_tensors : Literal['im', 'pt'] = "im",
) -> PILImage.Image:
    font = pillow.ImageFont.truetype(DEFAULT_FONT_PATH, 18)
    img = pillow.Image.new("RGB", size, color)
    draw = pillow.ImageDraw.Draw(img)
    if text:
        text_size = draw.textsize(text, font=font)
        position = ((size[0] - text_size[0]) // 2, (size[1] - text_size[1]) // 2)
        draw.text(position, text, fill=(80, 80, 80), font=font)
    if return_tensors == "pt":
        return to_tensor(img)
    return img

def tensor_to_png_bytes(t: torch.Tensor) -> bytes:
    """Convert a CxHxW **or** HxWxC float / uint8 tensor to PNG bytes."""
    if t.ndim == 3 and t.shape[0] in {1, 3, 4}:           # CxHxW
        t = t.permute(1, 2, 0)                            # → HxWxC
    if t.ndim != 3 or t.shape[2] not in {1, 3, 4}:
        raise ValueError("Expected shape (H,W,C) or (C,H,W) with C in {1,3,4}")
    if t.dtype != torch.uint8:
        t = (t.clamp_(0, 1) * 255).byte()                  # normalise floats
    t = t.detach().cpu().numpy()
    img = PILImage.fromarray(t)
    buf = io.BytesIO()
    img.save(buf, format="png")
    return buf.getvalue()

class Image(W.Box):
    """A clickable image display that keeps its backing data as a torch.Tensor."""

    value  = Instance(torch.Tensor, allow_none=True, help="image buffer.")
    format = Unicode("png").tag(sync=True)

    def __init__(self, 
        image: Optional[torch.Tensor] = None,
        click_callback : Optional[Callable[[dict[str,Any]],None]] = None,
        **kwargs):
        """
        Args:
            image (torch.Tensor[C,H,W], optional): image tensor.
            **kwargs: Any other `ipywidget.Image` kwargs (e.g. `layout`).
        """
        self._img_widget = W.Image(format="png", **kwargs)
        super().__init__([self._img_widget], layout = W.Layout(width="100%", height="100%"))
        self._click_callback = click_callback
        self._click_event = ipyevents.Event(
            source=self._img_widget,
            watched_events=["click"],
            prevent_default_action=True
        )
        self._click_event.on_dom_event(self.handle_click)
        self.value = image

    @observe("value")
    def _on_change(self, changed):
        """Update when the torch tensor changes."""
        self.refresh(changed)

    @property
    def size(self) -> tuple[int,int]:
        if self.value is None:
            return (0,0)
        return (self.value.shape[-1], self.value.shape[-2])
    
    def handle_click(self, event: dict):
        """Translate DOM click coords to original tensor coords."""
        event = self._get_click_data(event)
        if self._click_callback:
            self._click_callback(event)
        
    def _get_click_data(self, event : dict) -> dict:
        w_dom, h_dom = event["boundingRectWidth"], event["boundingRectHeight"]
        x_dom, y_dom = event["relativeX"], event["relativeY"]
        w_orig, h_orig = self.size
        x_orig = int(round(x_dom / w_dom * w_orig))
        y_orig = int(round(y_dom / h_dom * h_orig))
        color = self[:,y_orig,x_orig]
        return dict(x = x_orig, y = y_orig, width = self.size[0], height = self.size[1], color = color) 

    def __getitem__(self, idx):
        """Access pixel data directly from the underlying image tensor."""
        return self.value[idx]
    
    def __setitem__(self, idx, value):
        """Set pixel(s) in-place and refresh display directly."""
        self.value[idx] = value # following torch api

    def refresh(self, changed): # dont forget to call this when you update via indexing!
        """Update the widget display from the current tensor."""
        if self.value is not None:
            self._img_widget.value = tensor_to_png_bytes(self.value)

class AnnotatedImage(Image):

    boxes = Instance(torch.Tensor, allow_none=True, default_value=None,
                     help="Tensor of shape (N, 4) in xyxy format")
    mask  = Instance(torch.Tensor, allow_none=True, default_value=None,
                     help="Mask tensor of shape (H, W)")
    
    def __init__(
        self,
        image: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        layout: Optional[W.Layout] = None,
        fill_color: Optional[Color] = None,
        line_color: Optional[Color] = None,
        line_width: Optional[int] = None,
        click_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        drop_annotation_on_change : bool = True, # TODO do we need this?
        **kwargs
    ):
        self._refreshing = False # flag that avoids nested calls to refresh
        self._drop_annotation_on_change = drop_annotation_on_change # whether to drop the current boxes if the image changes
        self.box_fill_color = fill_color
        self.box_line_color = line_color if line_color else color_to_tuple("red")
        _image_max = max(image.shape[-2:]) if image is not None else 0
        self.box_line_width = line_width if line_width else int(max(2, _image_max / 400))
        super().__init__(image, layout=layout, click_callback=click_callback, **kwargs)
        self.boxes = boxes
        self.mask = mask
        
    def _get_click_data(self, event: dict) -> dict:
        event = super()._get_click_data(event)
        x, y = event['x'], event['y']
        in_boxes = point_in_box(self.boxes, (x, y)) if self.boxes is not None else torch.empty((0,), dtype=torch.int64)
        event["in_boxes"] = in_boxes
        return event

    @observe("boxes")
    def _on_boxes_change(self, changed):
        """Update when the torch tensor changes."""
        self.refresh(changed)

    @observe("mask")
    def _on_mask_change(self, changed):
        """Update when the torch tensor changes."""
        self.refresh(changed)

    def drop_boxes(self):
        self.boxes = torch.empty((0,4))

    def refresh(self, changed : Any = None):    
        """Update when the torch tensor changes."""
        if self._refreshing:
            return # dont refresh again
        self._refreshing = True
        if changed is None:
            changed = dict()
        if self._drop_annotation_on_change and changed.get('name', None) == "value":
           self.drop_boxes() # the image changed this invalidates the current boxes

        value = self.value
        if self.boxes is not None and self.boxes.numel() > 0:
            # this will copy the value anyway
            value = draw_poly(value, self.boxes, fill_color=self.box_fill_color, line_color=self.box_line_color, line_width=self.box_line_width)
        if self.mask is not None:
            pass # TODO!
        self._img_widget.value = tensor_to_png_bytes(value)
        self._refreshing = False

class IconGrid(AnnotatedImage):
    
    # traitlets
    image = Instance(torch.Tensor, allow_none=True, help="the image that contains the boxed icons.")
    selected = Set(help="index of icons that are currently selected")
    nrow = Int(help="number of icons per row of the grid")
    cell_size = Tuple(Int(), Int(), help="size of each grid cell in pixels")
    background_color = TraitAny(help="background color for the icons if they are RGBA")

    def __init__(
        self,
        image: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        nrow : int = 16, # icons per row
        cell_size : tuple[int,int] | int = (32, 32),
        background_color : Optional[Color] = None,
        layout: Optional[W.Layout] = None,
        click_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        **kwargs
    ):
        if boxes is not None and image is None:
            raise ValueError("Argument: `image` must be provided if `boxes` is non-empty.")
        self._refreshing = False
        self.image = image
        self.nrow = nrow # how many icons to show per row of the grid
        self.cell_size = cell_size
        self.background_color = background_color if background_color else (0, 0, 0)
        super().__init__(None, boxes=boxes, layout=layout, click_callback=click_callback, _drop_boxes_on_change=False, **kwargs)
        self.selected = set() # set of currently selected icons, this will also initialise `mask`
        self._selection_mask_factor = 0.5 # used to darken selected icons

    @property
    def grid_size(self) -> tuple[int,int]:
        """Size of the grid (rows, cols)."""
        rows = math.ceil(self.boxes.shape[0] / self.nrow) 
        cols = min(self.nrow, self.boxes.shape[0])
        return (rows, cols)

    def _get_click_data(self, event: dict) -> dict:
        event = Image._get_click_data(self, event)
        event['cell_x'] = event['x'] // self.cell_size[0]
        event['cell_y'] = event['y'] // self.cell_size[1]
        event['index'] = event['cell_x'] + event['cell_y'] * self.nrow
        self.selected ^= {event['index']}
        self._on_selected_change({
            "name": "selected",             # the trait name
            "diff": {event['index']},       # set difference
            "owner": self,                  # the instance that changed
            "type": "change",               # always "change"
        })
        return event
    
    @observe("selected")
    def _on_selected_change(self, changed):
        self.refresh(changed)

    @validate("cell_size")
    def _validate_cell_size(self, proposal):
        value = proposal['value']
        print(proposal)
        if isinstance(value, (int, float)):
            return (int(value), int(value))
        else:
            assert len(value) == 2
            return tuple(value)
    
    @validate("background_color")
    def _validate_background_color(self, proposal):
        try:
            return color_to_tensor(proposal['value']) # RGB [0-1]
        except Exception as e:
            raise TraitError(f"Invalid colour: {proposal['value']}") from e

    def drop_boxes(self):
        super().drop_boxes()
        self.value = None # empty image...
        self._img_widget.value = tensor_to_png_bytes(self.value)
        self.drop_selected()

    def drop_selected(self):
        self.mask = torch.zeros((1, self.size[1], self.size[0]))
        self.selected = set() # also clear selected

    def _make_grid(self, image, boxes) -> torch.Tensor:
        print(boxes)
        crops = box_crop_ragged_(image, boxes)
        crops = downscale_to(crops, self.cell_size)
        # alpha blend with the background
        if crops.shape[1] == 4: # has alpha, blend with the background color
            crops = crops[:,:3] * crops[:,-3:] + self.background_color.view(1,3,1,1) * (1 - crops[:,-3:])
        return make_image_grid(crops, nrow=self.nrow, padding=0, return_tensors="pt")

    def refresh(self, changed : Any = None):    
        """Update when the torch tensor changes."""
        if self._refreshing:
            return # dont refresh again
        self._refreshing = True
        if changed is None:
            changed = dict()

        if changed.get('name', None) == "image":
            self.drop_boxes() # the image changed this invalidates the current boxes
        elif changed.get('name', None) == "boxes":
            # the selection mask is invalidated if the boxes change
            if self.boxes is not None and self.boxes.numel() > 0:
                # extract from boxes and make a grid
                self.value = self._make_grid(self.image, self.boxes)
                self.drop_selected() # dont apply the mask, it is zero
                self._img_widget.value = tensor_to_png_bytes(self.value)
            else: # boxes are now empty... remove all icons
                self.drop_boxes()
        elif changed.get('name', None) == "selected":
            diff = set()
            if not 'diff' in changed:
                self.mask = torch.zeros((1, self.size[1], self.size[0]))
                diff = self.selected
            else:
                diff = changed['diff']
            for indx in diff:
                row, col = self.cell_size[0] * (indx // self.nrow),  self.cell_size[1] * (indx % self.nrow)
                print(indx, row, col)
                self.mask[:,row:row+self.cell_size[0],col:col+self.cell_size[1]] = 1 - self.mask[:,row:row+self.cell_size[0],col:col+self.cell_size[1]]
                # update the value if the mask changed, darken the selected
                value = self.value * (1 - self.mask * (1 - self._selection_mask_factor))
                self._img_widget.value = tensor_to_png_bytes(value)

        self._refreshing = False    


    