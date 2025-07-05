import ipywidgets as W

from ipymlwidgets.widgets.image import Image, _STYLE_HTML_LAYOUT
from ipymlwidgets.widgets.box_overlay import BoxOverlay, BoxSelection
from ipymlwidgets.traits import Tensor, link, dlink
from traitlets import Instance
from typing import Optional
import torch


DEFAULT_STROKE_COLOR = (255, 0, 0, 255)
DEFAULT_STROKE_WIDTH = 2
DEFAULT_SELECTED_CMAP = torch.tensor((255, 0, 0, 255)).unsqueeze(0)

# 5 pixels in the client...?
# TODO this must depend on the client size resolution... use the mouse event to do it?
SELECT_NODE_SIZE = 5

LAYER_BOXES = 0
LAYER_KEYPOINTS = 1
LAYER_MASK = 2
LAYER_SELECTED = 3

_OVERLAY_STYLE = """
<style>
.overlay-image {
    position: absolute;
    top: 0;
    left: 0;
}
</style>
"""


class ImageAnnotated(W.Box):

    image = Tensor(allow_none=True).tag(sync=False)
    boxes = Tensor(allow_none=True).tag(sync=False)
    keypoints = Tensor(allow_none=True).tag(sync=False)
    mask = Tensor(allow_none=True).tag(sync=False)

    selection = Instance(BoxSelection, allow_none=True, default_value=None).tag(
        sync=False
    )

    def __init__(
        self,
        image,
        boxes: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cmap: Optional[torch.Tensor] = None,
        display_size: tuple[int, int] = (320, 320),
    ):
        # empty image, will be set by trailet link
        self._image_canvas = Image()
        self._image_canvas.add_class("overlay-image")
        self.image = image  # set the image here and in self._image_canvas
        # link the image trait (image <-> overlay)
        link((self, "image"), (self._image_canvas, "image"))

        self._overlay_box = BoxOverlay(None, overlay_size=display_size)
        self.boxes = BoxOverlay.validate_boxes(boxes, self._image_canvas.size)
        # link the selection trait (overlay -> image)
        dlink((self._overlay_box, "selection"), (self, "selection"))
        # link the boxes trait (image <-> overlay)
        link(
            (self, "boxes"),
            (self._overlay_box, "boxes"),
            transform=(self._boxes_image_to_overlay, self._boxes_overlay_to_image),
        )
        self._overlay_keypoints = None
        self._overlay_mask = None
        size = (f"{display_size[0]}px", f"{display_size[1]}px")
        super().__init__(
            [
                # this style stacks the overlays on top of each other
                W.Box(
                    [
                        W.HTML(_OVERLAY_STYLE, layout=_STYLE_HTML_LAYOUT),
                        self._image_canvas,
                        self._overlay_box,
                        # self._overlay_keypoints,
                        # self._overlay_mask,
                    ],
                    layout=W.Layout(width=size[0], height=size[1]),
                ),
            ],
        )

    def add_child(self, child: W.Widget):
        self.children = self.children + (child,)

    @property
    def size(self):
        return self._image_canvas.size

    def _boxes_overlay_to_image(self, boxes: Optional[torch.Tensor]):
        """Transform boxes coming from the overlay to the image size"""
        if boxes is None:
            return None
        boxes = boxes.clone()
        xscale = self._image_canvas.size[0] / self._overlay_box.size[0]
        yscale = self._image_canvas.size[1] / self._overlay_box.size[1]
        boxes[:, :4] *= torch.tensor([xscale, yscale, xscale, yscale])
        return boxes

    def _boxes_image_to_overlay(self, boxes: Optional[torch.Tensor]):
        """Transform boxes coming from the image to the overlay size"""
        if boxes is None:
            return None
        boxes = boxes.clone()
        xscale = self._overlay_box.size[0] / self._image_canvas.size[0]
        yscale = self._overlay_box.size[1] / self._image_canvas.size[1]
        boxes[:, :4] *= torch.tensor([xscale, yscale, xscale, yscale])
        return boxes
