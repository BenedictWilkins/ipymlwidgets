from ipymlwidgets.widgets.image import Image
from traitlets import Instance, observe
from typing import Optional

import torch

DEFAULT_STROKE_COLOR = (255, 0, 0, 255)
DEFAULT_STROKE_WIDTH = 2
from ipymlwidgets.utils import color_to_hex


class ImageAnnotated(Image):

    boxes = Instance(torch.Tensor, allow_none=True).tag(sync=False)
    keypoints = Instance(torch.Tensor, allow_none=True).tag(sync=False)
    mask = Instance(torch.Tensor, allow_none=True).tag(sync=False)

    def __init__(
        self,
        image: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(image=image, layers=4, **kwargs)
        # self.image = image layer 0
        self.boxes = boxes  # layer 1
        self.keypoints = keypoints  # layer 2
        self.mask = mask  # layer 3

    @observe("boxes")
    def _on_boxes_change(self, change):
        self.draw_boxes(change)

    @observe("keypoints")
    def _on_keypoints_change(self, change: dict):
        self.draw_keypoints(change)

    @observe("mask")
    def _on_mask_change(self, change):
        value = change["new"]
        if value.ndim == 2:
            pass
        elif value.ndim == 3:
            pass
        else:
            raise ValueError(f"Mask must have 2 or 3 dimensions, got {value.ndim}")

    def _resolve_color(self, cls: torch.Tensor, cmap: Optional[torch.Tensor] = None):
        if cmap is None:
            cmap = torch.tensor([DEFAULT_STROKE_COLOR], dtype=torch.uint8)  # [1, 4]
        if cmap.shape[-1] == 3:
            cmap = torch.cat([cmap, torch.ones_like(cmap[..., :1]) * 255], dim=-1)
        if cls.numel() == 0:
            return cmap[:1].expand(cls.shape[0], -1)
        else:
            return cmap[cls.squeeze(-1)]

    def draw_keypoints(
        self,
        change: dict,
        cmap: Optional[torch.Tensor] = None,
        radius: float = 4,
    ):
        """Draw circles on the canvas at the keypoint locations."""
        canvas = self._canvas[2]  # keypoint canvas
        canvas.clear()
        canvas.save()

        keypoints = change["new"].detach().cpu()
        cmap = change.get("cmap", None) if cmap is None else cmap
        color = self._resolve_color(keypoints[:, 2:], cmap)

        for i in range(keypoints.shape[0]):
            x, y = keypoints[i, :2]
            c = color[i]
            canvas.stroke_style = color_to_hex(c)
            c[-1] = int(c[-1] * 0.2)  # reduce alpha for fill
            canvas.fill_style = color_to_hex(c)

            canvas.begin_path()
            canvas.arc(x.item(), y.item(), radius, 0, 2 * torch.pi)
            canvas.fill()
            canvas.stroke()
        canvas.restore()

    def draw_boxes(
        self,
        change: dict,
        cmap: Optional[torch.Tensor] = None,
    ):
        canvas = self._canvas[1]  # boxes canvas
        canvas.clear()

        boxes = change["new"].detach().cpu()
        cmap = change.get("cmap", None) if cmap is None else cmap
        color = self._resolve_color(boxes[:, 5:], cmap)
        canvas.line_width = 1
        canvas.fill_style = "transparent"  # Explicitly transparent

        for i in range(boxes.shape[0]):
            box = boxes[i, :4] + 0.5  # TODO handle the angle
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
