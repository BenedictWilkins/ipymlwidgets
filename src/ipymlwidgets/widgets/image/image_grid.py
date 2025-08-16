from typing import Optional, Any, Iterable
import traitlets
import numpy as np
from PIL import Image
import math
from pathlib import Path

from traitlets import observe
from ipymlwidgets.traits.tensor import SupportedTensor, Tensor as TTensor
from ipymlwidgets.widgets.canvas.canvas import Canvas
from ipymlwidgets.image_utils import resize_image_letterbox, to_numpy_image
from ipymlwidgets.widgets.canvas.canvas import color_to_hex

class ImageGrid(Canvas):
    """A widget that displays a grid of images and handles click selection."""

    # List of images to display
    _images = traitlets.List(TTensor(allow_none=True)).tag(sync=False)
    
    # Size of each cell in the grid
    cell_size = traitlets.Tuple(traitlets.Int(), traitlets.Int(), default_value=(128, 128)).tag(sync=True)
    
    columns = traitlets.Int(default_value=16).tag(sync=True)
    rows = traitlets.Int(default_value=16).tag(sync=True)
    
    selection = traitlets.Set(traitlets.Int(), default_value=set()).tag(sync=False)

    def __init__(
        self, 
        images: Optional[list[SupportedTensor]] = None,
        cell_size: tuple[int, int] = (128, 128),
        columns: int = 16,
        rows: int = 16, # page!
        **kwargs
    ):
        """Initialize the image grid selector widget.
        
        Args:
            images (list[SupportedTensor], optional): List of images to display. Defaults to None.
            cell_size (tuple[int, int]): Size of each grid cell (width, height). Defaults to (128, 128).
            columns (int): Number of columns in the grid. Defaults to 4.
            **kwargs: Additional arguments passed to Canvas.
        """
        images = images or []
        
        # Calculate total canvas size
        total_width = columns * cell_size[0]
        total_height = rows * cell_size[1]
        
        super().__init__(
            size=(total_width, total_height),
            cell_size=cell_size,
            columns=columns,
            rows=rows,
            layers=2,
            **kwargs
        )
        self._images = images
        # this is a mirror of images, but contains resized images!
        self._images_cache = [self._new_cached_image(i) for i in range(len(images))]
        if self._images:
            self._render_grid()

        self._selection_fill_color = color_to_hex((144, 238, 144, 100)) # Light green
        self._selection_stroke_color = color_to_hex((144, 238, 144, 0))
        with self.hold_repaint(layer=1):
            # this is probably static?
            self.fill_color = self._selection_fill_color
            self.stroke_color = self._selection_stroke_color

        self._clear_selection = False
    
    def clear_selection(self):
        self.selection = set()
        self.clear(layer=1)

    def get_selected(self) -> list[SupportedTensor]:
        return [self._images[i] for i in self.selection]

    @property
    def grid_size(self):
        return self.columns * self.rows

    def _new_cached_image(self, index: int) -> np.ndarray:
        """Create a new cached image."""
        image = self._images[index]
        image = to_numpy_image(image)
        image = resize_image_letterbox(image, self.cell_size)
        return image

    def clear_all(self):
        self._images.clear()
        self._images_cache.clear()
        self._render_grid()

    def extend(self, images: Iterable[SupportedTensor]):
        """Extend the grid with a list of images."""
        for image in images:
            self._images.append(image)
            self._images_cache.append(self._new_cached_image(-1))
        self._render_grid()

    def insert(self, index: int, image: SupportedTensor):
        """Insert an image at the specified index."""
        self._images.insert(index, image)
        # this image is the one that will be rendered
        self._images_cache.insert(index, self._new_cached_image(index))
        self._render_grid()

    def append(self, image: SupportedTensor):
        """Add an image to the end of the grid."""
        self._images.append(image)
        self._images_cache.append(self._new_cached_image(-1))
        self._render_image_at_index(-1)
    
    def remove(self, index: int | list[int] | set[int]):
        """Remove image(s) from the grid."""
        if isinstance(index, int):
            index = [index]
        for i in sorted(index, reverse=True):
            self._images.pop(i)
            self._images_cache.pop(i)
        self._render_grid()

    @observe("mouse_down")
    def _on_mouse_down(self, event: dict):
        if len(self._images) == 0:
            return
        index = event['new']['y'] // self.cell_size[1] * self.columns + event['new']['x'] // self.cell_size[0]
        if index > len(self._images):
            return
        if index in self.selection:
            self.selection = self.selection - {index}
            self._clear_selection = True
        else:
            self.selection = self.selection | {index}
            self._clear_selection = False
        self._render_selected_at_index(index)
    
    @observe("mouse_drag")
    def _on_mouse_drag(self, event: dict):
        if len(self._images) == 0:
            return
        index = event['new']['y'] // self.cell_size[1] * self.columns + event['new']['x'] // self.cell_size[0]
        if index > len(self._images):
            return
        if index in self.selection:
            if self._clear_selection:
                self.selection = self.selection - {index}
                return self._render_selected_at_index(index)
        else:
            if not self._clear_selection:
                self.selection = self.selection | {index}
                self._render_selected_at_index(index)
        
    def _render_selected_at_index(self, index: int):
        row = index // self.columns
        col = index % self.columns
        x = col * self.cell_size[0]
        y = row * self.cell_size[1]
        rect = (x, y, x + self.cell_size[0], y + self.cell_size[1])
        with self.hold_repaint(layer=1):
            self.clear_rect(rect)
            if not self._clear_selection:
                self.draw_rect(rect)

    def _render_selected(self):
        with self.hold_repaint(layer=1):
            self.clear()
            # just to be safe...
            # self.fill_color = self._selection_fill_color
            # self.stroke_color = self._selection_stroke_color
            # self.stroke_width = 10 # no border?
            for idx in self.selection:
                self._render_selected_at_index(idx)

    def _render_grid(self, start: int = 0, end: int = None):
        """Render all images in the grid."""
        with self.hold_repaint(layer=0):
            self.clear()
            end = end or min(self.grid_size, len(self._images))
            for idx in range(start, end):
                self._render_image_at_index(idx)
    
    def _render_image_at_index(self, index: int):
        """Render a single image at the specified grid index.
        
        Args:
            index (int): index of the image to render.
        """
        if index >= self.grid_size:
            raise ValueError(f"Index {index} is out of bounds for grid size {self.grid_size}.")

        index = index % self.grid_size
        image = self._images_cache[index]
       
        # Calculate grid position
        row = index // self.columns
        col = index % self.columns
        # Calculate pixel position
        x = col * self.cell_size[0]
        y = row * self.cell_size[1]
                
        # Draw the image on the canvas
        self.set_patch(x, y, self.cell_size[0], self.cell_size[1], image, layer=0)
        # TODO clear the patch?