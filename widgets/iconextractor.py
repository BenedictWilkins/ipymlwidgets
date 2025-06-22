"""Widget for extracting icon from icon sheets.

Icon sheets are a specific kind of sprite sheet that typically have "gaps" between icons (i.e. the sprites are not full "tiles"). Icon may not be arranged into equal sized blocks as is commonly the case for tilesheets.
"""

import ipywidgets as W
from typing import Any, Optional
from pathlib import Path

import torch

from datautils.widgets import Swatch, AnnotatedImage, IconGrid
from datautils.image import load, iconsheet_extract, downscale_to


class IconExtractor(W.Box):

    def __init__(self, files_or_dir : list[Path] | Path, out_dir : Optional[Path] = None, **kwargs):
        self._loader = load(
            files_or_dir, 
            extensions=[".png", ".jpeg", ".jpg", ".webp", ".tiff"], 
            convert_mode="RGB",
            show_progress=False, 
            return_files=True,
            return_tensors="pt")

        self._out_dir = out_dir if out_dir else Path("./icons/").resolve() # path to save any outputs
        self._out_dir.mkdir(parents=True, exist_ok=True) #TODO ?? hmmm
        self._btn_save = W.Button(description="Save")
        self._btn_save.on_click(self.on_save_click)
        self._btn_next = W.Button(description="Next")
        self._btn_next.on_click(self.on_next_click)

        # color picker for the background of the image
        self._swatch = Swatch("#ffffff") # default to white...?

        
        # load the initial image
        image, path = self._next_image()

        image = image.cuda() # TODO if cuda is not avaliable, we need to fallback to skimage region proposal!
        image = downscale_to([image], (1920,1080))[0]

        self._file_text = W.Text(value=path.resolve().as_posix(), disabled=True)
        # params for icon extraction function
        self._dilation = W.IntSlider(value=0, min=0, max=50, step=1, description="dilation")
        self._tolerance = W.IntSlider(value=0, min=0, max=255, step=1, description="tolerance")

        # params for background removal
        self._threshold_transparency = W.IntSlider(value=0, min=0, max=255, step=1, description="transparency threshold")
        self._threshold_opacity = W.IntSlider(value=0, min=0, max=255, step=1, description="opacity threshold")
        self._mode = W.ToggleButtons(
            options=[('max', 'max'), ('l2', 'euclidean')],  # label → value
            value='euclidean',                              # default
            description='mode'
        )
        self._annotated_image = AnnotatedImage(image, layout=W.Layout(max_width="50%", flex="0 0 auto", width="auto"))
        self._icon_grid = IconGrid(image, layout=W.Layout(max_width="50%", flex="1 1 auto", width="auto"))
        
        self._controls_alpha = [self._threshold_transparency, self._threshold_opacity, self._mode]
        self._controls_blobs = [self._dilation, self._tolerance]
        controls = W.VBox([
            W.HBox([self._file_text, self._btn_next, self._btn_save]),
            W.HBox([W.VBox(self._controls_blobs + self._controls_alpha), 
                    W.VBox([self._swatch]) ]),
        ])
        all = W.VBox([
            controls, 
            W.HBox([self._annotated_image, self._icon_grid], layout=W.Layout(align_items = "flex-start")) 
        ])
        
        super().__init__([all], **kwargs)

        for control in self._controls_alpha:
            control.observe(self.refresh_alpha, "value")
        for control in self._controls_blobs:
            control.observe(self.refresh_blobs, "value")

    def _next_image(self) -> tuple[torch.Tensor, Path]: 
        return next(self._loader)

    def refresh_blobs(self, changed : Any = None):
        # do i need to de-bounce?
        print(changed, self._icon_grid.background_color)
        result = iconsheet_extract(
            self._annotated_image.value, # backing buffer, 
            background_color=self._icon_grid.background_color,
            dilation=self._dilation.value,
            return_boxes=True, # just need the boxes, the backing tensor is the same
            return_icons=False,
            #return_connected_components=True,
            #return_foreground_mask=True
        )
        self._annotated_image.boxes = result.boxes
        self._icon_grid.boxes = result.boxes
    
    def refresh_alpha(self, changed : Any = None):
        pass 

    def on_save_click(self, b):
        pass 
        # image_file = Path(files[int(file_idx.value)]) 
        # icons = getattr(out, "icons_cache", [])
        # ignore = [row + col * GRID_NROW for (row, col) in out.icons_selected]
        # icons = [icon for i, icon in enumerate(icons) if not i in ignore]
        # save(image_file, icons, out_path)
        # with open(out_cache_done.as_posix(), "a") as f:
        #     f.write(image_file.expanduser().resolve().as_posix())
        #     f.write("\n")
        
        # file_idx.value = (file_idx.value + 1) % len(files) #saved!
        # blob_dist_sl.value = min(Image.open(files[file_idx.value]).size)//160
        # refresh()

    def on_next_click(self, b):
        pass 
        # image_file = Path(files[int(file_idx.value)]) 
        # with open(out_cache_skip.as_posix(), "a") as f:
        #     f.write(image_file.expanduser().resolve().as_posix())
        #     f.write("\n")
        # file_idx.value = (file_idx.value + 1) % len(files)
        # blob_dist_sl.value = min(Image.open(files[file_idx.value]).size)//160
        # refresh()

    # def refresh(self, changed : Any = None):
    #     pass 


