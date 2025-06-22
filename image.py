"""Utilities for images."""

from typing import Literal, Optional, Iterable, Generator
from itertools import chain
from tqdm import tqdm

import torch
import kornia as K
import PIL as pillow
import PIL.Image as _PILImage

import torchvision.transforms.v2 as T
from torchvision.utils import make_grid

from datautils.font import DEFAULT_FONT_PATH
from datautils.types import Image, ImageBatch, Size, Color, Namespace
from datautils.box import box_to_poly, connected_components, connected_components_to_boxes
from datautils.types.convert import image_to_pil as to_pil, image_to_tensor as to_tensor, color_to_tensor, color_to_tuple, color_with_alpha


# these will be added to this image module, they are just torchscript optimised image utilities
from datautils.jit.image import (
    downscale_to,
    resize_to,
    box_crop,
    box_crop_ragged,
    box_crop_ragged_,
)

from pathlib import Path

__all__ = (
    "load",
    # from datautils.types.convert
    "to_pil",
    "to_tensor",
    # from datautils.jit.image (torchscript optimised)
    "downscale_to",
    "resize_to",
    "box_crop",
    "box_crop_ragged",
    "box_crop_ragged_",
)

def tilesheet_background_color(image : torch.Tensor, color : Optional[Color] = None, mode : Literal["median", "corner1", "corner4"] = "median") -> torch.Tensor:
    if color:
        return color_to_tensor(color)
    if mode == "median":
        # pretty common for the background to be the most common color in a tile sheet
        return torch.median(image.view(image.shape[0], -1), axis=0)
    elif mode == "corner1":
        return image[:,0,0].clone()
    elif mode == "corner4":
        return (image[:,0,0] + image[:,0,-1] + image[:,-1,0] + image[:,-1,0]) / 4.0
    else:
        raise ValueError(f'Argument: `mode` expected one of ["median", "corner1", "corner4"] but got {mode}')

def iconsheet_extract(
    iconsheet : torch.Tensor,
    background_color : torch.Tensor,
    tolerance : int = 0,
    dilation : int | tuple[int,int] = 0, 
    return_boxes : bool = True,
    return_icons : bool = True,
    return_foreground_mask : bool = False,
    return_connected_components : bool = False,
) -> Namespace:  
    # TODO validate arguments
    diff = torch.norm(iconsheet[:3] - background_color[:3,None,None].to(iconsheet.device), dim=0, p=2)
    fg_mask = (diff > tolerance).float().unsqueeze(0).unsqueeze(0) #BCHW
    print(fg_mask.shape)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if min(dilation) > 0:
        kernel = torch.ones(dilation, device=iconsheet.device)
        fg_mask = K.morphology.dilation(fg_mask,  kernel)
    # region proposal
    ccs = connected_components(fg_mask, tile_size=256).squeeze(0)
    print(ccs.shape)
    boxes = connected_components_to_boxes(ccs)
    result = dict()
    if return_boxes:
        result['boxes'] = boxes
    if return_icons:
        result['icons'] = box_crop_ragged(iconsheet, boxes)
    if return_connected_components:
        result['connect_components'] = ccs
    if return_foreground_mask:
        result['foreground_mask'] = fg_mask.squeeze(0)
    return Namespace(**result)

def load(
    files_or_dir: str | Path | Iterable[str | Path],
    extensions: Optional[list[str]] = None,
    convert_mode: Optional[str] = "RGB",
    show_progress: bool = True,
    return_files: bool = False,
    return_tensors: Literal["im", "pt"] = "im",
) -> Generator[torch.Tensor, None, None] | Generator[_PILImage.Image, None, None]:
    """Load images from files or a directory.

    Args:
        files_or_dir (str | Path | Iterable[str  |  Path]): image file(s) or directory to search
        extensions (Optional[list[str]], optional): extensions to include. Defaults to ["jpg", "jpeg", "png", "webp", "tiff"].
        convert_mode (Optional[str], optional): convert image to RGBA,RGB,L or no conversion (None). Defaults to "RGB".
        show_progress (bool, optional): Whether to show a progress bar during. Defaults to True.
        return_files (bool, optional): whether to return the file paths as well as the images. Defaults to False.
        return_tensors (Literal['im', 'pt'], optional): the return format of the image: 'im' for pillow images, 'pt' for torch tensors in float32 [0-1] CHW format. Defaults to 'im'.

    Yields:
        Generator[torch.Tensor, None, None] | Generator[_PILImage.Image, None, None]: loaded images
    """

    def _load(files_or_dir: str | Path | Iterable[str | Path]):
        if isinstance(files_or_dir, str):
            files_or_dir = Path(files_or_dir)

        if isinstance(files_or_dir, Path):
            files_or_dir = files_or_dir.expanduser().resolve()
            if not files_or_dir.exists():  # does it exist?
                raise FileNotFoundError(files_or_dir.as_posix())
            # load the file with PIL
            if files_or_dir.is_file():
                image = _PILImage.open(files_or_dir)
                if convert_mode is not None:
                    image = image.convert(convert_mode)
                if return_tensors == "pt":
                    image = to_tensor(image)
                if not return_files:
                    yield image
                else:
                    yield image, files_or_dir
            else:  # is an image file
                assert files_or_dir.is_dir()
                exts = extensions  # outer variable
                if exts is None:
                    exts = [
                        "jpg",
                        "jpeg",
                        "png",
                        "webp",
                        "tiff",
                    ]  # TODO any others supported by pil?
                ext_patterns = [
                    ext.replace(".", "").replace("*", "").lower().strip()
                    for ext in exts
                ]
                ext_patterns = ["*." + ext for ext in ext_patterns]  # glob patterns
                all_files = chain(files_or_dir.glob(extp) for extp in ext_patterns)
                yield from (f for file in all_files for f in _load(file))
        elif isinstance(files_or_dir, Iterable):
            yield from (f for file in files_or_dir for f in _load(file))
        else:
            raise ValueError(
                f"Argument: Expected `files_or_dir` to a file-like type but got {type(files_or_dir)}"
            )

    if show_progress:
        yield from tqdm(_load(files_or_dir))
    else:
        yield from _load(files_or_dir)

def make_image_grid(
    images: ImageBatch,
    nrow: int = 8,
    padding: int = 2,
    pad_value: float = 0,
    return_tensors: Literal["im", "pt"] = "im",
) -> pillow.Image.Image | torch.Tensor:
    """Make a grid of images. If the images have differing sizes they will be center padded to match.

    Args:
        images (ImageBatch): images to make a grid with.
        nrow (int, optional): number of images per row. Defaults to 8.
        padding (int, optional): amount of padding between each value to use. Defaults to 2.
        pad_value (float, optional): value of padding to use. Defaults to 0.
        return_tensors (Literal['img', 'pt'], optional): the image return type.

    Returns:
        pillow.Image.Image | torch.Tensor: the grid of images, see argument `return_tensors`.
    """
    if isinstance(images, torch.Tensor):
        if images.ndim == 3:
            result = images  # ??? error?
        elif images.ndim == 4:
            result = make_grid(images, nrow=nrow, padding=padding, pad_value=pad_value)
    elif isinstance(images, list):
        # make sure all images are the same size by center padding them to the max size
        sizes = get_image_sizes(images)
        max_size = sizes.amax(dim=-2)
        max_size = max_size.view(2)  # deal with any singleton leading dims
        max_width, max_height = max_size[0].item(), max_size[0].item()
        _pad = (max_width // 2 + 1, max_height // 2 + 1)
        # TODO this is problematict use a different approach (e.g. downscale_to?)
        pad_transform = T.Compose(
            [
                T.Pad(_pad, fill=pad_value),
                T.CenterCrop((max_height, max_width)),
            ]
        )
        images = [pad_transform(to_tensor(image)) for image in images]
        result = make_grid(images, nrow=nrow, padding=padding, pad_value=pad_value)
    else:
        raise ValueError(
            f"Expected list of images or batched image tensor but got: {type(images)}"
        )
    if return_tensors == "im":
        return to_pil(result)
    elif return_tensors == "pt":
        return result
    else:
        raise ValueError(
            f"Argument `return_tensors` must be one of ['im', 'pt'], got {return_tensors}"
        )

# TODO torchvision.transforms.v2.functional actually has `get_dimensions` which works with PIL images or torch tensors, we should make use of this here to clean things up.
def get_image_sizes(
    sizes_or_images: Size | ImageBatch,
    error_on_ambiguity: bool = True,
) -> torch.Tensor:
    """Get standardized sizes for images as a tensor of shape [..., 2] where each row in the last dimension contains (width, height).

    This function does the following for different input types:
        - list[Tensor[C,H,W] | pillow.Image.Image]: Get the sizes of images in a list as a tensor of shape [N, 2].
        - Tensor[...,C,H,W]: Get the size of a tensor of images while maintaining the batch dimension i.e. [..., C, H, W] -> [..., 2]
        - list[iterable[2]]: Get sizes from `image_sizes` provided in various formats (list, tuple, etc) as a tensor of shape [N, 2].
        - Tensor[N,2]: returns the tensor unaltered (already in a standardized format).
        - Tensor[C,H,W] | pillow.Image.Image: Get the size of the image as a tensor of shape [1, 2].
        - Tensor[..., M, N, 2]]: **GOTCHA WARNING** the tensor will be interpreted as a batch of images with width 2, see `error_on_ambiguity`. If you already have a size tensor of shape [..., 2] you don't need this function!

    Args:
        sizes_or_images (IMAGE_INPUT | SIZE): The sizes to standardize or the images to get the size of.
        error_on_ambiguity (bool): Whether to throw an error if there is a `sizes_or_images` shape ambiguity (attempts to prevent miss-use of this function). Defaults to True.

    Returns:
        torch.Tensor: sizes of shape [..., 2]
    """
    if isinstance(sizes_or_images, pillow.Image.Image):
        return torch.tensor(tuple(sizes_or_images.size)).unsqueeze(0)  # [1, 2]
    elif isinstance(sizes_or_images, torch.Tensor):
        if sizes_or_images.shape[-1] == 2:  # assume this is a size tensor
            if sizes_or_images.ndim == 1:
                return sizes_or_images.unsqueeze(0)  # [1, 2]
            elif sizes_or_images.ndim == 2:
                return sizes_or_images  # [..., 2]
            elif (
                error_on_ambiguity
            ):  # this could be an image (width=2) or a size tensor...
                raise ValueError(
                    "`sizes_or_images` tensor is ambiguous, set `error_on_ambiguity=False` if you provided an images tensor."
                )
        # assume this is an image tensor
        if sizes_or_images.ndim < 3:
            raise ValueError(
                f"Expected size tensor of shape [..., 2] or image tensor of shape [..., C, H, W] but got {tuple(sizes_or_images.shape)}"
            )
        elif sizes_or_images.ndim == 3:
            hw = sizes_or_images.shape[-2:]  # assumed CHW!
            return torch.tensor((hw[1], hw[0])).unsqueeze(0)  # [1, 2]
        else:  # assume images tensor was provided! [..., C, H, W]
            hw = sizes_or_images.shape[-2:]  # assumed CHW!
            return torch.tensor((hw[1], hw[0])).broadcast_to(
                (*sizes_or_images.shape[:-3], 2)
            )  # [..., 2]

    elif isinstance(sizes_or_images, (tuple, list)):
        if isinstance(sizes_or_images[0], (int, float)):
            if len(sizes_or_images) == 2:
                return torch.tensor(tuple(sizes_or_images)).unsqueeze(0)  # [1, 2]
            else:
                raise ValueError(
                    f"Expected size tuple/list of length 2, got {sizes_or_images}"
                )
        else:
            return torch.cat([get_image_sizes(sori) for sori in sizes_or_images])
    else:
        raise ValueError(
            f"Expected pillow Image, torch.Tensor, tuple or list, got {type(sizes_or_images)}"
        )

def draw_ocr(
    image: pillow.Image.Image,
    boxes: torch.Tensor,
    texts: list[str],
    background_color: str = "white",
    font_path: Optional[str] = None,
    font_color: str | tuple = "black",
    line_width: int = 1,
    line_color: str = "grey",
    fill_color: str | tuple = "green",
    _fill_alpha: float = 0.4,
    arrange_vertical: bool = False,
):
    """Visualise an OCR result.

    Args:
        image (pillow.Image.Image): image to visualise (had OCR performed on it).
        boxes (torch.Tensor): OCR boxes
        texts (list[str]): OCR
        background_color (str, optional): Background colour of the recognised text image. Defaults to "white".
        font_path (Optional[str], optional): Path to font of the recognised text (to render). Defaults to None.
        font_color (str, optional): Color of recognised text. Defaults to "black".
        line_width (int, optional): Width of box lines. Defaults to 1.
        line_color (str | tuple, optional): Colour of box lines. Defaults to "grey".
        fill_color (str | tuple, optional): Color of the box fill. Defaults to "red".
        _fill_alpha (float, optional): Alpha for the box fill. Defaults to 0.2.
        arrange_vertical (bool, optional): _description_. Defaults to False.

    Returns:
        pillow.Image.Image: visualised OCR result.
    """
    (w, h) = image.size
    pos = (0, h) if arrange_vertical else (w, 0)
    siz = (w * (1 + (1 - arrange_vertical)), h * (1 + arrange_vertical))
    out_img = pillow.Image.new("RGB", siz, "white")
    out_img.paste(
        draw_poly(
            image,
            boxes,
            line_color=fill_color,
            fill_color=fill_color,
            fill_alpha=_fill_alpha,
        ),
        (0, 0),
    )
    out_img.paste(
        draw_text_boxes(
            boxes,
            texts,
            (w, h),
            line_width=line_width,
            line_color=line_color,
            background_color=background_color,
            font_path=font_path,
            font_color=font_color,
        ),
        pos,
    )
    return out_img

def draw_text_boxes(
    boxes: torch.Tensor,
    texts: list[str],
    image_or_size: pillow.Image.Image | tuple[int, int],
    background_color: str = "white",
    font_path: Optional[str] = None,
    font_min_size: int = 14,
    font_color: str = "black",
    line_width: int = 1,
    line_color: str = "grey",
):
    """Draw text inside boxes on a new blank image - useful for visualising OCR results.

    Args:
        boxes (torch.Tensor[N,4]): Boxes in xyxy format image coordinate space.
        texts (list[str]): text for each box
        image_or_size (pillow.Image.Image | tuple[int,int]): A reference image or its size (width, height).
        background_color (str): Background color of the image. Defaults to white.
        font_path (str, optional): Path to .ttf font (defaults to roboto).
        font_min_size (int): Minimum font size for the text (in pixels).
        font_color (str): Color of the font. Defaults to black.
        line_width (int): Width of the box lines. Defaults to 1.
        line_color (str): Color of the box lines. Defaults to grey.

    Returns:
        pillow.Image.Image: Image with drawn text in boxes.
    """
    if boxes.shape[0] != len(texts):
        raise ValueError(
            f"Argument: `boxes` and `texts` have miss-matched shapes {boxes.shape[0]} != {len(texts)}"
        )
    if isinstance(image_or_size, pillow.Image.Image):
        image_size = image_or_size.size
    elif isinstance(image_or_size, (list, tuple)) and len(image_or_size) == 2:
        image_size = image_or_size
    # let handle bad inputs...
    img = pillow.Image.new("RGB", image_size, background_color)
    draw = pillow.ImageDraw.Draw(img)
    font_size_step = 4  # how often to change the font size
    hoff = 4  # text shouldnt expand to fill the entire box (it gives overlapping text otherwise)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1) - hoff
    font_sizes = torch.clamp(
        (heights // font_size_step) * font_size_step, min=font_min_size
    ).to(torch.int)
    if font_path is None:
        font_path = DEFAULT_FONT_PATH
    # Preload unique font sizes
    font_cache = {}
    for size in font_sizes.unique().tolist():
        font_cache[size] = pillow.ImageFont.truetype(font_path, size)

    for box, text, font_size in zip(boxes, texts, font_sizes.tolist()):
        x1, y1, x2, y2 = box.tolist()
        font = font_cache[font_size] if font_path else pillow.ImageFont.load_default()
        draw.rectangle([x1, y1, x2, y2], outline=line_color, width=line_width)
        draw.text((x1, y1 + hoff / 2), text, fill=font_color, font=font)
    return img

def draw_poly(
    image: Image,
    polygons: torch.Tensor,
    labels : Optional[torch.Tensor] = None,
    fill_color: Optional[Color] = None,
    line_color: Optional[Color] = "red",
    line_width: Optional[int] = None,
) -> Image:
    """Draws polygons on an image. Polygons must be in flat format [x1, y1, x2, y2, ...].

    Args:
        image (Image): image to annotate
        polygons (torch.Tensor[N, M]): polygons to draw.
        labels (torch.Tensor[N]): label for each polygon - determines which `fill_color` and `line_color` to use.
        fill_color (Optional[Color | list[Color]], optional): color(s) of the polygon fill. Defaults to no fill.
        line_color (Optional[Color | list[Color]], optional): color(s) of polygon lines. Defaults to red.
        line_width (float, optional): Width of the line in pixels. Defaults to 2 or more to match image size.

    Returns:
        Image: image with the polygons drawn.
    """
    if labels is not None:
        if labels.shape[0] != polygons.shape[0]:
            raise ValueError(f"Argument: `labels` expected initial dimension {polygons.shape} but got {labels.shape}")
    if torch.is_tensor(polygons):
        polygons = polygons.detach().float().cpu()
    else:
        raise ValueError(
            f"Argument: `polygons` expected to be of type torch.Tensor but got {type(polygons)}"
        )
    if polygons.shape[-1] == 4:
        polygons = box_to_poly(polygons)  # assume these are bounding boxes...
    if polygons.ndim == 1:
        polygons.unsqueeze(0)
    if polygons.ndim != 2:
        raise ValueError(
            f"Argument: `polygons` has invalid shape, expected [N,M] but got {list(polygons.shape)}"
        )
    return_tensors = isinstance(image, torch.Tensor)
    # this is not implemented using torch... 
    # its just easier and doesnt need to be super fast as its for visualisation purpose only
    image = to_pil(image).convert("RGBA")
    polygons = polygons.numpy()  # required for draw.polygon
    overlay = pillow.Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = pillow.ImageDraw.Draw(overlay)

    if line_width is None:
        line_width = int(max(2, max(image.size) / 400))
    
    def _expand_color(color : Color, alpha : int = 255) -> torch.Tensor:
        if isinstance(color, torch.Tensor):
            if color.ndim == 1:
                color = color.unsqueeze(0)
            # check the colors are in uint8 255 format
            if torch.is_floating_point(color): # assume [0-1] range
                color = (color.clamp_(0.0, 1.0) * 255.).to(torch.uint8)
            else:
                color = (color % 255).to(torch.uint8)
        elif isinstance(color, list) and not isinstance(color, (str,int,float)):
            color = torch.tensor([color_with_alpha(color_to_tuple(c), alpha) for c in color], dtype=torch.uint8)
        else:
            color = torch.tensor(color_with_alpha(color_to_tuple(color), alpha), dtype=torch.uint8).unsqueeze(0)
        assert color.ndim == 2 and color.shape[-1] == 4
        return color
    fill_color = _expand_color(fill_color if fill_color else (0,0,0,0), alpha=60) 
    line_color = _expand_color(line_color if line_color else (0,0,0,0), alpha=200)
    if labels is None:
        fill_color = fill_color.expand(polygons.shape[0], -1)
        line_color = line_color.expand(polygons.shape[0], -1)
    else:
        fill_color = fill_color[labels]
        line_color = line_color[labels]

    for polygon, lc, fc in zip(polygons, line_color, fill_color):
        if lc.sum() > 0:
            draw.polygon(polygon, outline=tuple(lc.tolist()), width=line_width)
        if fc.sum() > 0:
            fc = tuple(fc.tolist())
            draw.polygon(polygon, outline=fc, fill=fc, width=1)
    image = pillow.Image.alpha_composite(image, overlay).convert("RGB")
    if not return_tensors:
        return image
    else:
        return to_tensor(image)