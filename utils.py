
@torch.jit.script
def box_crop_ragged_(image: torch.Tensor, boxes: torch.Tensor) -> list[torch.Tensor]:
    """Inplace (zero copy) version of `box_crop_ragged` crops will be a view into `image`."""
    if image.ndim != 3:
        _shape = list(image.shape)
        raise ValueError(
            f"Invalid Argument: `image` expected shape [C, H, W] but got {_shape}"
        )
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        _shape = list(image.shape)
        raise ValueError(
            f"Invalid Argument: `boxes` expected shape [N, 4] but got {_shape}"
        )
    if not _utils.is_integer(boxes):
        raise ValueError(
            f"Invalid Argument: `boxes` expected integer type but got {boxes.dtype}"
        )
    out = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        out.append(image[:, box[1] : box[3], box[0] : box[2]])
    return out



@torch.jit.script
def _downscale_to(
    image: torch.Tensor, target_size: tuple[int, int], mode: str = "bilinear"
) -> torch.Tensor:
    """Single image tensor version of `downscale_to`."""
    _, h, w = image.shape
    max_w, max_h = target_size
    if w <= max_w and h <= max_h:
        return image
    scale_w = max_w / w
    scale_h = max_h / h
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    _image = F.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode=mode,
        align_corners=False if mode in ["bilinear", "bicubic"] else None,
    ).squeeze(0)
    torch.clamp_(_image, 0.0, 1.0)
    return _image


@torch.jit.script
def downscale_to(
    images: list[torch.Tensor],
    target_size: tuple[int, int],
    mode: str = "bilinear",
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Downscale a ragged batch of images so that they fit within `target_size` while maintaining aspect ratio. Images will be packed and bottom-right padded into a tensor of shape [N,C,TargetH,TargetW] where `(TargetWidth, TargetHeight) = target_size`.

    Images already smaller than `target_size` will be unchanged.

    Args:
        images (list[torch.Tensor[C, H, W]]): Input image tensor.
        target_size (tuple[int, int]): Size (width, height) to downscale to.
        mode (str, optional): Interpolation mode, one of: 'bilinear', 'nearest', 'bicubic'. Defaults to 'bilinear'.
        pad_value (float, optional): Value used to pad the output tensor.

    Returns:
        torch.Tensor[N,C,TargetH,TargetW] : The downscaled packed and padded tensors.
    """
    n = len(images)
    if n == 0:
        raise ValueError("Argument: `images` cannot be empty.")
    if any(image.ndim != 3 for image in images):
        raise ValueError(
            "Argument: `images` expected to contain image tensors with shape [C, H, W]."
        )
    w, h = target_size
    c = images[0].shape[0]
    out = torch.full((n, c, h, w), pad_value)
    for i in range(n):
        image = images[i]
        image = _downscale_to(image, target_size, mode=mode)
        _c, _h, _w = image.shape
        out[i, :_c, :_h, :_w] = image
    return out