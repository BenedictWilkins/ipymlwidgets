from typing import Optional, Any
import numpy as np
from PIL import Image
from ipymlwidgets.traits.tensor import SupportedTensor
from ipymlwidgets.traits.tensor import Tensor as TTensor, OptionalDependency

def to_numpy_image(
    tensor: Optional[SupportedTensor],
) -> Optional[np.ndarray]:
    """Convert image array to numpy array.

    Args:
        tensor (SupportedTensor): Image tensor to convert.

    Returns:
        Optional[np.ndarray]: Converted image data as numpy array or None if tensor is None.
    """
    if tensor is None:
        return None
    dependency: OptionalDependency = TTensor.get_dependency(tensor)
    array = dependency.to_numpy_image(tensor)
    assert array.ndim == 3
    assert array.shape[2] == 4  # HWC format RGBA
    assert array.dtype == np.uint8
    return array

def resize_image_letterbox(image: np.ndarray, target_size: tuple[int, int], fill_color: tuple[int, int, int, int] = (0, 0, 0, 255)) -> np.ndarray:
    """Resize image with letterboxing to maintain aspect ratio, positioned at top-left.
    
    Args:
        image (np.ndarray): Input image array of shape (H, W, C) where C is 3 or 4
        target_size (tuple[int, int]): Target size as (width, height)
        fill_color (tuple[int, int, int, int]): RGBA fill color for letterbox areas
        
    Returns:
        np.ndarray: image array with shape (target_height, target_width, 4)
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D array (H, W, C), got shape {image.shape}")
    
    height, width, channels = image.shape
    target_width, target_height = target_size
    
    # Convert to PIL Image
    if channels == 3:
        pil_image = Image.fromarray(image, mode='RGB').convert('RGBA')
    elif channels == 4:
        pil_image = Image.fromarray(image, mode='RGBA')
    else:
        raise ValueError(f"Expected 3 or 4 channels, got {channels}")
    
    # Calculate scaling to fit within target while maintaining aspect ratio
    scale = min(target_width / width, target_height / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image maintaining aspect ratio
    resized_pil = pil_image.resize((new_width, new_height), Image.Resampling.NEAREST)
    
    # Create target canvas with fill color
    canvas = Image.new('RGBA', target_size, fill_color)
    
    # Paste the resized image at top-left (0, 0)
    canvas.paste(resized_pil, (0, 0))
    
    return np.array(canvas)

def resize_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize a numpy image array to target size using PIL.
    
    Args:
        image (np.ndarray): Input image array of shape (H, W, C) where C is 3 or 4
        target_size (tuple[int, int]): Target size as (width, height)
        
    Returns:
        np.ndarray: Resized image array with shape (target_height, target_width, 4) in RGBA format
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D array (H, W, C), got shape {image.shape}")
    
    height, width, channels = image.shape
    
    # Convert to PIL Image
    if channels == 3:
        # RGB -> RGBA
        pil_image = Image.fromarray(image, mode='RGB')
        pil_image = pil_image.convert('RGBA')
    elif channels == 4:
        # Already RGBA
        pil_image = Image.fromarray(image, mode='RGBA')
    else:
        raise ValueError(f"Expected 3 or 4 channels, got {channels}")
    
    # Resize using PIL's high-quality resampling
    target_width, target_height = target_size
    resized_pil = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.array(resized_pil)