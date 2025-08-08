import numpy as np

def demo_image(
    width: int = 32,
    height: int = 32,
    square_size: int = 4,
    color1: tuple[int, int, int, int] = (255, 255, 255, 255),
    color2: tuple[int, int, int, int] = (0, 0, 0, 255),
) -> np.ndarray:
    """Create a checkerboard pattern image.

    Args:
        width (int): Image width in pixels. Defaults to 32.
        height (int): Image height in pixels. Defaults to 32.
        square_size (int): Size of each square in pixels. Defaults to 4.
        color1 (tuple[int, int, int, int]): RGBA color for first squares. Defaults to (255, 255, 255, 255).
        color2 (tuple[int, int, int, int]): RGBA color for second squares. Defaults to (0, 0, 0, 255).

    Returns:
        np.ndarray: RGBA image array with shape (height, width, 4).
    """
    image = np.zeros((height, width, 4), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Determine which square we're in
            square_x = x // square_size
            square_y = y // square_size

            # Alternate colors based on square position
            if (square_x + square_y) % 2 == 0:
                image[y, x] = color1
            else:
                image[y, x] = color2

    return image