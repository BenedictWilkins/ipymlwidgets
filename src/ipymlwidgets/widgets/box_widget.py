import anywidget
import traitlets
from typing import Optional, Tuple
import pathlib


class BoxWidget(anywidget.AnyWidget):
    """A widget displaying 4 editable text fields for box coordinates (x0, y0, x1, y1).

    The fields are horizontally aligned, expand equally, and are disabled if `box_coords` is None.
    The coordinates are stored in the `box_coords` traitlet as either None or a tuple of 4 integers.
    """

    box_coords: Optional[Tuple[int, int, int, int]] = traitlets.Tuple(
        (traitlets.Int(),) * 4, allow_none=True
    ).tag(sync=True)

    _esm = pathlib.Path(__file__).parent / "box_widget.js"

    def __init__(
        self, box_coords: Optional[Tuple[int, int, int, int]] = None, **kwargs
    ) -> None:
        """Initialize the BoxWidget.

        Args:
            box_coords (Optional[tuple[int, int, int, int]]):
                Initial box coordinates (x0, y0, x1, y1), or None. Defaults to None.
            **kwargs: Additional keyword arguments passed to AnyWidget.
        """
        super().__init__(box_coords=box_coords, **kwargs)
