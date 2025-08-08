from typing import Optional
import ipywidgets as W

class Button(W.Button):
    """Wrapper around `ipywidgets.Button` that allows for setting the button label."""

    def __init__(self, label: Optional[str] = None):
        super().__init__(description=label)

    def set_label(self, label: str):
        self.description = label

    def get_label(self) -> str:
        return self.description