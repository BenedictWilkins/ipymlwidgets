from typing import Any
import ipywidgets as W

class Text(W.Text):
    """Wrapper around `ipywidgets.Text` that allows for setting the text value."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_text(self, text: Any):
        self.value = str(text)

    def get_text(self) -> str:
        return self.value