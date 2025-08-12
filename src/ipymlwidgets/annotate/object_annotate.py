from ipymlwidgets import ImageAnnotated, Box, Button
from ipymlwidgets.traits import Tensor as TTensor
import numpy as np
import ipywidgets as W
from typing import Optional, Callable, Dict, Any
from PIL import Image
from pathlib import Path

class ObjectAnnotator(Box):
    """A class for annotating object detection data with navigation functionality."""
    
    def __init__(
        self, 
        data: Any,
        save_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        load_callback: Optional[Callable[[int, Dict[str, Any]], Dict[str, Any]]] = None,
        auto_save: bool = True
    ):
        """Initialize the object detection annotator.
        
        Args:
            data: Any indexable collection where each item contains:
                - 'image': numpy array or image data
                - 'boxes': numpy array of bounding boxes (N, 4) format [x0, y0, x1, y1]
            save_callback: callback for saving any new data.
            auto_save: Whether to auto-save when navigating between examples
        """
        self._data = data
        self.save_callback = save_callback
        self.load_callback = load_callback
        self.auto_save = auto_save
        self.index = 0

        # Setup UI components
        children = self._setup_ui()
        super().__init__(children=children, layout={"width": "100%"})
        self._load_current_example()
    
    def _setup_ui(self):
        """Setup the user interface components."""
        # Create initial image annotator (will be updated with first example)
        self.image_annotator = ImageAnnotated()
        
        # Setup event observers
        self.image_annotator.observe(self._on_key_press, "key_press")
        self.image_annotator.observe(self._on_boxes_change, "boxes")
        self.image_annotator.observe(self._on_selection_change, "selection")

        # Create navigation buttons
        self.btn_prev = Button("Previous")
        self.btn_next = Button("Next")
        # Setup button click handlers
        self.btn_prev.on_click(self._on_prev_click)
        self.btn_next.on_click(self._on_next_click)

        buttons = [self.btn_prev, self.btn_next]
        # Setup save button (if auto save is not enabled)
        if not self.auto_save:
            self.btn_save = Button("Save")
            self.btn_save.on_click(self._on_save_click)
            buttons.append(self.btn_save)
        else:
            self.btn_save = None
        
        # Create button layout
        self.button_box = Box(
            buttons,
            layout={
            "display": "flex", 
            "flex-direction": "row", 
            "padding": "10px", 
            "width": "100%",
            "gap": "10px",
            "align-items": "center"
        })
        return [
            self.image_annotator, 
            self.button_box
        ]

    def _validate_example(self, example: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Validate and normalize an example from the collection."""

        if 'image' not in example:
            raise ValueError(f"Example at index {index} missing required `image` key")
        
        # Normalize image
        if isinstance(example['image'], str):
            image = np.array(Image.open(example['image']))
        elif isinstance(example['image'], Path):
            image = np.array(Image.open(example['image'].as_posix()))
        elif isinstance(example['image'], Image.Image):
            image = np.array(example['image'])
        elif isinstance(example['image'], np.ndarray):
            image = example['image']
        else:
            pass 

        if image.ndim not in [2, 3]:
            raise ValueError(f"`image` at index {index} must be 2D or 3D array")
        
        if example.get('boxes', None) is None:
            boxes = np.array([])
        else:
            boxes = np.array(example['boxes'])

        if boxes.ndim < 2:
            boxes = boxes.reshape(-1, 4)
        elif boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"`boxes` at index {index} must have shape [N, 4] but got {list(boxes.shape)}")
        
        return {
            'image': image,
            'boxes': boxes,
        }

    def _on_selection_change(self, change):
        if change['new'] is not None:
            # delete the selecction - ctrl is the delete modifier
            if self.image_annotator.mouse_down['ctrl']:
                self.image_annotator.remove_box(change['new'].index)
        
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get the current data including any modifications."""
        return {
            'image': self.image_annotator.image,
            'boxes': self.image_annotator.boxes.copy(),
        }
    
    def _save_current(self):
        """Save the current example if there are changes."""
        if self.save_callback is not None:
            try:
                current_data = self.get_current_data()
                self.save_callback(self.index, current_data)
                #self.status_display.value = f"<p style='color: green;'>Saved example {self.index}</p>"
            except Exception as e:
                # self.status_display.value = f"<p style='color: red;'>Error saving: {str(e)}</p>"
                raise e
    
    def _load_current_example(self):
        """Load the current example into the image annotator."""
        try:
            example = self._data[self.index]
            if self.load_callback is not None:
                example = self.load_callback(self.index, example)
            validated_example = self._validate_example(example, self.index)
            
            self.image_annotator.set_image(validated_example['image'])
            self.image_annotator.set_boxes(validated_example['boxes'])
        except Exception as e:
            # TODO? 
            #self.status_display.value = f"<p style='color: red;'>Error loading example: {str(e)}</p>"
            raise e

    def _on_prev_click(self, _):
        """Handle previous button click."""
        if self.auto_save:
            self._save_current()
        if self.index > 0:
            self.index -= 1
            self._load_current_example()
        
    def _on_next_click(self, _):
        """Handle next button click."""
        if self.auto_save:
            self._save_current()
        if self.index < len(self._data) - 1: 
            self.index += 1
            self._load_current_example()
        
    def _on_save_click(self, _):
        """Handle save button click."""
        self._save_current()
        
    def _on_key_press(self, change):
        """Handle keyboard events."""
        event = change['new']
        
        if event['key'] == 'ArrowRight':
            self._on_next_click(None)
        elif event['key'] == 'ArrowLeft':
            self._on_prev_click(None)
        elif event['key'] == 's' and event.get('ctrl', False):
            # TODO not sure this would work ...? hmm...
            self._on_save_click(None)
    
    def _on_boxes_change(self, _):
        """Handle changes to bounding boxes."""
        self._data[self.index]['boxes'] = self.image_annotator.boxes.copy()
