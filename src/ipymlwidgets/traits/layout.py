from traitlets import TraitType, TraitError
from typing import Dict, Any, Optional

class Layout(TraitType):
    """A trait type for CSS layout properties with simple custom validation."""
    
    # Define which properties we want to support
    SUPPORTED_PROPERTIES = {
        'display', 'position', 'overflow', 'float', 'clear', 'z-index',
        'width', 'height', 'min-width', 'max-width', 'min-height', 'max-height',
        'padding', 'margin', 'gap', 'top', 'right', 'bottom', 'left',
        'flex-direction', 'flex-wrap', 'justify-content', 'align-items', 'align-self',
        'flex-grow', 'flex-shrink', 'flex-basis', 'flex', 'order',
        'grid-template-columns', 'grid-template-rows', 'grid-area', 
        'grid-column', 'grid-row', 'grid-gap',
        'text-align', 'vertical-align', 'background', 'background-color',
        'border', 'border-radius', 'color', 'opacity', 'box-shadow',
        'transform', 'transition', 'box-sizing',
    }
   
    def __init__(self, default_value: Optional[Dict[str, str]] = None, **kwargs):
        if default_value is None:
            default_value = {}
        super().__init__(default_value, **kwargs)
    
    def validate(self, obj: Any, value: Any) -> Optional[Dict[str, str]]:
        if value is None:
            return {}
        
        # Handle Layout instances by extracting their dictionary data
        if isinstance(value, Layout):
            value = getattr(value, 'default_value', {})
            
        if not isinstance(value, dict):
            self.error(obj, value, info="Layout must be a dictionary or Layout instance")
        
        validated = {}
        
        for prop, val in value.items():
            if not isinstance(prop, str):
                self.error(obj, value, info=f"Property names must be strings, got {type(prop).__name__}")
            
            if not isinstance(val, (str, int, float)):
                self.error(obj, value, info=f"Property values must be strings or numbers, got {type(val).__name__} for '{prop}'")
            
            # Convert to string for validation
            val_str = str(val)
            
            # Check if property is supported
            if prop not in self.SUPPORTED_PROPERTIES:
                self.error(obj, value, info=f"Unsupported CSS property: '{prop}'. Supported: {', '.join(sorted(self.SUPPORTED_PROPERTIES))}")
            
            # Simple validation - just check it's not empty and doesn't contain invalid characters
            if self._is_valid_css_value(val_str):
                validated[prop] = val_str
            else:
                self.error(obj, value, info=f"Invalid value '{val_str}' for CSS property '{prop}'")
        
        return validated
    
    def _is_valid_css_value(self, value: str) -> bool:
        """Simple CSS value validation."""
        # Must not be empty
        if not value.strip():
            return False
        
        # Must not contain characters that would break CSS
        invalid_chars = [';', '{', '}', '\n', '\r']
        if any(char in value for char in invalid_chars):
            return False
        
        # Everything else is valid - be permissive
        return True

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __ne__(self, other: Any) -> bool:
        return self is not other


if __name__ == "__main__":
    # Test the layout validation
    layout = Layout()
    test_values = {
        "display": "flex", 
        "flex-direction": "row", 
        "margin": "10px", 
        "gap": "20px"
    }
    result = layout.validate(None, test_values)
    print("Layout validation working!", result)

