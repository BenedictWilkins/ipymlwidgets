from traitlets import TraitType, TraitError
from typing import Dict, Any, Optional, Union
import re

class Layout(TraitType):
    """A trait type for CSS layout properties as a dictionary.
    
    Supports: display, flex-direction, gap, padding, margin
    """
    
    VALID_DISPLAY = {
        'block', 'inline', 'inline-block', 'flex', 'inline-flex', 
        'grid', 'inline-grid', 'none', 'table', 'table-cell'
    }
    
    VALID_FLEX_DIRECTION = {
        'row', 'row-reverse', 'column', 'column-reverse'
    }
    
    # Regex for valid CSS length values (px, em, rem, %, auto, etc.)
    CSS_LENGTH_PATTERN = re.compile(
        r'^(auto|0|(-?\d*\.?\d+)(px|em|rem|%|vh|vw|vmin|vmax|ch|ex|cm|mm|in|pt|pc)?)(\s+(auto|0|(-?\d*\.?\d+)(px|em|rem|%|vh|vw|vmin|vmax|ch|ex|cm|mm|in|pt|pc)?)){0,3}$'
    )
    
    SUPPORTED_PROPERTIES = {
        'display', 'flex-direction', 'gap', 'padding', 'margin', "width", "height", "min-width", "max-width", "min-height", "max-height"
    }
   
    
    def __init__(self, default_value : Optional[Dict[str, str]] = None, **kwargs):
        """Initialize Layout trait.
        
        Args:
            default_value (dict | None): CSS properties dict. Defaults to None (don't apply any custom css on initialisation).
            **kwargs: Additional arguments passed to TraitType.
        """
        if default_value is None:
            default_value = {}
        super().__init__(default_value, **kwargs)
    
    @property
    def info_text(self):
        props = ', '.join(self.SUPPORTED_PROPERTIES)
        return f"a dict of CSS properties ({props})"
    
    def validate(self, obj: Any, value: Any) -> Optional[Dict[str, str]]:
        """Validate CSS layout properties dictionary.
        
        Args:
            obj (Any): The object that owns this trait.
            value (Any): The value to validate.
            
        Returns:
            Optional[Dict[str, str]]: Validated CSS properties dict.
            
        Raises:
            TraitError: If validation fails.
        """
        if value is None:
            return {}
            
        if not isinstance(value, dict):
            self.error(obj, value, info="Layout must be a dictionary")
        
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
                self.error(obj, value, info=f"Unsupported CSS property: '{prop}'. Supported: {', '.join(self.SUPPORTED_PROPERTIES)}")
            
            # Validate specific properties
            if prop == 'display':
                validated[prop] = self._validate_display(obj, value, val_str)
            elif prop == 'flex-direction':
                validated[prop] = self._validate_flex_direction(obj, value, val_str)
            elif prop in {'gap', 'padding', 'margin', 'width', 'height', 'min-width', 'max-width', 'min-height', 'max-height'}:
                validated[prop] = self._validate_length(obj, value, prop, val_str)
            else:
                validated[prop] = val_str
        
        return validated
    
    def _validate_display(self, obj: Any, full_value: Any, value: str) -> str:
        """Validate display property value."""
        if value not in self.VALID_DISPLAY:
            valid_values = ', '.join(sorted(self.VALID_DISPLAY))
            self.error(obj, full_value, info=f"Invalid display value: '{value}'. Valid values: {valid_values}")
        return value
    
    def _validate_flex_direction(self, obj: Any, full_value: Any, value: str) -> str:
        """Validate flex-direction property value."""
        if value not in self.VALID_FLEX_DIRECTION:
            valid_values = ', '.join(sorted(self.VALID_FLEX_DIRECTION))
            self.error(obj, full_value, info=f"Invalid flex-direction value: '{value}'. Valid values: {valid_values}")
        return value
    
    def _validate_length(self, obj: Any, full_value: Any, prop: str, value: str) -> str:
        """Validate length properties (gap, padding, margin, width*, height*)."""
        # Allow shorthand values like "10px 20px" or single values like "10px" or "auto"
        if not self.CSS_LENGTH_PATTERN.match(value.strip()):
            self.error(obj, full_value, 
                      info=f"Invalid {prop} value: '{value}'. Expected CSS length values (e.g., '10px', '1em 2em', 'auto')")
        return value.strip()

    def __eq__(self, other: Any) -> bool:
        """Use identity comparison to avoid issues with dict comparison."""
        return self is other

    def __ne__(self, other: Any) -> bool:
        """Use identity comparison to avoid issues with dict comparison."""
        return self is not other