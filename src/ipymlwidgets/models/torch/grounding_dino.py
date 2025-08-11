"""Grounding DINO wrapper for easy object detection."""

import torch
import numpy as np
from typing import List, Union, Optional, Tuple
from PIL import Image
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection


class GroundingDINO:
    """A simple wrapper for Grounding DINO object detection.
    
    This class provides an easy-to-use interface for Grounding DINO with a single
    annotate method that takes text prompts and returns bounding boxes.
    """
    
    def __init__(
        self, 
        model_name: str = "IDEA-Research/grounding-dino-base",
        device: Optional[str] = None,
        box_threshold: float = 0.0,
        text_threshold: float = 0.0
    ):
        """Initialize the Grounding DINO model.
        
        Args:
            model_name (str): Model name from Hugging Face. Defaults to "IDEA-Research/grounding-dino-base".
            device (Optional[str]): Device to run the model on. Defaults to auto-detect.
            box_threshold (float): Threshold for box confidence. Defaults to 0.3.
            text_threshold (float): Threshold for text confidence. Defaults to 0.25.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Load processor and model
        self.processor = GroundingDinoProcessor.from_pretrained(model_name)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def _preprocess_caption(self, caption: str) -> str:
        """Preprocess caption to ensure proper format.
        
        Args:
            caption (str): Input caption.
            
        Returns:
            str: Processed caption ending with a period.
        """
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."
    
    def _prepare_image(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Convert various image formats to PIL Image.
        
        Args:
            image: Input image in various formats.
            
        Returns:
            Image.Image: PIL Image object.
        """
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            # Assume tensor is in format (C, H, W) or (H, W, C)
            if image.dim() == 3:
                if image.shape[0] in [1, 3]:  # (C, H, W)
                    image = image.permute(1, 2, 0)
                # Convert to numpy and then PIL
                image_np = image.cpu().numpy()
                if image_np.dtype != np.uint8:
                    image_np = (image_np * 255).astype(np.uint8)
                return Image.fromarray(image_np).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def annotate(
        self, 
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        prompts: Union[str, List[str]],
        return_labels: bool = False,
        return_scores: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Annotate image with bounding boxes for given prompts.
        
        Args:
            image: Input image in various formats (file path, PIL Image, numpy array, or torch tensor).
            prompts: Text prompt(s) describing objects to detect.
            return_labels (bool): Whether to return predicted labels. Defaults to False.
            return_scores (bool): Whether to return confidence scores. Defaults to False.
            
        Returns:
            torch.Tensor: Bounding boxes as tensor of shape (N, 4) in format [x0, y0, x1, y1].
            If return_labels=True, returns tuple (boxes, labels).
            If return_scores=True, returns tuple (boxes, scores) or (boxes, labels, scores).
        """
        # Prepare image
        pil_image = self._prepare_image(image)
        
        # Prepare prompts
        if isinstance(prompts, str):
            text = self._preprocess_caption(prompts)
        else:
            text = self._preprocess_caption(". ".join(prompts))
        
        # Process inputs
        inputs = self.processor(
            images=pil_image, 
            text=text, 
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        width, height = pil_image.size
        postprocessed_outputs = self.processor.image_processor.post_process_object_detection(
            outputs,
            target_sizes=[(height, width)],
            threshold=self.box_threshold
        )
        
        results = postprocessed_outputs[0]
        
        # Extract results
        boxes = results['boxes']  # Already in [x0, y0, x1, y1] format
        scores = results['scores']
        labels = results['labels']
        
        # Filter by text threshold if needed
        if hasattr(outputs, 'logits'):
            # Additional filtering could be added here based on text threshold
            pass
        
        # Prepare return values
        return_values = [boxes]
        
        if return_labels:
            # Convert label indices to text (this would need the class mapping)
            return_values.append(labels)
            
        if return_scores:
            return_values.append(scores)
        
        return tuple(return_values)
    
    def set_thresholds(self, box_threshold: float, text_threshold: float) -> None:
        """Update detection thresholds.
        
        Args:
            box_threshold (float): New box confidence threshold.
            text_threshold (float): New text confidence threshold.
        """
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold