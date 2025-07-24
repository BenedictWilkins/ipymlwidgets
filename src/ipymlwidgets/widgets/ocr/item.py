
import ipywidgets as W
import traitlets
import anywidget

from ipymlwidgets.widgets import Image, Text
from ipymlwidgets.traits import SupportedTensor



class ItemOCR(anywidget.AnyWidget):
    """A widget that can be placed in a list and used as part of an OCR annotation widget to display an image crop and associated text.
    """

    _esm = """
    async function unpack_models(model_ids, manager) {
        return Promise.all(
            model_ids.map(id => manager.get_model(id.slice("IPY_MODEL_".length)))
        );
    }
    async function render({ model, el }) {
       let model_ids = model.get("children");
        let children_models = await unpack_models(model_ids, model.widget_manager);

        // Assuming there are only two children: image and text
        let imageModel = children_models[0];
        let textModel = children_models[1];

        // Create views for the image and text
        let imageView = await model.widget_manager.create_view(imageModel);
        let textView = await model.widget_manager.create_view(textModel);
       
        // Append the children directly
        el.appendChild(imageView.el);
        el.appendChild(textView.el);

        // Add the "ocr-item" class to the container for custom styling
        el.classList.add("ocr-item");
        imageView.el.classList.add("widget-image");
        textView.el.classList.add("widget-text");

    }
    export default { render };
    """

    _css = """
    .ocr-item {
        display: flex;
        flex-direction: column;
        padding: 10px;
        width: 100%;
        height: 100%; /* Ensures the container takes full height */
        box-sizing: border-box; /* Includes padding in height/width calculation */

        /* Add a border and rounded corners */
        border: 1px solid #ccc;  /* You can change the border color and thickness */
        border-radius: 8px;      /* This gives the rounded corners */
        background-color: #f0f0f0;

        /* Align children to the top */
        align-items: flex-start;  /* Align image and text to the left */
        justify-content: flex-start; /* Align content to the top */
    }

    .ocr-item > .widget-image {
        border: 1px solid #ccc;  /* You can change the border color and thickness */
        border-radius: 4px;      /* This gives the rounded corners */
        object-fit: contain; /* Ensure image retains aspect ratio */
        width: auto;      /* Ensure image takes up full width */
        height: 100%;     /* Ensure image takes up full height available */
        margin-bottom: 10px; /* Space below the image */
        display: block;
    }

    .ocr-item > .widget-text {
        flex: 0 1 auto;   /* Let the text take the remaining space */
        width: 100%;      /* Ensure text box fills the available width */
        margin: 0;
        padding: 0;
    }

    .ocr-item > .widget-text input {
        border: 1px solid #ccc;      /* Add border to the input element */
        border-radius: 4px;         /* Rounded corners for the text input */
        background-color: white;     /* Background color for the input box */
    }
    """
    children = traitlets.List(trait=traitlets.Instance(W.DOMWidget)).tag(sync=True, **W.widget_serialization)

    def __init__(self, image: SupportedTensor, text: str):
        self._image = Image(image)
        self._text = W.Text(value=text, layout=W.Layout(width="100%", margin="0", padding="0", box_sizing="border-box", border="none"))
        super().__init__()        
        self.children = [self._image, self._text]

    def set_image(self, image: SupportedTensor):
        self._image.set_image(image)

    def set_text(self, text: str):
        self._text.value = text

    def focus(self):
        """Focus the text input."""
        self._text.focus()

    def unfocus(self):
        """Unfocus the text input."""
        self._text.unfocus()