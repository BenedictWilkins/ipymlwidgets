
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

        let imageModel = children_models[0];
        let textModel = children_models[1];

        let imageView = await model.widget_manager.create_view(imageModel);
        let textView = await model.widget_manager.create_view(textModel);

        // Create a wrapper for the image to make it scrollable
        const imageContainer = document.createElement("div");
        imageContainer.classList.add("ocr-image-container");
        
        // Add wrapper for border-radius clipping
        const imageWrapper = document.createElement("div");
        imageWrapper.classList.add("ocr-image-wrapper");
        imageWrapper.appendChild(imageView.el);

        // Create the X button
        const closeButton = document.createElement("button");
        closeButton.classList.add("ocr-close-button");
        closeButton.innerHTML = "✕";
        //closeButton.setAttribute("title", "Remove item");
        
        // Add click handler for the close button
        closeButton.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const event = {
                t: Date.now(),
                action: "close"
            };
            model.set("click_close", event);
            model.save_changes();
        });
        
        imageContainer.appendChild(imageWrapper);
        imageContainer.appendChild(closeButton);  // Add button to container
        el.appendChild(imageContainer);
        el.appendChild(textView.el);

        el.classList.add("ocr-item");
        imageView.el.classList.add("widget-image");
        textView.el.classList.add("widget-text");
        }
    export default { render };
    """

    _css = """
    .ocr-item {
        height: 90px;  /* Set explicit height for the container */
        display: flex;
        flex-direction: column;
        width: 100%;
        box-sizing: border-box;
    }

    .ocr-close-button:hover {
        color: #333;
        transform: scale(1.1);
    }

    .ocr-close-button:active {
        transform: scale(0.95);
    }

    .ocr-close-button {
        position: absolute;
        top: -2px;
        right: -2px;
        width: 16px;
        height: 16px;
        border: none;
        padding-right: 8px;
        padding-top: 4px;
        padding-bottom: 1px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 4px;    /* This makes it circular */
        color: #666;
        cursor: pointer;
        font-size: 12px;
        font-weight: bold;
        line-height: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10; 
        transition: all 0.1s ease;
    }

    .ocr-image-wrapper {
        display: inline-block;
        height: 100%;
        width: auto;
        box-sizing: border-box;
        overflow: hidden;
        //overflow-x: auto;
        //overflow-y: hidden;
        white-space: nowrap;
        border: 1px solid #ccc;
        border-radius: 4px;
    }

    .ocr-image-container {
        flex: 1;  /* Take up remaining space, leaving room for text input */
        width: auto;
        overflow-x: auto;
        overflow-y: hidden;
        display: block;
        white-space: nowrap;
        box-sizing: border-box;
        margin-bottom: 4px;
        padding: 0;
    }

    /* Target the widget wrapper */
    .ocr-item > .ocr-image-container > .ocr-image-wrapper > .widget-image {
        display: inline-block;
        height: 100% !important;
        width: auto !important;
        vertical-align: top;
        max-width: none !important;
        min-width: fit-content !important;  /* Allow natural width */
        box-sizing: content-box;
        white-space: nowrap;
        overflow: visible;
    }

    /* Force child elements (canvas or otherwise) to grow */
    .ocr-item > .ocr-image-container > .ocr-image-wrapper > .widget-image * {
        height: 100% !important;
        width: auto !important;
        max-width: none !important;
        display: inline-block !important;
        box-sizing: content-box !important;
        object-fit: contain !important;
        white-space: nowrap;
    }

    .ocr-item > .ocr-image-container {
        scrollbar-width: thin;                  /* Firefox */
        scrollbar-color: #bbb transparent;      /* Firefox */
    }

    .ocr-item > .ocr-image-container::-webkit-scrollbar {
        height: 8px;
    }
    .ocr-item > .ocr-image-container::-webkit-scrollbar-track {
        background: transparent;
    }
    .ocr-item > .ocr-image-container::-webkit-scrollbar-thumb {
        background-color: #bbb;
        border-radius: 2px;
    }
    .ocr-item > .ocr-image-container::-webkit-scrollbar-thumb:hover {
        background-color: #999;
    }

    /* Style the text input */
    .ocr-item > .widget-text {
        flex: 0 0 auto;  /* Don't grow, fixed size */
        width: 100%;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    .ocr-item > .widget-text input {
        width: 100%;
        height: 24px;  
        padding: 4px;
        border: 1px solid #ccc;
        border-radius: 4px;
        box-sizing: border-box;
        font-size: 14px;
        line-height: 1.2;
    }
    """
    children = traitlets.List(trait=traitlets.Instance(W.DOMWidget)).tag(sync=True, **W.widget_serialization)
    click_close = traitlets.Dict().tag(sync=True)

    def __init__(self, image: SupportedTensor, text: str):
        self._image = Image(image)
        self._text = W.Text(value=text, layout=W.Layout(width="100%", margin="0", box_sizing="border-box"))
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




# .ocr-item {
#         display: flex;
#         flex-direction: column;
#         padding: 4px;
#         margin: 0;
#         width: 100%;
#         height: 100%; /* Ensures the container takes full height */
#         min-height: 86px; /* Set a minimum height */
#         box-sizing: border-box; /* Includes padding in height/width calculation */

#         /* Add a border and rounded corners */
#         border: 1px solid #ccc;  /* You can change the border color and thickness */
#         border-radius: 8px;      /* This gives the rounded corners */
#         background-color: #f0f0f0;

#         /* Align children to the top */
#         align-items: flex-start;  /* Align image and text to the left */
#         justify-content: flex-start; /* Align content to the top */
#     }

#     .ocr-item > .widget-image {
#         border: 1px solid #ccc;  /* You can change the border color and thickness */
#         border-radius: 4px;      /* This gives the rounded corners */
#         object-fit: contain; /* Ensure image retains aspect ratio */
#         width: auto;      /* Ensure image takes up full width */
#         height: 100%;     /* Ensure image takes up full height available */
#         // min-height: 100px; /* Set a minimum height */
#         margin-bottom: 4px; /* Space below the image */
#         display: block;
#     }

#     .ocr-item > .widget-text {
#         flex: 0 1 auto;   /* Let the text take the remaining space */
#         width: 100%;      /* Ensure text box fills the available width */
#         margin: 0;
#         padding: 0;
#     }

#     .ocr-item > .widget-text input {
#         border: 1px solid #ccc;      /* Add border to the input element */
#         border-radius: 4px;         /* Rounded corners for the text input */
#         background-color: white;     /* Background color for the input box */
#     }
