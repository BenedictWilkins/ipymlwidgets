
import ipywidgets as W
import traitlets
import anywidget

from ipymlwidgets.widgets import Image, Text
from ipymlwidgets.traits import SupportedTensor



class ItemOCR(anywidget.AnyWidget):
    """A widget that can be placed in a list and used as part of an OCR annotation widget to display an image crop and associated text."""

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
        display: flex;
        flex-direction: column;
        width: 100%;
        box-sizing: border-box;
        padding: 4px;             /* ⬅ padding so corners are visible */

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

    .ocr-image-container {
        flex: 0 0 auto;
        width: 100%;
        height: 90px;             /* fixed display height */
        overflow-x: auto;         /* horizontal scroll if needed */
        overflow-y: hidden;
        box-sizing: border-box;
        margin-bottom: 4px;
        white-space: nowrap;

        // border: 1px solid #ccc;
        //border-radius: 8px;       /* make corners more visible */
        //background-color: #f8f8f8;/* background peeks through padding */
    }

    .ocr-image-wrapper {
        display: inline-block;
        height: 100%;             /* fill remaining container height (minus padding) */
        width: auto;
        box-sizing: border-box;
        overflow: hidden;

        border-radius: 4px;       /* optional: clip the image too */
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


