import traitlets
import ipywidgets as W
import anywidget

class Box(anywidget.AnyWidget):
    _esm = """
    async function unpack_models(model_ids, manager) {
        return Promise.all(
            model_ids.map(id => manager.get_model(id.slice("IPY_MODEL_".length)))
        );
    }
    async function render({ model, el }) {
       let model_ids = model.get("children");
        let children_models = await unpack_models(model_ids, model.widget_manager);
        for (let model of children_models) {
            let child_view = await model.widget_manager.create_view(model);
            el.appendChild(child_view.el);
        }
    }
    export default { render };
    """
    children = traitlets.List(trait=traitlets.Instance(W.DOMWidget)).tag(sync=True, **W.widget_serialization)

    def __init__(self, children: list[W.DOMWidget], **kwargs):
        super().__init__(children=children, **kwargs)

    