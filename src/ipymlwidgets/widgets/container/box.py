import ipywidgets as W
import anywidget
import traitlets
from typing import Optional, Dict, List

from ipymlwidgets.traits.layout import Layout

class Box(anywidget.AnyWidget):
    """Box widget is a simple container widget that displays its children according to the layout css provided."""

    _esm = """
    async function render({ model, el }) {

        // this contains the mapping {id, AnyView} of each child widget
        let children = new Map();
       
        async function unpack_views(ids, widget_manager) {
            return Promise.all(
                ids.map(async (id) => {
                    let widget_view = children.get(id)
                    if (widget_view) {
                        console.log("found existing view", id)
                        return [id, widget_view];
                    } else {
                        console.log("creating new view", id)
                        const widget_model = await widget_manager.get_model(id.slice("IPY_MODEL_".length));
                        const widget_view = await widget_manager.create_view(widget_model);
                        return [id, widget_view]
                    }
                })
            );
        }

        async function set_children(model, el) {
            const ids = model.get("children");
            console.log("ids", ids)
            const _children = new Map(await unpack_views(ids, model.widget_manager));
            el.innerHTML = ""; // clear the container
            _children.forEach((child) => {
                el.appendChild(child.el);
            });
            children = _children;
        }

        async function set_layout(model, el) {
            const layout = model.get("layout");
            console.log("new layout", layout)
            Object.keys(layout).forEach((key) => {
                const cssProperty = key.replace(/([A-Z])/g, '-$1').toLowerCase();  // Convert camelCase to kebab-case
                el.style[cssProperty] = layout[key];  // Apply the CSS property to the element
            });
        }

        requestAnimationFrame(() => set_children(model, el));
        requestAnimationFrame(() => set_layout(model, el));
        
        model.on("change:children", () => {
            requestAnimationFrame(() => set_children(model, el));
        });

        // apply the layout css on change
        model.on("change:layout", () => {
            requestAnimationFrame(() => set_layout(model, el));
        });
    }
    export default { render };
    """

    children = traitlets.List(trait=traitlets.Instance(W.DOMWidget)).tag(sync=True, **W.widget_serialization)
    layout = Layout().tag(sync=True)

    def __init__(self, children: Optional[List[W.DOMWidget]] = None, layout: Optional[Dict[str,str]] = None):
        """Initialize Box widget.
        
        Args:
            children (list[W.DOMWidget], optional): List of child widgets. Defaults to None.
            layout (dict, optional): CSS layout properties as a dictionary. Defaults to None.
        """
        children = children if children is not None else []
        layout = layout if layout is not None else {}
        super().__init__(children=children, layout=layout)

    def insert_child(self, widget, index : int = 0):
        self.children.insert(index, widget)
        self.send_state()

    def add_child(self, widget):
        self.children.append(widget)
        self.send_state()

    def replace_child(self, widget, index: int):
        self.children[index] = widget
        self.send_state()

    def remove_child(self, widget):
        self.children.remove(widget)
        self.send_state()



