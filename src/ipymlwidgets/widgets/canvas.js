// --- Size & Layout Helpers ---
function updateStyles(model, wrapper) {
    wrapper.style.width = model.get("css_width");
    wrapper.style.height = model.get("css_height");
}

function updateClientSize(model, wrapper) {
    const rect = wrapper.getBoundingClientRect();
    const clientWidth = Math.round(rect.width);
    const clientHeight = Math.round(rect.height);
    const currentSize = model.get("client_size");
    if (currentSize[0] !== clientWidth || currentSize[1] !== clientHeight) {
        model.set("client_size", [clientWidth, clientHeight]);
        model.save_changes();
    }
}

function updateCanvas(model, canvases) {
    const width = model.get("width");
    const height = model.get("height");
    for (let i = 0; i < canvases.length; i++) {
        const canvas = canvases[i];
        canvas.width = width;
        canvas.height = height;
    }
}

// --- Mouse Event Handling ---
function setupMouseEvents(model, wrapper, canvases) {
    let isMouseDown = false;
    let dragStartPos = null;
    let dragThreshold = 3; // pixels

    function getMouseData(event) {
        const rect = wrapper.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const canvasX = Math.floor((x / rect.width) * canvases[0].width);
        const canvasY = Math.floor((y / rect.height) * canvases[0].height);
        return {
            x: canvasX,
            y: canvasY,
            x_client: x,
            y_client: y,
            w_client: rect.width,
            h_client: rect.height,
            w: canvases[0].width,
            h: canvases[0].height
        };
    }

    function handleMouseDown(event) {
        isMouseDown = true;
        const mouseData = getMouseData(event);
        dragStartPos = { x: mouseData.x, y: mouseData.y, clientX: event.clientX, clientY: event.clientY };
        model.set("mouse_down", mouseData);
        model.save_changes();
    }

    function handleMouseUp(event) {
        const mouseData = getMouseData(event);
        if (isMouseDown && dragStartPos) {
            const dx = event.clientX - dragStartPos.clientX;
            const dy = event.clientY - dragStartPos.clientY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < dragThreshold) {
                model.set("mouse_click", mouseData);
                model.save_changes();
            } else {
                const dragData = {
                    ...mouseData,
                    x_start: dragStartPos.x,
                    y_start: dragStartPos.y
                };
                model.set("mouse_drag", dragData);
                model.save_changes();
            }
            model.set("mouse_up", mouseData);
            model.save_changes();
        }
        isMouseDown = false;
        dragStartPos = null;
    }

    function handleMouseMove(event) {
        const mouseData = getMouseData(event);
        model.set("mouse_move", mouseData);
        model.save_changes();
        if (isMouseDown && dragStartPos) {
            const dx = event.clientX - dragStartPos.clientX;
            const dy = event.clientY - dragStartPos.clientY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance >= dragThreshold) {
                const dragData = {
                    ...mouseData,
                    x_start: dragStartPos.x,
                    y_start: dragStartPos.y
                };
                model.set("mouse_drag", dragData);
                model.save_changes();
            }
        }
    }

    function handleMouseEnter(event) {
        const mouseData = getMouseData(event);
        model.set("mouse_enter", mouseData);
        model.save_changes();
    }

    function handleMouseLeave(event) {
        const mouseData = getMouseData(event);
        model.set("mouse_leave", mouseData);
        isMouseDown = false;
        dragStartPos = null;
    }

    wrapper.addEventListener('mousedown', handleMouseDown);
    wrapper.addEventListener('mouseup', handleMouseUp);
    wrapper.addEventListener('mousemove', handleMouseMove);
    wrapper.addEventListener('mouseenter', handleMouseEnter);
    wrapper.addEventListener('mouseleave', handleMouseLeave);

    // Return a cleanup function
    return () => {
        wrapper.removeEventListener('mousedown', handleMouseDown);
        wrapper.removeEventListener('mouseup', handleMouseUp);
        wrapper.removeEventListener('mousemove', handleMouseMove);
        wrapper.removeEventListener('mouseenter', handleMouseEnter);
        wrapper.removeEventListener('mouseleave', handleMouseLeave);
    };
}

function drawRect(data, ctx, model) {
    if (!data) return;
    const { rects, count } = data;
    if (!rects || !count) {
        console.log('[draw] No rects or count');
        return;
    }
    let buffer;
    if (rects instanceof ArrayBuffer) {
        buffer = rects;
    } else if (rects instanceof DataView) {
        buffer = rects.buffer;
    } else {
        console.log('[draw] rects is not ArrayBuffer or DataView:', rects);
        return;
    }
    const intArr = new Int32Array(buffer);
    console.log('[draw] Decoded intArr:', intArr, 'count:', count);
    const strokeWidth = model.get("stroke_width");
    const strokeColor = model.get("stroke_color");
    const fillColor = model.get("fill_color");
    ctx.strokeStyle = strokeColor || 'red';
    ctx.lineWidth = strokeWidth || 1;
    if (fillColor && fillColor !== "") {
        ctx.fillStyle = fillColor;
    }
    for (let i = 0; i < count; ++i) {
        const baseIdx = i * 4;
        const x0 = intArr[baseIdx];
        const y0 = intArr[baseIdx + 1];
        const x1 = intArr[baseIdx + 2];
        const y1 = intArr[baseIdx + 3];
        console.log(`[draw] Drawing rect #${i}:`, { x0, y0, x1, y1 });
        ctx.beginPath();
        ctx.rect(x0, y0, x1 - x0, y1 - y0);
        if (fillColor && fillColor !== "") {
            ctx.fill();
        }
        ctx.stroke();
    }
}

function drawPatch(patchData, ctx) {
    if (patchData && patchData.data) {
        const { x, y, width, height, data } = patchData;
        const patchImageData = ctx.createImageData(width, height);
        let uint8Array;
        if (data instanceof ArrayBuffer) {
            uint8Array = new Uint8Array(data);
        } else if (data.buffer) {
            uint8Array = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        } else {
            uint8Array = new Uint8Array(data);
        }
        if (uint8Array.length > 0) {
            patchImageData.data.set(uint8Array);
            ctx.putImageData(patchImageData, x, y);
        }
    }
}

function clear(ctx) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

// --- Draw Command API ---
//
// Each entry in the _buffer array is a "draw command" object.
// The following command types are supported:
//
// 1. Draw Rectangle
//    {
//      type: 'rect',
//      rects: ArrayBuffer | DataView, // int32 array, shape [N, 4] (x0, y0, x1, y1) for N rectangles
//      count: number,                 // number of rectangles (N)
//      layer: number                  // (optional) layer index (default 0)
//    }
//
// 2. Clear Layer
//    {
//      type: 'clear',
//      layer: number                  // (optional) layer index (default 0)
//    }
//
// 3. Draw Patch (Image Data)
//    {
//      type: 'patch',
//      x: number,                     // x coordinate (left)
//      y: number,                     // y coordinate (top)
//      width: number,                 // width of patch
//      height: number,                // height of patch
//      data: ArrayBuffer | TypedArray // RGBA bytes, length = width * height * 4
//      layer: number                  // (optional) layer index (default 0)
//    }
//
// 4. Set Context Property
//    {
//      type: 'set',
//      name: string,                  // context property name, e.g. 'strokeStyle', 'lineWidth', 'fillStyle'
//      value: any,                    // value to set
//      layer: number                  // (optional) layer index (default 0)
//    }
//
// 5. Save Context State
//    {
//      type: 'save',
//      layer: number                  // (optional) layer index (default 0)
//    }
//
// 6. Restore Context State
//    {
//      type: 'restore',
//      layer: number                  // (optional) layer index (default 0)
//    }
//
// All commands may include a 'layer' property to specify which canvas layer to use.
// The buffer is expected to be ordered by layer, but this is not required for correctness.

function draw(model, contexts) {
    const buffer = model.get("_buffer");
    // Stable sort by layer (default 0)
    const sortedBuffer = buffer.slice().sort((a, b) => (a.layer || 0) - (b.layer || 0));
    for (const command of sortedBuffer) {
        const layer = command.layer || 0;
        const ctx = contexts[layer] || contexts[0];
        switch (command.type) {
            case 'set':
                ctx[command.name] = command.value;
                break;
            case 'rect':
                drawRect(command, ctx, model);
                break;
            case 'clear':
                clear(ctx);
                break;
            case 'patch':
                drawPatch(command, ctx);
                break;
            case 'save':
                ctx.save();
                break;
            case 'restore':
                ctx.restore();
                break;
            default:
                console.log('[draw] Unknown command:', command);
                break;
        }
    }
    model.set("_buffer", []);
    model.save_changes();
}

// --- Main Render ---
function render({ model, el }) {
    // Create inner wrapper for canvas positioning
    const wrapper = document.createElement("div");
    wrapper.classList.add("multicanvas-wrapper");
    // Array to hold all canvas elements and contexts
    const canvases = [];
    const contexts = [];
    const numLayers = model.get("layers") || 1;
    for (let i = 0; i < numLayers; i++) {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        canvas.classList.add("multicanvas-canvas");
        ctx.imageSmoothingEnabled = false;
        canvases.push(canvas);
        contexts.push(ctx);
        wrapper.appendChild(canvas);
    }
    el.appendChild(wrapper);

    // Initial render - just set up the canvas dimensions
    updateStyles(model, wrapper);
    updateCanvas(model, canvases);

    // Setup mouse events
    const cleanupMouse = setupMouseEvents(model, wrapper, canvases);

    // ResizeObserver to track actual rendered size
    let resizeObserver;
    if (typeof ResizeObserver !== 'undefined') {
        resizeObserver = new ResizeObserver(entries => {
            for (let entry of entries) {
                if (entry.target === wrapper) {
                    updateClientSize(model, wrapper);
                }
            }
        });
        resizeObserver.observe(wrapper);
    }

    // Listen for changes
    model.on("change:width", () => {
        updateCanvas(model, canvases);
    });
    model.on("change:height", () => {
        updateCanvas(model, canvases);
    });
    model.on("change:_buffer", () => {
        draw(model, contexts);
    });
    model.on("change:css_width change:css_height", () => {
        updateStyles(model, wrapper);
    });

    // Process any patches that were buffered
    const buffer = model.get("_buffer");
    if (buffer && buffer.length > 0) {
        draw(model, contexts);
    }

    // Return cleanup
    return () => {
        if (resizeObserver) {
            resizeObserver.disconnect();
        }
        cleanupMouse();
    };
}
export default { render }; 