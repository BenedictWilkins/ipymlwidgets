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
    const size = model.get("size");
    const width = size[0];
    const height = size[1];
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

function drawRect(command, ctx, offCtx) {
    if (!command.data) return;
    const { data, count, pixel_perfect } = command;
    if (!data || !count) {
        //console.log('[draw] No data or count');
        return;
    }
    let buffer;
    if (data instanceof ArrayBuffer) {
        buffer = data;
    } else if (data instanceof DataView) {
        buffer = data.buffer;
    } else {
        //console.log('[draw] data is not ArrayBuffer or DataView:', data);
        return;
    }

    const intArr = new Int32Array(buffer);

    if (pixel_perfect) {
        const border = ctx.lineWidth || 1;
        const strokeStyle = ctx.strokeStyle;
        const fillStyle = ctx.fillStyle;
        const isTransparent = (
            fillStyle === "transparent" ||
            /^rgba?\(\s*0\s*,\s*0\s*,\s*0\s*,\s*0\s*\)$/.test(fillStyle)
        );
        // clear the offscreen canvas
        offCtx.clearRect(0, 0, offCtx.canvas.width, offCtx.canvas.height);
        for (let i = 0; i < count; ++i) {
            const baseIdx = i * 4;
            const x0 = intArr[baseIdx];
            const y0 = intArr[baseIdx + 1];
            const x1 = intArr[baseIdx + 2];
            const y1 = intArr[baseIdx + 3];

            // Inclusive: width/height is (x1-x0)+1, (y1-y0)+1
            const w = x1 - x0;
            const h = y1 - y0;

            // Clear just the region at the top-left of the offscreen canvas
            // offCtx.clearRect(0, 0, w, h);
            offCtx.clearRect(0, 0, offCtx.canvas.width, offCtx.canvas.height);


            // Draw border as filled rect using strokeStyle at (0, 0)
            offCtx.fillStyle = strokeStyle;
            offCtx.fillRect(0, 0, w, h);

            // Clear the inner rect for transparency
            if (w > 2 * border && h > 2 * border) {
                offCtx.clearRect(
                    border,
                    border,
                    w - 2 * border,
                    h - 2 * border
                );
                // If fillStyle is not fully transparent, fill the inner rect
                if (!isTransparent) {
                    offCtx.fillStyle = fillStyle;
                    offCtx.fillRect(
                        border,
                        border,
                        w - 2 * border,
                        h - 2 * border
                    );
                }
            }
            // Blit just the region for this rect from (0,0) to (x0,y0)
            ctx.drawImage(
                offCtx.canvas,
                0, 0, w, h, // source rect
                x0, y0, w, h  // destination rect
            );
        }
    } else {
        //console.log('[draw] Decoded intArr:', intArr, 'count:', count);
        //console.log('[draw] ctx:', ctx.fillStyle, ctx.lineWidth, ctx.strokeStyle);
        for (let i = 0; i < count; ++i) {
            const baseIdx = i * 4;
            const x0 = intArr[baseIdx];
            const y0 = intArr[baseIdx + 1];
            const x1 = intArr[baseIdx + 2];
            const y1 = intArr[baseIdx + 3];
            //console.log(`[draw] Drawing rect #${i}:`, { x0, y0, x1, y1 });
            ctx.beginPath();
            ctx.rect(x0, y0, x1 - x0, y1 - y0);
            ctx.fill();
            ctx.stroke();
        }
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

function clear(data, ctx) {
    const { xyxy } = data;
    if (xyxy) {
        const x0 = xyxy[0];
        const y0 = xyxy[1];
        const x1 = xyxy[2];
        const y1 = xyxy[3];
        ctx.clearRect(x0, y0, x1 - x0, y1 - y0);
    } else {
        //console.log('[clear]', data);
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }
}


// --- Draw Command API ---
//
// Each entry in the _buffer array is a "draw command" object.
// The following command types are supported:
//
// 1. Draw Rectangle
//    {
//      type: 'draw'
//      shape: 'rect',
//      data: ArrayBuffer | DataView, // int32 array, shape [N, 4] (x0, y0, x1, y1) for N rectangles
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
//      layer: number                  // layer to set property on
//    }
//
// 5. Save Context State
//    {
//      type: 'set',
//      name: 'save',
//      layer: number                 
//    }
//
// 6. Restore Context State
//    {
//      type: 'set',
//      name: 'restore',
//      layer: number                  
//    }
//
// All commands may include a 'layer' property to specify which canvas layer to use.
// The buffer is expected to be ordered by layer, but this is not required for correctness.

function draw(buffer, contexts, offCtx) {
    try {
        while (buffer.length > 0) {
            const command = buffer.shift();
            const layer = command.layer || 0;
            const ctx = contexts[layer];
            switch (command.type) {
                case 'set':
                    if (typeof ctx[command.name] === 'function') {
                        ctx[command.name]();
                        //console.log(`[set ${command.layer}]: called ${command.name}()`);
                    } else {
                        ctx[command.name] = command.value;
                        //console.log(`[set ${command.layer}]: set ${command.name} =`, ctx[command.name]);
                    }
                    break;
                case 'draw':
                    switch (command.shape) {
                        case 'rect':
                            //console.log(`[draw rect ${command.layer}]`, command.count, ctx.fillStyle, ctx.lineWidth, ctx.strokeStyle);
                            drawRect(command, ctx, offCtx);
                            break;
                        default:
                            c//onsole.log('[draw] Unknown shape:', command.shape);
                            break;
                    }
                    break;
                case 'clear':
                    clear(command, ctx);
                    //console.log(`[clear ${command.layer}]`);
                    break;
                case 'patch':
                    drawPatch(command, ctx);
                    //console.log(`[patch ${command.layer}]`, command.x, command.y, command.width, command.height);
                    break;
                case 'debug':
                    //console.log(`[debug]`, command.value);
                    break;
                default:
                    //console.log(`[unknown command ${command.layer}]`, command);
                    break;
            }
        }
    } catch (err) {
        //console.log("[canvas.js] draw error");
    }
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


    // offscreen canvas for blit drawing tricks
    const offCanvas = document.createElement('canvas');
    const offCtx = offCanvas.getContext('2d');

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
    model.on("change:size", () => {
        updateCanvas(model, canvases);
    });

    const buffer = []; // queue for incoming draw commands
    let scheduled = false;

    function scheduleDraw() {
        // grab any new commands from the backend buffer
        const commands = model.get("_buffer");
        buffer.push(...commands);
        // schedule a new draw, if not already scheduled
        if (!scheduled) {
            scheduled = true;
            requestAnimationFrame(() => {
                //console.log("[draw]", buffer.length);
                draw(buffer, contexts, offCtx);
                scheduled = false;
            });
        }
        // acknowledge commands were processed (scheduled for draw)
        const ack = Date.now();
        model.set("_buffer_ack", ack);
        model.save_changes();
    }


    model.on("change:_buffer_syn", () => {
        scheduleDraw();
    });
    model.on("change:css_width change:css_height", () => {
        updateStyles(model, wrapper);
    });

    scheduleDraw(); // schedule an initial draw 
    //console.log("[draw] initial draw complete!");

    // Return cleanup
    return () => {
        if (resizeObserver) {
            resizeObserver.disconnect();
        }
        cleanupMouse();
    };
}
export default { render }; 