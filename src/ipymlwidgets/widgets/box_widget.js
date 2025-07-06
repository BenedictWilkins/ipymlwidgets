export function render({ model, el }) {
    // Create container
    el.innerHTML = '';
    const container = document.createElement('div');
    container.style.display = 'grid';
    container.style.gridTemplateColumns = 'repeat(4, 1fr)';
    container.style.gap = '0.5em';
    container.style.width = '100%';

    // Create 4 input fields
    const inputs = Array.from({ length: 4 }, (_, i) => {
        const input = document.createElement('input');
        input.type = 'text';
        input.style.width = '100%';
        input.style.boxSizing = 'border-box';
        input.style.textAlign = 'center';
        input.autocomplete = 'off';
        input.inputMode = 'numeric';
        input.pattern = '[0-9]*';
        container.appendChild(input);
        return input;
    });

    // Sync from model to view
    function updateInputs() {
        const coords = model.get('box_coords');
        const disabled = coords === null;
        for (let i = 0; i < 4; ++i) {
            inputs[i].value = (coords && coords[i] != null) ? coords[i] : '';
            inputs[i].disabled = disabled;
        }
    }

    // Sync from view to model
    inputs.forEach((input, i) => {
        input.addEventListener('change', () => {
            let coords = model.get('box_coords');
            if (coords === null) {
                // If currently None, start with [null, null, null, null]
                coords = [null, null, null, null];
            } else {
                coords = coords.slice();
            }
            const val = input.value.trim();
            coords[i] = val === '' ? null : parseInt(val, 10);
            // If all are valid ints, set as a tuple; if any are null, keep as null or partial
            if (coords.every(v => typeof v === 'number' && !isNaN(v))) {
                model.set('box_coords', coords);
            } else if (coords.every(v => v === null)) {
                model.set('box_coords', null);
            } else {
                // Partial: keep as is, but do not set as a valid tuple
                model.set('box_coords', null);
            }
            model.save_changes();
        });
    });

    // Listen for traitlet changes
    model.on('change:box_coords', updateInputs);

    // Initial render
    updateInputs();
    el.appendChild(container);
} 