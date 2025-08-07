var model = null;
var color_pos = [9, 119, 189];
var color_neg = [245, 148, 37];
let DIGITS = '0123456789';

class Model {
    constructor(_model) {
        this._model = _model;
        this.layers = [];
        this.layers.push({
            name: _model.inputMetadata[0].name,
            outputShape: _model.inputMetadata[0].shape,
        });
        for (let i = 1; i < _model.outputMetadata.length - 1; i++) {
            if (Number.isInteger(_model.outputMetadata[i].shape[1])) {
                let shape = _model.outputMetadata[i].shape;
                for (let k = 0; k < shape.length; k++) {
                    if (typeof shape[k] === 'string') {
                        shape[k] = 1;
                    }
                }
                shape = [shape[0], ...shape.slice(1).reverse()];
                this.layers.push({
                    name: _model.outputMetadata[i].name,
                    outputShape: shape,
                });
            }
        }
        let shape = _model.outputMetadata[0].shape;
        for (let k = 0; k < shape.length; k++) {
            if (typeof shape[k] === 'string') {
                shape[k] = 1;
            }
        }
        shape = [shape[0], ...shape.slice(1).reverse()];
        this.layers.push({
            name: _model.outputMetadata[0].name,
            outputShape: shape,
        });
    }
}

async function load() {
    let _model = await ort.InferenceSession.create("model.onnx");
    model = new Model(_model);
    console.log("Loaded model!");
    resize();
    update();
}

let worker = new Worker(`worker.js?${Math.random()}`);
let drawn_something = false;
let durations = [];

worker.addEventListener('message', function (e) {
    if ('prediction' in e.data) {
        let prediction = e.data.prediction;
        let pos = box_for_unit[model.layers.length - 1][prediction];
        document.getElementById('result').style.left = '' + (pos[0] + pos[2] / 2) + 'px';
        document.getElementById('result').style.top = '' + (pos[1] + pos[3] / 2) + 'px';
        document.getElementById('result').childNodes[0].textContent = `${DIGITS[prediction]}`;
    }
    if ('dot_colors' in e.data) {
        drawn_something = true;
        const canvas = document.getElementById('graph');
        const ctx = canvas.getContext('2d');
        ctx.beginPath();
        ctx.save();
        ctx.scale(SCALE, SCALE);
        ctx.fillStyle = 'black';
        for (let layer_index = 0; layer_index < model.layers.length; layer_index++) {
            for (let i = 0; i < unit_counts[layer_index]; i++) {
                let box = box_for_unit[layer_index][i];
                let value = (e.data.dot_colors[layer_index].colors[i] ?? 0.0) * e.data.dot_colors[layer_index].scale;
                let color = [];
                if (value < 0.0) {
                    for (let i = 0; i < 3; i++)
                        color.push(Math.round(color_neg[i] * -value + 255 * (1.0 - -value)));
                } else {
                    for (let i = 0; i < 3; i++)
                        color.push(Math.round(color_pos[i] * value + 255 * (1.0 - value)));
                }
                ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
                ctx.fillRect(box[0] + 0.5, box[1] + 0.5, box[2], box[3]);
            }
        }
        ctx.restore();
        durations.unshift(e.data.duration);
        durations = durations.slice(0, 10);
        let mean = 0;
        for (let d of durations) mean += d;
        mean /= durations.length;
        document.getElementById('stats').innerHTML = `Inference: ${mean.toFixed(2)} ms.`;
    }
});

function update() {
    let imageData = ctx.getImageData(0, 0, 256, 256);
    worker.postMessage({ imageData: imageData });
    if (!drawn_something)
        setTimeout(update, 100);
}

var box_for_unit = [];
var unit_counts = [];

function calc_pos(i, sub_width, sub_height, sub_unit_count, cols, s, layer_index, layer_width, layer_height, margin, field_offset_x, field_offset_y) {
    let k = i;
    let pz = Math.floor(k / (sub_height * sub_width));
    k -= pz * sub_height * sub_width;
    let py = Math.floor(k / (sub_width)) % sub_height;
    k -= py * sub_width;
    let px = k;
    px = px * s;
    py = py * s;
    let gap_x = s * (sub_width + 0.5);
    let gap_y = s * (sub_height + 0.5);
    gap_x = s * sub_width + 4.0;
    gap_y = s * sub_height + 4.0;
    px += (pz % cols) * gap_x;
    py += Math.floor(pz / cols) * gap_y;
    px += layer_width * layer_index;
    px += field_offset_x;
    py += Math.max(field_offset_y, (layer_index === 0) ? 300 : 0);
    px += margin;
    py += margin;
    return [px, py];
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

const SCALE = 2;
var color_pos = [9, 119, 189];
var color_neg = [245, 148, 37];

function resize() {
    const graphCanvas = document.getElementById('graph');
    graphCanvas.width = window.innerWidth * SCALE;
    graphCanvas.height = window.innerHeight * SCALE;
    const graphCtx = graphCanvas.getContext('2d');
    //   graphCtx.fillStyle = 'white';
    //   graphCtx.fillRect(0, 0, graphCanvas.width, graphCanvas.height);

    const overlayCanvas = document.getElementById('overlay');
    overlayCanvas.width = window.innerWidth * SCALE;
    overlayCanvas.height = window.innerHeight * SCALE;
    const overlayCtx = overlayCanvas.getContext('2d');
    overlayCtx.fillStyle = 'rgba(255, 255, 255, 0)';
    overlayCtx.fillRect(0, 0, graphCanvas.width, graphCanvas.height);

    for (let c of ['dot', 'label', 'frame']) {
        let elements = document.getElementsByClassName(c);
        Array.from(elements).forEach((element) => {
            element.parentNode.removeChild(element);
        });
    }

    let margin = 20;
    let layer_width = (window.innerWidth) / (model.layers.length);
    let layer_height = (window.innerHeight) - 50;
    console.log(`Got model with ${model.layers.length} layers.`);
    unit_counts = [];
    box_for_unit = [];
    for (let layer_index = 0; layer_index < model.layers.length; layer_index++) {
        let layer_boxes = [];
        let layer = model.layers[layer_index];
        let shape = layer.outputShape.slice(1);
        if (shape.length <= 3) {
            // one sub unit is sub_width x sub_height units
            let sub_width = 1;
            let sub_height = 1;
            let sub_unit_count = 1;
            // sub units are arranged in cols columns
            let cols = 1;
            if (shape.length >= 2) {
                sub_width = shape[0];
                sub_height = shape[1];
            }
            if (shape.length >= 3) {
                sub_unit_count = shape[2];
            }
            if (shape.length === 1) {
                sub_unit_count = shape[0];
            }
            // s is the size of a single unit in pixels
            let s = 24;
            let count = shape[0];
            if (shape.length > 1) count *= shape[1];
            if (shape.length > 2) count *= shape[2];
            // count is the total number of units
            unit_counts.push(count);
            let result = calc_pos(count - 1, sub_width, sub_height, sub_unit_count, cols, s, 1, 0, 0, 0, 0, 0);
            while (s > 2 && result[1] + margin * 2 > layer_height || result[0] + margin * 2 > layer_width) {
                if ((sub_width + 1) * s * (cols + 1) + margin * 2 < layer_width) {
                    cols *= 2;
                } else {
                    s -= 1;
                }
                result = calc_pos(count - 1, sub_width, sub_height, sub_unit_count, cols, s, 1, 0, 0, 0, 0, 0);
            }
            if (cols * s + margin * 2 > layer_width) {
                cols = Math.floor((layer_width - margin * 2) / s);
            }

            result = calc_pos(count - 1, sub_width, sub_height, sub_unit_count, cols, s, 1, 0, 0, 0, 0, 0);
            let field_width = result[0];
            let field_height = result[1];

            let field_offset_x = (layer_width - 2 * margin - field_width) / 2;
            let field_offset_y = (layer_height - 2 * margin - field_height) / 2;

            while (shape.length < 3)
                shape.push(1);

            console.log(`Layer ${layer_index}: ${sub_unit_count} total sub units @ ${sub_width}x${sub_height}, ${s}x${s} pixels, ${cols} columns (${field_width}x${field_height})`);

            for (let i = 0; i < count; i++) {
                result = calc_pos(i, sub_width, sub_height, sub_unit_count, cols, s, layer_index, layer_width, layer_height, margin, field_offset_x, field_offset_y);
                px = result[0];
                py = result[1];
                let s2 = s;
                if (s < 8) {
                    px -= 1;
                    py -= 1;
                    s2 += 2;
                }
                px -= s2 / 2;
                py -= s2 / 2;
                layer_boxes.push([Math.round(px) + 0.5, Math.round(py) + 0.5, Math.round(s2 - 1), Math.round(s2 - 1)]);
            }
            let frame = document.createElement('div');
            frame.classList.add('frame');
            frame.style = `left: ${layer_width * layer_index}px; top: 0px; width: ${layer_width}px; height: ${layer_height}px;`;
            document.body.appendChild(frame);
            let label = document.createElement('div');
            label.classList.add('label');
            label.style = `left: ${layer_width * layer_index}px; top: ${layer_height}px; width: ${layer_width}px; height: ${layer_height}px;`;
            let name = (model.layers[layer_index] ?? {}).name.split('/');
            try {
                name = name[name.length - 2].split(':')[0];
                name = name.split('_').filter((x) => `${parseInt(x)}` !== x).join('_');
            } catch {
                name = '(unknown)';
            }
            if (layer_index === 0) name = 'input';
            if (layer_index === model.layers.length - 1) name = 'output';
            label.innerHTML = `<strong>${name}</strong>`;
            label.innerHTML += "<br>";
            label.innerHTML += `(${model.layers[layer_index].outputShape.slice(1).filter((x) => x !== null).map((x) => x ?? '?').join('×')})`;
            document.body.appendChild(label);
        } else {
            console.log(`${layer.name}: Unknown shape!`);
            unit_counts.push(0);
        }
        box_for_unit.push(layer_boxes);
    }

    graphCtx.save();
    overlayCtx.save();
    overlayCtx.beginPath();
    //   overlayCtx.translate(0.5, 0.5);
    graphCtx.scale(SCALE, SCALE);
    overlayCtx.scale(SCALE, SCALE);
    for (let layer_index = 0; layer_index < model.layers.length; layer_index++) {
        let x0 = 1e9;
        let y0 = 1e9;
        let x1 = -1e9;
        let y1 = -1e9;
        let drawn_something = false;

        for (let i = 0; i < unit_counts[layer_index]; i++) {
            let p = box_for_unit[layer_index][i];
            if (p[0] < x0) x0 = p[0];
            if (p[1] < y0) y0 = p[1];
            if (p[0] + p[2] > x1) x1 = p[0] + p[2];
            if (p[1] + p[3] > y1) y1 = p[1] + p[3];
            if (p[3] > 10) {
                drawn_something = true;
                let opacity = p[2] / 10.0 * 0.3;
                if (opacity > 0.3) opacity = 0.3;
                overlayCtx.strokeStyle = `rgba(0, 0, 0, ${opacity})`;
                overlayCtx.strokeRect(p[0], p[1], p[2], p[3]);
            }
        }
        // if (!drawn_something) {
        graphCtx.fillStyle = `#ffffffa0`;
        drawRoundedRect(graphCtx, x0 - 8, y0 - 8, x1 - x0 + 16, y1 - y0 + 16, 8);
        graphCtx.fill();

        // graphCtx.fillRect(x0 - 8, y0 - 8, x1 - x0 + 16, y1 - y0 + 16);
        overlayCtx.strokeStyle = `rgba(0, 0, 0, 0.2)`;
        // overlayCtx.strokeRect(x0 - 8, y0 - 8, x1 - x0 + 16, y1 - y0 + 16);
        drawRoundedRect(overlayCtx, x0 - 8, y0 - 8, x1 - x0 + 16, y1 - y0 + 16, 8);
        overlayCtx.stroke();

        // }
    }
    overlayCtx.restore();
    graphCtx.restore();
    //   overlayCtx.beginPath();
    //   for (let i = 0; i < box_for_unit.length - 1; i++) {
    //     for (let a = 0; a < box_for_unit[i].length; a++) {
    //         for (let b = 0; b < box_for_unit[i + 1].length; b++) {
    //             let p = box_for_unit[i][a];
    //             let q = box_for_unit[i + 1][b];
    //             overlayCtx.moveTo(p[0], p[1]);
    //             overlayCtx.strokeStyle = `rgba(0, 0, 0, ${Math.random() * 0.1})`;
    //             overlayCtx.lineTo(q[0], q[1]);
    //         }
    //     }
    //   }
    //   overlayCtx.stroke();
    let allLines = [];

    for (let i = 0; i < box_for_unit.length - 1; i++) {
        let ac = [];
        for (let a = 0; a < box_for_unit[i].length; a++) ac.push(a);
        for (let _ = ac.length - 1; _ > 0; _--) {
            const j = Math.floor(Math.random() * (_ + 1));
            [ac[_], ac[j]] = [ac[j], ac[_]];
        }
        ac = ac.slice(0, 100);
        let bc = [];
        for (let b = 0; b < box_for_unit[i + 1].length; b++) bc.push(b);
        for (let _ = bc.length - 1; _ > 0; _--) {
            const j = Math.floor(Math.random() * (_ + 1));
            [bc[_], bc[j]] = [bc[j], bc[_]];
        }
        bc = bc.slice(0, 100);

        for (let a = 0; a < ac.length; a++) {
            for (let b = 0; b < bc.length; b++) {
                allLines.push({
                    from: box_for_unit[i][ac[a]],
                    to: box_for_unit[i + 1][bc[b]],
                });
            }
        }
    }

    for (let i = allLines.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [allLines[i], allLines[j]] = [allLines[j], allLines[i]];
    }

    const batchSize = Math.ceil(allLines.length / 10);

    let index = [];
    for (let i = 0; i < 10; i++) index.push(i);
    for (let i = index.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [index[i], index[j]] = [index[j], index[i]];
    }

    overlayCtx.save();
    overlayCtx.scale(SCALE, SCALE);
    for (let i = 0; i < 10; i++) {
        let level = index[i];
        const alpha = Math.abs(level / 10.0) * 0.1;
        const start = level * batchSize;
        const end = Math.min(start + batchSize, allLines.length);

        overlayCtx.beginPath();
        overlayCtx.strokeStyle = `rgba(0, 0, 0, ${alpha})`;

        for (let i = start; i < end; i++) {
            const line = allLines[i];
            let x0 = line.from[0] + line.from[2] / 2;
            let y0 = line.from[1] + line.from[3] / 2;
            let x1 = line.to[0] + line.to[2] / 2;
            let y1 = line.to[1] + line.to[3] / 2;
            if (Math.round(y0) !== Math.round(y1)) {
                overlayCtx.moveTo(x0, y0);
                overlayCtx.lineTo(x1, y1);
            }
        }

        overlayCtx.stroke();
    }
    overlayCtx.restore();

    console.log('done drawing!');

    update();
}

let canvas = null;
let ctx = null;

window.addEventListener('load', function () {
    canvas = document.getElementById('drawingCanvas');
    ctx = canvas.getContext('2d', { willReadFrequently: true });
    load();
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    let isDrawing = false;

    let isUpdateScheduled = false;

    function scheduleUpdate() {
        if (!isUpdateScheduled) {
            isUpdateScheduled = true;
            requestAnimationFrame(() => {
                update();
                isUpdateScheduled = false;
            });
        }
    }

    canvas.addEventListener('mousedown', function (e) {
        isDrawing = true;
        ctx.beginPath();
        if (e.button === 0) {
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 20;
        } else if (e.button === 2) {
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 40;
        }
        ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
        ctx.lineTo(e.clientX - canvas.offsetLeft + 0.1, e.clientY - canvas.offsetTop + 0.1);
        ctx.stroke();
        scheduleUpdate();
        e.preventDefault();
    });

    canvas.addEventListener("contextmenu", e => e.preventDefault());

    canvas.addEventListener('mousemove', function (e) {
        if (!isDrawing) return;
        ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
        ctx.stroke();
        scheduleUpdate();
        e.preventDefault();
    });

    window.addEventListener('mouseup', function (e) {
        isDrawing = false;
        e.preventDefault();
    });

    canvas.addEventListener('touchstart', function (e) {
        isDrawing = true;
        ctx.beginPath();
        ctx.moveTo(e.touches[0].clientX - canvas.offsetLeft, e.touches[0].clientY - canvas.offsetTop);
        ctx.lineTo(e.touches[0].clientX - canvas.offsetLeft + 0.1, e.touches[0].clientY - canvas.offsetTop + 0.1);
        ctx.stroke();
        scheduleUpdate();
        e.preventDefault();
    });

    canvas.addEventListener('touchmove', function (e) {
        if (!isDrawing) return;
        ctx.lineTo(e.touches[0].clientX - canvas.offsetLeft, e.touches[0].clientY - canvas.offsetTop);
        ctx.stroke();
        scheduleUpdate();
        e.preventDefault();
    });

    window.addEventListener('touchend', function () {
        isDrawing = false;
    });

    window.addEventListener('resize', function () {
        resize();
    });

    document.getElementById('bu_clear').addEventListener('click', function () {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        update();
    });
});