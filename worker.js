importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");
// ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4; // optional
// ort.env.wasm.simd = true;
// ort.env.wasm.proxy = true;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

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

var queue = [];
var model = null;

ort.InferenceSession.create("model.onnx", { executionProviders: ['wasm'] }).then(function (m) {
    model = new Model(m);
    console.log("Loaded model in web worker!");
    self.addEventListener('message', function (e) {
        let imageData = e.data.imageData;
        queue.push(imageData);
        this.setTimeout(handleQueue, 0);
    });
});

async function handleQueue() {
    if (queue.length === 0) return;

    let imageData = queue.pop();
    queue = [];
    let data = imageData.data;
    let left = imageData.width, right = 0, top = imageData.height, bottom = 0;

    left = 0; top = 0; right = imageData.width - 1; bottom = imageData.height - 1;
    while (left < right) {
        let found = false;
        for (let y = top; y <= bottom; y++) {
            let index = (y * imageData.width + left) * 4;
            if (data[index + 0] < 128) {
                found = true;
                break;
            }
        }
        if (found) break;
        left++;
    }
    while (right > left) {
        let found = false;
        for (let y = top; y <= bottom; y++) {
            let index = (y * imageData.width + right) * 4;
            if (data[index + 0] < 128) {
                found = true;
                break;
            }
        }
        if (found) break;
        right--;
    }
    while (top < bottom) {
        let found = false;
        for (let x = left; x <= right; x++) {
            let index = (top * imageData.width + x) * 4;
            if (data[index + 0] < 128) {
                found = true;
                break;
            }
        }
        if (found) break;
        top++;
    }
    while (bottom > top) {
        let found = false;
        for (let x = left; x <= right; x++) {
            let index = (bottom * imageData.width + x) * 4;
            if (data[index + 0] < 128) {
                found = true;
                break;
            }
        }
        if (found) break;
        bottom--;
    }

    if (right <= left) right = left + 1;
    if (bottom <= top) bottom = top + 1;

    let c = new OffscreenCanvas(imageData.width, imageData.height);
    let ctx = c.getContext('2d');
    ctx.putImageData(imageData, 0, 0);

    let cropped = ctx.getImageData(left, top, right - left, bottom - top);

    let tempCanvas = new OffscreenCanvas(right - left, bottom - top);
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(cropped, 0, 0);

    let canvasScaled = new OffscreenCanvas(28, 28);
    let ctxScaled = canvasScaled.getContext('2d');
    ctxScaled.fillStyle = 'white';
    ctxScaled.fillRect(0, 0, 28, 28);

    let croppedWidth = right - left;
    let croppedHeight = bottom - top;

    let scaleX = 26 / croppedWidth;
    let scaleY = 26 / croppedHeight;

    let scale = Math.min(scaleX, scaleY);

    let scaledWidth = croppedWidth * scale;
    let scaledHeight = croppedHeight * scale;

    let posX = (28 - scaledWidth) / 2;
    let posY = (28 - scaledHeight) / 2;

    ctxScaled.drawImage(tempCanvas, 0, 0, tempCanvas.width, tempCanvas.height, posX, posY, scaledWidth, scaledHeight);

    let finalImageData = ctxScaled.getImageData(0, 0, 28, 28);

    const sourceCanvas = new OffscreenCanvas(28, 28);
    const sourceCtx = sourceCanvas.getContext('2d');

    sourceCtx.imageSmoothingEnabled = false;

    sourceCtx.putImageData(finalImageData, 0, 0);

    const ctxData = sourceCtx.getImageData(0, 0, 28, 28).data;
    const pixels = new Float32Array(28 * 28);

    for (let i = 0; i < 28 * 28; i++) {
        pixels[i] = 1.0 - ctxData[i * 4] / 255.0;
    }

    let dim = [1, 28, 28];
    if (model.layers[0].outputShape.length === 4)
        dim.push(1);
    const inputTensor = new ort.Tensor("float32", pixels, dim);

    let t0 = performance.now();
    const results = await model._model.run({ [model.layers[0].name]: inputTensor });
    let t1 = performance.now();
    const outputTensor = results[model._model.outputNames[0]];
    const prediction = outputTensor.data.indexOf(Math.max(...outputTensor.data));

    self.postMessage({ prediction });

    let dot_colors = [];

    for (let layer = 0; layer < model.layers.length; layer++) {
        let layer_colors = [];
        let hiddenLayer = model.layers[layer];
        let min = 0.0;
        let max = 1.0;
        if (layer === 0 && croppedWidth > 1 && croppedHeight > 1) {
            for (let i = 0; i < 28 * 28; i++) {
                let value = pixels[i];
                // if (value < -1.0) value = -1.0;
                // if (value > 1.0) value = 1.0;
                if (value < min) min = value;
                if (value > max) max = value;
                layer_colors.push(value);
            }
        } else if (hiddenLayer.name in results) {
            min = 1e9;
            max = -1e9;
            let hiddenOutput = results[hiddenLayer.name].data;
            for (let i = 0; i < hiddenOutput.length; i++) {
                let value = hiddenOutput[i];
                // if (value < -1.0) value = -1.0;
                // if (value > 1.0) value = 1.0;
                if (value < min) min = value;
                if (value > max) max = value;
                layer_colors.push(value);
            }
        }
        // if (layer === 0 || layer === model.layers.length - 1)
        let scale = 1.0;
        if (Math.abs(min) > Math.abs(max)) scale = -1.0 / min; else scale = 1.0 / max;
        dot_colors.push({colors: layer_colors, min: min, max: max, scale: scale});
    }
    self.postMessage({ dot_colors: dot_colors, duration: t1 - t0 });

    this.setTimeout(handleQueue, 0);
}
