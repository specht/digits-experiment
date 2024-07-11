importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest");

var queue = [];
var model = null;

tf.loadLayersModel('05-model/model.json').then(function(m) {
    model = m;
    console.log("Loaded model in web worker!");
    self.addEventListener('message', function(e) {
        let imageData = e.data.imageData;
        queue.push(imageData);
        this.setTimeout(handleQueue, 0);
    });
});

function handleQueue() {
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

    // crop the image data to the bounding box
    let c = new OffscreenCanvas(imageData.width, imageData.height);
    let ctx = c.getContext('2d');
    ctx.putImageData(imageData, 0, 0);

    let cropped = ctx.getImageData(left, top, right - left, bottom - top);

    // create a temporary canvas to put the cropped image data
    let tempCanvas = new OffscreenCanvas(right - left, bottom - top);
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(cropped, 0, 0);

    // create a new canvas to scale the image
    let canvasScaled = new OffscreenCanvas(28, 28);
    let ctxScaled = canvasScaled.getContext('2d');
    ctxScaled.fillStyle = 'white';
    ctxScaled.fillRect(0, 0, 28, 28);

    // calculate the width and height of the cropped image
    let croppedWidth = right - left;
    let croppedHeight = bottom - top;

    // calculate the scaling factors
    let scaleX = 26 / croppedWidth;
    let scaleY = 26 / croppedHeight;

    // use the smaller of the two scaling factors
    let scale = Math.min(scaleX, scaleY);

    // calculate the size of the scaled image
    let scaledWidth = croppedWidth * scale;
    let scaledHeight = croppedHeight * scale;

    // calculate the position to center the image
    let posX = (28 - scaledWidth) / 2;
    let posY = (28 - scaledHeight) / 2;

    // draw the cropped image on the new canvas, scaling it down and centering it
    ctxScaled.drawImage(tempCanvas, 0, 0, tempCanvas.width, tempCanvas.height, posX, posY, scaledWidth, scaledHeight);

    // // get the image data from the scaled canvas
    let finalImageData = ctxScaled.getImageData(0, 0, 28, 28);
    // self.postMessage({update_pixels: finalImageData});
    // return;

    const sourceCanvas = new OffscreenCanvas(28, 28);
    const sourceCtx = sourceCanvas.getContext('2d');

    // disable antialiasing
    sourceCtx.imageSmoothingEnabled = false;

    // render the finalImageData to the source canvas
    sourceCtx.putImageData(finalImageData, 0, 0);

    var img = tf.browser.fromPixels(sourceCanvas, 1);

    // let x = tf.cast(img.reshape([1, 28, 28]), 'float32');
    // if (model.layers[0].inputSpec[0].shape.length === 4)
        // x = tf.cast(img.reshape([1, 28, 28, 1]), 'float32');
    x = tf.cast(img.reshape(model.layers[0].inputSpec[0].shape.map((x) => x ?? 1)), 'float32');
    x = x.div(tf.scalar(255));

    let y = model.predict(x);
    const prediction = y.argMax(1).dataSync()[0];
    // console.log(y.argMax(1).dataSync()[0]);
    self.postMessage({prediction: prediction});

    let dot_colors = [];

    for (let layer = 0; layer < model.layers.length; layer++) {
        let layer_colors = [];
        let hiddenLayer = model.getLayer(null, layer);
        let hiddenModel = tf.model({inputs: model.inputs, outputs: hiddenLayer.output});
        let hiddenOutput = hiddenModel.predict(x).dataSync();
        for (let i = 0; i < hiddenOutput.length; i++) {
            let value = hiddenOutput[i];
            // if (layer > 1) value = 1.0 - 1.0 / Math.exp(value);
            // if (value < 0.0) value = 0.0;
            // if (value > 1.0) value = 1.0;
            if (value < -1.0) value = -1.0;
            if (value > 1.0) value = 1.0;
            // if (layer == model.layers.length - 1) value = 1.0 - value;

            layer_colors.push(value);
        }
        dot_colors.push(layer_colors);
    }
    self.postMessage({dot_colors: dot_colors});

    this.setTimeout(handleQueue, 0);
}