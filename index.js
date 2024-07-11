var model = null;
var color_pos = [9, 119, 189];
var color_neg = [245, 148, 37];
let DIGITS = '0123456789';

async function load() {
  model = await tf.loadLayersModel('05-model/model.json');
  console.log("Loaded model!");
  resize();
  update();
}

let worker = new Worker(`worker.js?${Math.random()}`);
let drawn_something = false;

worker.addEventListener('message', function(e) {
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
    ctx.fillStyle = 'black';
    for (let layer_index = 0; layer_index < model.layers.length; layer_index++) {
      for (let i = 0; i < unit_counts[layer_index]; i++) {
        let box = box_for_unit[layer_index][i];
        let value = e.data.dot_colors[layer_index][i];
        let color = [];
        if (value < 0.0) {
          for (let i = 0; i < 3; i++)
            color.push(Math.round(color_neg[i] * -value + 255 * (1.0 - -value)));
        } else {
          for (let i = 0; i < 3; i++)
            color.push(Math.round(color_pos[i] * value + 255 * (1.0 - value)));
        }
        ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
        ctx.fillRect(box[0] + 1, box[1] + 1, box[2] - 2, box[3] - 2);
      }
    }
  }
});

function update() {
  const canvas = document.getElementById('drawingCanvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  let imageData = ctx.getImageData(0, 0, 256, 256);
  worker.postMessage({imageData: imageData});
  if (!drawn_something)
    setTimeout(update, 100);
}

