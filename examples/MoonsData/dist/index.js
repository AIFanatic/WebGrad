// src/index.ts
import { Tensor, Module, nn, MatrixToTensorBuffer, TensorBufferToMatrix } from "micrograd-ts";
import { Random } from "micrograd-ts";

// src/GraphUtils.ts
var GraphUtils = class {
  static PrintMatrix(m) {
    return `Tensor(shape=[${m.shape}], mean=${m.mean().get(0).toFixed(4)})`;
  }
  static trace(root) {
    const nodes = /* @__PURE__ */ new Set();
    const edges = /* @__PURE__ */ new Set();
    function build(v) {
      if (!nodes.has(v)) {
        nodes.add(v);
        for (let child of v._prev) {
          edges.add([child, v]);
          build(child);
        }
      }
    }
    build(root);
    return [nodes, edges];
  }
  static draw_dot(root) {
    let str = `digraph g {rankdir="TB"; nodesep=0.5;
`;
    const [nodes, edges] = GraphUtils.trace(root);
    for (let node of nodes) {
      const nodeName = node.id;
      str += `"${nodeName}" [label="{data (${GraphUtils.PrintMatrix(node.data)}) | grad (${GraphUtils.PrintMatrix(node.grad)})}"] [shape=record, width=5];
`;
      if (node._op !== "") {
        str += `"${nodeName + node._op}" [label="${node._op}"]
`;
        str += `"${nodeName + node._op}" -> "${nodeName}";
`;
      }
    }
    for (let edge of edges) {
      const n1 = edge[0];
      const n2 = edge[1];
      str += `"${n1.id}" -> "${n2.id + n2._op}"
`;
    }
    str += "}";
    var image = Viz(str, { format: "svg" });
    const imgElement = document.createElement("div");
    imgElement.innerHTML = image;
    return imgElement;
  }
};

// src/index.ts
var SimpleModel = class extends Module {
  constructor(input_sample, output_sample) {
    super();
    this.dense1 = new nn.Linear(input_sample, 16);
    this.dense2 = new nn.Linear(16, 16);
    this.dense3 = new nn.Linear(16, output_sample);
  }
  forward(x) {
    x = this.dense1.forward(x).tanh();
    x = this.dense2.forward(x).tanh();
    x = this.dense3.forward(x).tanh();
    return x;
  }
};
var MoonsDemo = class {
  constructor(lossElement, networkElement, trainingElement) {
    this.WIDTH = 512;
    this.HEIGHT = 512;
    this.RES = 32;
    this.SS = 100;
    Random.SetRandomSeed(1337);
    this.lossElement = lossElement;
    this.networkElement = networkElement;
    this.trainingElement = trainingElement;
    this.canvas = document.createElement("canvas");
    this.canvas.style.display = "inline-block";
    this.canvas.width = 512;
    this.canvas.height = 512;
    this.ctx = this.canvas.getContext("2d");
    this.trainingElement.appendChild(this.canvas);
    const [Xb, yb] = this.generateData(100);
    this.Xb = Xb;
    this.yb = yb;
    this.model = new SimpleModel(2, 1);
    this.XbPoints = this.pointsToPredict(this.WIDTH, this.HEIGHT, this.RES, this.SS);
    this.train();
  }
  // Generate moons data
  generateData(n) {
    var data = [];
    var labels = [];
    for (var i = 0; i < Math.PI; i += Math.PI * 2 / n) {
      var point_1 = [
        Math.cos(i) + Random.RandomRange(-0.1, 0.1),
        Math.sin(i) + Random.RandomRange(-0.1, 0.1)
      ];
      data.push(point_1);
      labels.push(-1);
      var point_2 = [
        1 - Math.cos(i) + Random.RandomRange(-0.1, 0.1),
        1 - Math.sin(i) + Random.RandomRange(-0.1, 0.1) - 0.5
      ];
      data.push(point_2);
      labels.push(1);
    }
    return [data, labels];
  }
  // Creates a tensor that has the points that are going to be predicted after each epoch
  pointsToPredict(width, height, res, ss) {
    let points = [];
    for (var x = 0, cx = 0; x <= width; x += res, cx++) {
      for (var y = 0, cy = 0; y <= height; y += res, cy++) {
        var point = [(x - width / 2) / ss, (y - height / 2) / ss];
        points.push(point);
      }
    }
    return new Tensor(points);
  }
  drawCircle(x, y, r) {
    this.ctx.beginPath();
    this.ctx.arc(x, y, r, 0, 2 * Math.PI);
    this.ctx.fill();
    this.ctx.closePath();
  }
  drawPredictionGrid() {
    console.time("forward");
    const preds = this.model.forward(this.XbPoints);
    console.timeEnd("forward");
    let p = 0;
    for (var x = 0, cx = 0; x <= this.WIDTH; x += this.RES, cx++) {
      for (var y = 0, cy = 0; y <= this.HEIGHT; y += this.RES, cy++) {
        const pred = preds.data.get(p);
        if (pred < 0)
          this.ctx.fillStyle = "#ff000050";
        else
          this.ctx.fillStyle = "#00ff0050";
        this.ctx.fillRect(
          x - this.RES / 2 - 1,
          y - this.RES / 2 - 1,
          this.RES + 2,
          this.RES + 2
        );
        p++;
      }
    }
  }
  drawGroundTruth() {
    for (var i = 0; i < this.Xb.length; i++) {
      if (this.yb[i] === 1)
        this.ctx.fillStyle = "rgb(100,200,100)";
      else
        this.ctx.fillStyle = "rgb(200,100,100)";
      this.drawCircle(
        this.Xb[i][0] * this.SS + this.WIDTH / 2,
        this.Xb[i][1] * this.SS + this.HEIGHT / 2,
        5
      );
    }
  }
  drawPredections() {
    this.ctx.clearRect(0, 0, this.WIDTH, this.HEIGHT);
    this.drawPredictionGrid();
    this.drawGroundTruth();
  }
  updateElements(step, loss) {
    this.lossElement.textContent = `step: ${step}, loss: ${loss}`;
    const image = GraphUtils.draw_dot(loss);
    this.networkElement.innerHTML = "";
    this.networkElement.appendChild(image);
  }
  train() {
    const epochs = 100;
    let k = 0;
    const interval = setInterval(() => {
      if (k == epochs)
        clearInterval(interval);
      const X = new Tensor(this.Xb);
      const y = new Tensor(this.yb).reshape([100, 1]);
      const pred = this.model.forward(X);
      const loss = pred.sub(y).pow(2).mean();
      this.model.zero_grad();
      loss.backward();
      const learning_rate = 0.5;
      for (let p of this.model.parameters()) {
        p.data = MatrixToTensorBuffer(0, TensorBufferToMatrix(p.data).sub(p.grad.mul(0.5)));
      }
      this.drawPredections();
      this.updateElements(k, loss);
      k++;
    }, 10);
  }
};
export {
  MoonsDemo
};
