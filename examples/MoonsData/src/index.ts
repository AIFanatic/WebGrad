import { Tensor, Module, nn, MatrixToTensorBuffer, TensorBufferToMatrix } from "micrograd-ts";
import { Random } from "micrograd-ts";

import { GraphUtils } from './GraphUtils.js';

class SimpleModel extends Module {
    private dense1: Module;
    private dense2: Module;
    private dense3: Module;

    constructor(input_sample: number, output_sample: number) {
        super();

        this.dense1 = new nn.Linear(input_sample, 16);
        this.dense2 = new nn.Linear(16, 16);
        this.dense3 = new nn.Linear(16, output_sample);
    }

    forward(x: Tensor): Tensor {
        x = this.dense1.forward(x).tanh()
        x = this.dense2.forward(x).tanh()
        x = this.dense3.forward(x).tanh()
        return x;
    }
}

export class MoonsDemo {
    private lossElement: HTMLSpanElement;
    private networkElement: HTMLDivElement;
    private trainingElement: HTMLDivElement;

    private model: Module;
    private Xb: number[];
    private yb: number[];
    private XbPoints: Tensor;
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;

    private WIDTH = 512;
    private HEIGHT = 512;
    private RES = 32;
    private SS = 100;

    constructor(lossElement: HTMLSpanElement, networkElement: HTMLDivElement, trainingElement: HTMLDivElement) {
        Random.SetRandomSeed(1337);

        this.lossElement = lossElement;
        this.networkElement = networkElement;
        this.trainingElement = trainingElement;
        
        // Create canvas
        this.canvas = document.createElement("canvas");
        this.canvas.style.display = "inline-block";
        this.canvas.width = 512;
        this.canvas.height = 512;
        this.ctx = this.canvas.getContext("2d");

        this.trainingElement.appendChild(this.canvas);

        
        // Create model
        const [Xb, yb] = this.generateData(100);
        this.Xb = Xb;
        this.yb = yb;
        this.model = new SimpleModel(2, 1);
        this.XbPoints = this.pointsToPredict(this.WIDTH, this.HEIGHT, this.RES, this.SS);
        
        this.train();
    }

    // Generate moons data
    private generateData(n: number): number[][] {
        var data: number[][] = [];
        var labels: number[] = [];
        for (var i = 0; i < Math.PI; i += (Math.PI * 2) / n) {
            var point_1 = [
                Math.cos(i) + Random.RandomRange(-0.1, 0.1),
                Math.sin(i) + Random.RandomRange(-0.1, 0.1),
            ];
            data.push(point_1);
            labels.push(-1);
    
            var point_2 = [
                1 - Math.cos(i) + Random.RandomRange(-0.1, 0.1),
                1 - Math.sin(i) + Random.RandomRange(-0.1, 0.1) - 0.5,
            ];
            data.push(point_2);
            labels.push(1);
        }
        return [data, labels];
    }

    // Creates a tensor that has the points that are going to be predicted after each epoch
    private pointsToPredict(width: number, height: number, res: number, ss: number): Tensor {
        let points: number[][] = [];
        for (var x = 0.0, cx = 0; x <= width; x += res, cx++) {
            for (var y = 0.0, cy = 0; y <= height; y += res, cy++) {
                var point = [(x - width / 2) / ss, (y - height / 2) / ss];
                points.push(point);
            }
        }
        return new Tensor(points);
    }

    private drawCircle(x: number, y: number, r: number) {
        this.ctx.beginPath();
        this.ctx.arc(x, y, r, 0, 2 * Math.PI);
        this.ctx.fill()
        this.ctx.closePath();
    }

    private drawPredictionGrid() {
        console.time("forward");
        const preds = this.model.forward(this.XbPoints);
        console.timeEnd("forward")
        let p = 0;
        // draw decisions in the grid
        for (var x = 0.0, cx = 0; x <= this.WIDTH; x += this.RES, cx++) {
            for (var y = 0.0, cy = 0; y <= this.HEIGHT; y += this.RES, cy++) {
                const pred = preds.data.get(p);
                if (pred < 0.0) this.ctx.fillStyle = '#ff000050';
                else this.ctx.fillStyle = '#00ff0050';
                this.ctx.fillRect(
                    x - this.RES / 2 - 1,
                    y - this.RES / 2 - 1,
                    this.RES + 2,
                    this.RES + 2,
                );

                p++;
            }
        }
    }

    private drawGroundTruth() {
        for (var i = 0; i < this.Xb.length; i++) {
            if (this.yb[i] === 1) this.ctx.fillStyle = 'rgb(100,200,100)';
            else this.ctx.fillStyle = 'rgb(200,100,100)';
            this.drawCircle(
                this.Xb[i][0] * this.SS + this.WIDTH / 2,
                this.Xb[i][1] * this.SS + this.HEIGHT / 2,
                5.0,
            );
        }
    }

    private drawPredections() {
        this.ctx.clearRect(0, 0, this.WIDTH, this.HEIGHT);
        this.drawPredictionGrid();
        this.drawGroundTruth();
    }

    private updateElements(step: number, loss: Tensor) {
        this.lossElement.textContent = `step: ${step}, loss: ${loss}`;

        const image = GraphUtils.draw_dot(loss);
        this.networkElement.innerHTML = "";
        this.networkElement.appendChild(image);
    }

    private train() {
        // Train
        const epochs = 100;
        let k = 0;
        const interval = setInterval(() => {
            if (k == epochs) clearInterval(interval);

            const X = new Tensor(this.Xb);
            const y = new Tensor(this.yb).reshape([100, 1]);
            // console.log(`X ${X}`);
            // console.log(`y ${y}`);
            // throw Error("HERE")
            const pred = this.model.forward(X);

            const loss = pred.sub(y).pow(2).mean();

            this.model.zero_grad();
            loss.backward();
            // throw Error("HERE")


            const learning_rate = 0.5;
            for (let p of this.model.parameters()) {
                // p.data = p.data.sub(p.grad.mul(learning_rate));
                p.data = MatrixToTensorBuffer(0, TensorBufferToMatrix(p.data).sub(p.grad.mul(0.5)));
                // p.assign(p.sub(p.grad.mul(learning_rate)));

                // p.data = p.sub(p.grad.mul(learning_rate)).data;
            }

            this.drawPredections();
            this.updateElements(k, loss);
            k++;
        }, 10);
    }
}