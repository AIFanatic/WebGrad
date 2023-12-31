import { describe, assert, equal, TensorFactory } from "../../../TestUtils";
import { Device, Module, nn, Random, Tensor } from "../../../../src";
import { TestRunner } from "../../../run-web";

export function NetworkMoonsData(device: Device) {
    
    TestRunner.describe("MoonsData test", () => {
        Random.SetRandomSeed(1337);

        function generateData(n: number): number[][] {
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
    
        // Create data - This needs to be first because of Random seed
        const [Xb, yb] = generateData(100);
        const X = new Tensor(Xb, {device: device, requires_grad: true});
        const y = new Tensor(yb, {device: device, requires_grad: true}).reshape([100, 1]);
        
        // Create model
        const model = new SimpleModel(2, 1).to(device);

        // Train
        const learning_rate = new Tensor(0.5, {device: device});
        let last_loss: Tensor = new Tensor(0, {device: device});
        const epochs = 100;
        for (let k = 0; k < epochs; k++) {
            const pred = model.forward(X);
    
            const loss = pred.sub(y).pow(2).mean();
            
            model.zero_grad();
            loss.backward();
    
            for (let p of model.parameters()) {
                p.data = p.sub(new Tensor(p.grad).mul(learning_rate)).data;
            }
    
            last_loss = loss;
        }
    
        assert(equal(last_loss, TensorFactory({data: [0.003], grad: [1]}), 1e-4));
    })
}