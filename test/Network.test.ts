import { describe, assert, equal, TensorFactory } from "./TestUtils";

import { MatrixToTensorBuffer, Tensor, TensorBufferToMatrix } from "../src/Tensor";

import { Module } from "../src/Module";
import { Matrix, nn } from "../src/";
import { Random } from "../src/Random";


describe("1 dense layer", () => {
    Random.SetRandomSeed(1337);

    class SingleLayerModel extends Module {
        public dense1: nn.Linear;

        constructor(input_sample: number, output_sample: number) {
            super();
            this.dense1 = new nn.Linear(input_sample, output_sample);
        }

        public forward(x: Tensor): Tensor {
            x = this.dense1.forward(x);
            return x;
        }
    }

    const Xb = new Tensor([
        [0.2, 0.3],
        [-0.4, 0.8],
        [-0.3, 0.9],
        [0.5, 0.3]
    ], {requires_grad: true});

    const yb = new Tensor([[1.0], [0.2], [0.3], [0.7]], {requires_grad: true});

    const model = new SingleLayerModel(2, 1);
    model.dense1.weight = new Tensor([[-0.5963, -0.0062]]); // Match torch seed 1337
    model.dense1.bias = new Tensor([0.1741]); // Match torch seed 1337
    
    const output = model.forward(Xb);

    const loss = output.sub(yb).pow(2).mean();

    model.zero_grad();

    loss.backward();

    const learning_rate = 0.01;
    for (let p of model.parameters()) {
        p.data = MatrixToTensorBuffer(0, TensorBufferToMatrix(p.data).sub(p.grad.mul(learning_rate)));
    }

    assert(equal(loss, TensorFactory({data: [0.40608614805], grad: [1]})));
    assert(equal(model.dense1.weight, TensorFactory({data: [[-0.59280177,-0.00458459]], grad: [[-0.349823,-0.16154099999999993]]})));
    assert(equal(model.dense1.bias, TensorFactory({data: [0.1816893], grad: [-0.75893]})));
});

describe("3 dense layers", () => {
    Random.SetRandomSeed(1337);

    class SimpleModel extends Module {
        public dense1: nn.Linear;
        public dense2: nn.Linear;
        public dense3: nn.Linear;

        constructor(input_sample: number, output_sample: number) {
            super();
            
            this.dense1 = new nn.Linear(input_sample, 4);
            this.dense2 = new nn.Linear(4, 4);
            this.dense3 = new nn.Linear(4, output_sample);
        }

        public forward(x: Tensor): Tensor {            
            x = this.dense1.forward(x);
            x = this.dense2.forward(x);
            x = this.dense3.forward(x);

            return x;
        }
    }

    function get_loss(model: SimpleModel, x_tensor: Tensor, y_tensor: Tensor): Tensor {
        // Data loss
        const pred = model.forward(x_tensor).tanh()
        const data_loss = pred.sub(y_tensor).pow(2).sum();
        
        // Reg loss
        let reg_loss = new Tensor([0], {requires_grad: true});
        for (let p of model.parameters()) {
            const w_sum = p.mul(p).sum();
            const p_shape_prod = Matrix.prod(new Matrix(p.data.shape));
            const div = w_sum.div(new Tensor(p_shape_prod));
            reg_loss = reg_loss.add(div);
        }

        
        const alpha = new Tensor(1e-4);
        const total_loss = data_loss.mean().add(reg_loss.mul(alpha));

        return total_loss;
    }
    

    const X = new Tensor([
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ], {requires_grad: true});

    const Y = new Tensor([1.0, 0.2, 0.3, 0.7], {requires_grad: true});

    const model = new SimpleModel(3, 2);

    // Matches torch weights and biases with seed 1337
    model.dense1.weight = new Tensor([[-0.4869, -0.0051,  0.1421], [-0.0896, -0.3460, -0.5443], [ 0.0983,  0.2271, -0.3740], [-0.2777,  0.2408,  0.0935]], {requires_grad: true});
    model.dense1.bias = new Tensor([-0.5111,  0.3082,  0.4363, -0.2963], {requires_grad: true});

    model.dense2.weight = new Tensor([[ 0.1005,  0.2079,  0.0102, -0.0935], [ 0.3864, -0.1422,  0.3963,  0.4639], [-0.4852,  0.2358,  0.2884,  0.4469], [-0.0344,  0.3378, -0.3731, -0.2868]], {requires_grad: true});
    model.dense2.bias = new Tensor([-0.2056, -0.1323, -0.0170,  0.1752], {requires_grad: true});

    model.dense3.weight = new Tensor([[ 0.0226,  0.0536,  0.0701, -0.2519], [ 0.1040,  0.2077, -0.4210,  0.2629]], {requires_grad: true});
    model.dense3.bias = new Tensor([ 0.2708, -0.4257], {requires_grad: true});

    let last_loss;
    const epochs = 100;
    for (let k = 0; k < epochs; k++) {
        const Xb = new Tensor(X, {requires_grad: true});
        const Yb = new Tensor(Y.reshape([Y.shape[0], 1]), {requires_grad: true});
        
        const total_loss = get_loss(model, Xb, Yb);
        model.zero_grad();
        total_loss.backward();

        for (let p of model.parameters()) {
            const learning_rate = Matrix.full(p.grad.shape, 0.1);
            p.data = MatrixToTensorBuffer(0, TensorBufferToMatrix(p.data).sub(learning_rate.mul(p.grad)));
        }

        const total_loss2 = get_loss(model, new Tensor(X), new Tensor(Y.reshape([Y.shape[0], 1])));

        last_loss = total_loss2;
    }

    assert(equal(last_loss, TensorFactory({data: [0.017161300405859947], grad: [0]})));
})