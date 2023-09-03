import { describe, assert, equal, TensorFactory } from "../TestUtils";
import { Tensor, Random, nn } from "../../src/";

Random.SetRandomSeed(1337);


describe("Linear", () => {
    const linear = new nn.Linear(2, 4);
    linear.weight = new Tensor([[-0.5963, -0.0062],[ 0.1741, -0.1097],[-0.4237, -0.6666],[ 0.1204, 0.2781]]);
    linear.bias = new Tensor([-0.4580, -0.3401, 0.2950, 0.1145]);

    const input = new Tensor([[1,2], [3,4]]);
    const out = linear.forward(input);

    assert(`${linear}` === `Linear(in_features=2, out_features=4)`);
    assert(equal(out, TensorFactory({data: [[-1.0668, -0.3854, -1.4619, 0.7911], [-2.2719, -0.2566, -3.6425, 1.5882]], grad: [[0,0,0,0],[0,0,0,0]]}), 1e-3));
})

describe("Sequential", () => {
    const model = new nn.Sequential(
        new nn.Linear(2, 4),
        new nn.Linear(4, 4),
        new nn.Linear(4, 1),
    );

    assert(model.modules.length === 3);

    assert(`${model.modules[0]}` === `Linear(in_features=2, out_features=4)`);
    assert(`${model.modules[1]}` === `Linear(in_features=4, out_features=4)`);
    assert(`${model.modules[2]}` === `Linear(in_features=4, out_features=1)`);
})

describe("Dropout", () => {
    Random.SetRandomSeed(1337);
    let x = new Tensor([[0.0090, 0.0000, 0.1623, 0.0000, 0.0000, 0.4064, 0.0000, 0.0000, 0.1924,
        0.0000, 0.0000, 0.0542, 0.0000, 0.4154, 0.0000, 0.2993, 0.0000, 0.3429,
        0.3209, 0.0082]]);

    const dropout = new nn.Dropout();
    x = dropout.forward(x);
    assert(equal(x, TensorFactory({data: [[0,0,0.3246,0,0,0.8128,0,0,0,0,0,0.1084,0,0.8308,0,0.5986,0,0,0.6418,0]], grad: [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]})));
})

describe("LayerNorm", () => {
    const x = Tensor.arange(0, 10);
    const layerNorm = new nn.LayerNorm([10]);
    const r = layerNorm.forward(x);

    assert(equal(r, TensorFactory({data: [-1.5666979540876567,-1.2185428531792886,-0.8703877522709204,-0.5222326513625523,-0.17407755045418408,0.17407755045418408,0.5222326513625523,0.8703877522709204,1.2185428531792886,1.5666979540876567], grad: [0,0,0,0,0,0,0,0,0,0]})));
})

describe("Softmax", () => {
    const x = Tensor.arange(0, 10);

    const softmax = new nn.Softmax(0);
    const r = softmax.forward(x);
    assert(equal(r, TensorFactory({data: [0.00007801341612780744,0.00021206245143623275,0.0005764455082375903,0.0015669413501390806,0.004259388198344144,0.011578217539911801,0.031472858344688034,0.08555209892803112,0.23255471590259755,0.6321492583604867], grad: [0,0,0,0,0,0,0,0,0,0]})));

    // const x1 = new Tensor([[5.0, 6.0, 3.0]]);
    // const softmax1 = new nn.Softmax(1);
    // const r1 = softmax1.forward(x1);
    // assert(equal(r1, TensorFactory({data: [[0.25949646034241913,0.7053845126982412,0.03511902695933972]], grad: [[0,0,0]]})));

    // const softmax2 = new nn.Softmax(-1);
    // const r2 = softmax2.forward(x1);
    // assert(equal(r2, TensorFactory({data: [[0.25949646034241913,0.7053845126982412,0.03511902695933972]], grad: [[0,0,0]]})));
})

describe("Embedding", () => {
    Random.SetRandomSeed(1337);

    const embedding = new nn.Embedding(10, 3);
    const input = new Tensor([[1, 2, 4, 5], [4, 3, 2, 9]]);
    embedding.weight = new Tensor([[ 0.1808, -0.0700, -0.3596],
        [-0.9152,  0.6258,  0.0255],
        [ 0.9545,  0.0643,  0.3612],
        [ 1.1679, -1.3499, -0.5102],
        [ 0.2360, -0.2398, -0.4713],
        [ 0.0084, -0.6631, -0.2513],
        [ 1.0101,  0.1215,  0.1584],
        [ 1.1340, -0.2221,  0.6924],
        [-0.5075, -0.9239,  0.5467],
        [-1.4948, -1.2057,  0.5718]]);
    const out = embedding.forward(input);

    assert(equal(out, TensorFactory({data: [[[-0.9152,  0.6258,  0.0255],
        [ 0.9545,  0.0643,  0.3612],
        [ 0.2360, -0.2398, -0.4713],
        [ 0.0084, -0.6631, -0.2513]],

       [[ 0.2360, -0.2398, -0.4713],
        [ 1.1679, -1.3499, -0.5102],
        [ 0.9545,  0.0643,  0.3612],
        [-1.4948, -1.2057,  0.5718]]], grad: [[[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]]})));
})