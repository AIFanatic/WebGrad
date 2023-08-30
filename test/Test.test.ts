import { describe, assert, equal, TensorFactory } from "./TestUtils";
import { Tensor } from "../src/Tensor";
import { Matrix, nn } from "../src";

// describe("Sigmoid", () => {
//     const a = new Tensor([1,2,3,4], {requires_grad: true});
//     const b = a.sigmoid();

//     b.backward();

//     assert(equal(a, TensorFactory({data: [1,2,3,4], grad: [0.19661193324148188,0.1049935854035065,0.045176659730912144,0.017662706213291114]})));
//     assert(equal(b, TensorFactory({data: [0.7310585786300049,0.8807970779778823,0.9525741268224334,0.9820137900379085], grad: [1,1,1,1]})));
// })

// describe("Tanh", () => {
//     const a = new Tensor([1,2,3,4], {requires_grad: true});
//     const c = a.tanh();

//     c.backward();

//     console.log(`a ${a}`);

//     assert(equal(a, TensorFactory({data: [1,2,3,4], grad: [0.41997434161402614,0.07065082485316443,0.009866037165440211,0.0013409506830258655]})));
//     assert(equal(c, TensorFactory({data: [0.7615941559557649,0.9640275800758169,0.9950547536867305,0.999329299739067], grad: [1,1,1,1]})));
// })



describe("Softmax", () => {
    const x = new Tensor(Matrix.arange(0, 10));

    const softmax = new nn.Softmax(0);
    const r = softmax.forward(x);
    assert(equal(r, TensorFactory({data: [0.00007801341612780744,0.00021206245143623275,0.0005764455082375903,0.0015669413501390806,0.004259388198344144,0.011578217539911801,0.031472858344688034,0.08555209892803112,0.23255471590259755,0.6321492583604867], grad: [0,0,0,0,0,0,0,0,0,0]})));

    const x1 = new Tensor([[5.0, 6.0, 3.0]]);
    const softmax1 = new nn.Softmax(1);
    const r1 = softmax1.forward(x1);
    assert(equal(r1, TensorFactory({data: [[0.25949646034241913,0.7053845126982412,0.03511902695933972]], grad: [[0,0,0]]})));

    const softmax2 = new nn.Softmax(-1);
    const r2 = softmax2.forward(x1);
    assert(equal(r2, TensorFactory({data: [[0.25949646034241913,0.7053845126982412,0.03511902695933972]], grad: [[0,0,0]]})));
})