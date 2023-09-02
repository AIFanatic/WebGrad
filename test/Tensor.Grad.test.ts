import { describe, assert, equal, TensorFactory } from "./TestUtils";
import { Tensor } from "../src/Tensor";

describe("Add", () => {
    const a = new Tensor([1,2,3], {requires_grad: true});
    const b = new Tensor([4,5,6], {requires_grad: true});
    const c = a.add(b);

    c.backward();

    // console.log(`a ${a.requires_grad}`);
    // console.log(`b ${b.requires_grad}`);
    // console.log(`c ${c.requires_grad}`);

    assert(equal(a, TensorFactory({data: [1,2,3], grad: [1,1,1]})));
    assert(equal(b, TensorFactory({data: [4,5,6], grad: [1,1,1]})));
    assert(equal(c, TensorFactory({data: [5,7,9], grad: [1,1,1]})));
})

describe("Sub", () => {
    const a = new Tensor([1,2,3], {requires_grad: true});
    const b = new Tensor([4,5,6], {requires_grad: true});
    const c = a.sub(b);

    c.backward();

    assert(equal(a, TensorFactory({data: [1,2,3], grad: [1,1,1]})));
    assert(equal(b, TensorFactory({data: [4,5,6], grad: [-1,-1,-1]})));
    assert(equal(c, TensorFactory({data: [-3,-3,-3], grad: [1,1,1]})));
})

describe("Mul", () => {
    const a = new Tensor([1,2,3], {requires_grad: true});
    const b = new Tensor([4,5,6], {requires_grad: true});
    const c = a.mul(b);

    c.backward();

    assert(equal(a, TensorFactory({data: [1,2,3], grad: [4,5,6]})));
    assert(equal(b, TensorFactory({data: [4,5,6], grad: [1,2,3]})));
    assert(equal(c, TensorFactory({data: [4,10,18], grad: [1,1,1]})));
})

describe("Div", () => {
    const a = new Tensor([1,2,3], {requires_grad: true});
    const b = new Tensor([4,5,6], {requires_grad: true});
    const c = a.div(b);

    c.backward();

    assert(equal(a, TensorFactory({data: [1,2,3], grad: [0.25,0.2,0.16666666666666666]})));
    assert(equal(b, TensorFactory({data: [4,5,6], grad: [-0.0625,-0.08,-0.08333333333333333]})));
    assert(equal(c, TensorFactory({data: [0.25,0.4,0.5], grad: [1,1,1]})));
})

describe("Pow", () => {
    const a = new Tensor([4,5,6], {requires_grad: true});
    const b = a.pow(2);

    b.backward();

    assert(equal(a, TensorFactory({data: [4,5,6], grad: [8,10,12]})));
    assert(equal(b, TensorFactory({data: [16,25,36], grad: [1,1,1]})));
})

describe("Matmul", () => {
    const a = new Tensor([[1,2], [3,4]], {requires_grad: true});
    const b = new Tensor([[5,6], [7,8]], {requires_grad: true});
    const c = a.matmul(b);

    c.backward();

    assert(equal(a, TensorFactory({data: [[1,2],[3,4]], grad: [[11,15],[11,15]]})));
    assert(equal(b, TensorFactory({data: [[5,6],[7,8]], grad: [[4,4],[6,6]]})));
    assert(equal(c, TensorFactory({data: [[19,22],[43,50]], grad: [[1,1],[1,1]]})));
})

describe("Sum", () => {
    const a = new Tensor([1,2,3], {requires_grad: true});
    const c = a.sum();

    c.backward();

    assert(equal(a, TensorFactory({data: [1,2,3], grad: [1,1,1]})));
    assert(equal(c, TensorFactory({data: [6], grad: [1]})));
})

describe("Mean", () => {
    const a = new Tensor([1,2,3], {requires_grad: true});
    const c = a.mean();

    c.backward();

    assert(equal(a, TensorFactory({data: [1,2,3], grad: [0.3333333333333333,0.3333333333333333,0.3333333333333333]})));
    assert(equal(c, TensorFactory({data: [2], grad: [1]})));
})

describe("Rehape", () => {
    const a = new Tensor([1,2,3,4], {requires_grad: true});
    const c = a.reshape([2,2]);

    c.backward();

    assert(equal(a, TensorFactory({data: [1,2,3,4], grad: [1,1,1,1]})));
    assert(equal(c, TensorFactory({data: [[1,2],[3,4]], grad: [[1,1],[1,1]]})));
})

describe("Exp", () => {
    const a = new Tensor([1,2,3,4], {requires_grad: true});
    const c = a.exp();

    c.backward();

    assert(equal(a, TensorFactory({data: [1,2,3,4], grad: [2.718281828459045,7.38905609893065,20.085536923187668,54.598150033144236]})));
    assert(equal(c, TensorFactory({data: [2.718281828459045,7.38905609893065,20.085536923187668,54.598150033144236], grad: [1,1,1,1]})));
})

describe("ReLu", () => {
    const a = new Tensor([-1,2,3,4], {requires_grad: true});
    const c = a.relu();

    c.backward();

    assert(equal(a, TensorFactory({data: [-1,2,3,4], grad: [0,1,1,1]})));
    assert(equal(c, TensorFactory({data: [0,2,3,4], grad: [1,1,1,1]})));
})

describe("Reciprocal", () => {
    const a = new Tensor([1,2,3,4], {requires_grad: true});
    const b = a.reciprocal()
    
    b.backward()

    assert(equal(a, TensorFactory({data: [1,2,3,4], grad: [-1,-0.25,-0.1111111111111111,-0.0625]})));
    assert(equal(b, TensorFactory({data: [1,0.5,0.3333333333333333,0.25], grad: [1,1,1,1]})));
})

describe("Sigmoid", () => {
    const a = new Tensor([1,2,3,4], {requires_grad: true});
    const b = a.sigmoid();

    b.backward();

    assert(equal(a, TensorFactory({data: [1,2,3,4], grad: [0.19661193324148188,0.1049935854035065,0.045176659730912144,0.017662706213291114]})));
    assert(equal(b, TensorFactory({data: [0.7310585786300049,0.8807970779778823,0.9525741268224334,0.9820137900379085], grad: [1,1,1,1]})));
})

describe("Tanh", () => {
    const a = new Tensor([1,2,3,4], {requires_grad: true});
    const c = a.tanh();

    c.backward();

    assert(equal(a, TensorFactory({data: [1,2,3,4], grad: [0.41997434161402614,0.07065082485316443,0.009866037165440211,0.0013409506830258655]})));
    assert(equal(c, TensorFactory({data: [0.7615941559557649,0.9640275800758169,0.9950547536867305,0.999329299739067], grad: [1,1,1,1]})));
})

describe("Permute", () => {
    const a = new Tensor([[1, 2], [3, 4], [5, 6]], {requires_grad: true});
    const c = a.permute([1,0]);

    c.backward();

    assert(equal(a, TensorFactory({data: [[1, 2], [3, 4], [5, 6]], grad: [[1, 1], [1, 1], [1, 1]]})));
    assert(equal(c, TensorFactory({data: [[1, 3, 5], [2, 4, 6]], grad: [[1, 1, 1], [1, 1, 1]]})));
})

describe("Transpose", () => {
    const a = new Tensor([[1, 2], [3, 4], [5, 6]], {requires_grad: true});
    const c = a.transpose(1,0);

    c.backward();

    assert(equal(a, TensorFactory({data: [[1, 2], [3, 4], [5, 6]], grad: [[1, 1], [1, 1], [1, 1]]})));
    assert(equal(c, TensorFactory({data: [[1, 3, 5], [2, 4, 6]], grad: [[1, 1, 1], [1, 1, 1]]})));
})

describe("Abs", () => {
    const a = new Tensor([[-3, 3, 3], [4, -4, 4]]);
    const b = a.abs();

    b.backward();

    assert(equal(a, TensorFactory({data: [[-3, 3, 3], [4, -4, 4]], grad: [[-1, 1, 1], [1, -1, 1]]})));
    assert(equal(b, TensorFactory({data: [[3, 3, 3], [4, 4, 4]], grad: [[1, 1, 1], [1, 1, 1]]})));
})

describe("Log", () => {
    const a = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const b = a.log();
    b.backward()
    assert(equal(a, TensorFactory({data: [[1, 2, 3], [4, 5, 6]], grad: [[1.0000, 0.5000, 0.3333], [0.2500, 0.2000, 0.1667]]}), 1e-4));
    assert(equal(b, TensorFactory({data: [[0.0000, 0.6931, 1.0986], [1.3863, 1.6094, 1.7918]], grad: [[1, 1, 1], [1, 1, 1]]}), 1e-4));
})

describe("Simple model", () => {
    const weight = new Tensor([[-0.4869, -0.0896], [-0.0051, -0.3460], [ 0.1421, -0.5443]], {requires_grad: true});

    let x = new Tensor([[2.0, 3.0, -1.0]], {requires_grad: true});
    x = x.matmul(weight);

    const pred = x;
    const loss = pred.sum();

    loss.backward();
    assert(equal(weight, TensorFactory({data: [[-0.4869, -0.0896], [-0.0051, -0.3460], [ 0.1421, -0.5443]], grad: [[2, 2], [3, 3], [-1, -1]]})));
});

describe("Maximum", () => {
    const a = new Tensor([2, 3, 4]);
    const b = new Tensor([1, 5, 2]);
    const c = a.maximum(b);

    c.backward();

    assert(equal(a, TensorFactory({data: [2, 3, 4], grad: [1, 0, 1]})));
    assert(equal(b, TensorFactory({data: [1, 5, 2], grad: [0, 1, 0]})));
    assert(equal(c, TensorFactory({data: [2, 5, 4], grad: [1, 1, 1]})));
})

describe("Equal", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor([0, 2, 2]);
    const c = a.eq(b);

    c.backward();

    console.log(`a ${a}`);
    console.log(`b ${b}`);
    console.log(`c ${c}`);
    // assert(equal(c, new Tensor([0, 1, 0])));
})

// describe("Simple model more ops", () => {
//     const weight = new Tensor([[-0.4869, -0.0051,  0.1421],[-0.0896, -0.3460, -0.5443],[ 0.0983,  0.2271, -0.3740],[-0.2777,  0.2408,  0.0935]], {device: device,);
//     const bias = new Tensor([-0.5111,  0.3082,  0.4363, -0.2963], {device: device,);

//     const o = weight.permute().add(bias).sum();

//     o.backward();


//     assert(equal(weight, TensorFactory({
//         data: [[-0.4869, -0.0051,  0.1421], [-0.0896, -0.3460, -0.5443], [ 0.0983,  0.2271, -0.3740], [-0.2777,  0.2408,  0.0935]], 
//         grad: [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]
//     })));

//     assert(equal(bias, TensorFactory({
//         data: [-0.5111,  0.3082,  0.4363, -0.2963],
//         grad: [3., 3., 3., 3.]
//     })));
// });