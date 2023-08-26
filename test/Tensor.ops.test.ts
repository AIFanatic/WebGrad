import { describe, assert, equal, TensorFactory } from "./TestUtils";
import { Tensor } from "../src/Tensor";

describe("Tensor scalar", () => {
    const a = new Tensor(-2);
    const b = new Tensor(3);
    const d = a.mul(b);
    const e = a.add(b);
    const f = d.mul(e);

    f.sum().backward();

    console.log(`a ${a}`);
    console.log(`b ${b}`);
    console.log(`d ${d}`);
    console.log(`e ${e}`);
    console.log(`f ${f}`);

    // assert(`${a}` === `Tensor(data=Matrix([-2], shape=[1]), grad=Matrix([-3], shape=[1]))`);
    // assert(`${b}` === `Tensor(data=Matrix([3], shape=[1]), grad=Matrix([-8], shape=[1]))`);
    // assert(`${d}` === `Tensor(data=Matrix([-6], shape=[1]), grad=Matrix([1], shape=[1]))`);
    // assert(`${e}` === `Tensor(data=Matrix([1], shape=[1]), grad=Matrix([-6], shape=[1]))`);
    // assert(`${f}` === `Tensor(data=Matrix([-6], shape=[1]), grad=Matrix([1], shape=[1]))`);
})

// describe("Simple array", () => {
//     const a = new Tensor([1,2,3]);
//     const b = new Tensor([4,5,6]);
    
//     const c = a.mul(b).mul(10).sum();

//     c.backward();

//     assert(`${a}` === "Tensor(data=Matrix([1,2,3], shape=[3]), grad=Matrix([40,50,60], shape=[3]))");
//     assert(`${b}` === "Tensor(data=Matrix([4,5,6], shape=[3]), grad=Matrix([10,20,30], shape=[3]))");
//     assert(`${c}` === "Tensor(data=Matrix([320], shape=[1]), grad=Matrix([1], shape=[1]))");
// })


// describe("Simple Tensor", () => {

//     const a1 = new Tensor([1,3,1]);
//     const b1 = new Tensor([7,3,5]);

//     const a2 = new Tensor([4,3,1]);
//     const a3 = new Tensor([3,3,1]);
//     const a4 = new Tensor([7,1,6]);
//     const b2 = new Tensor([1,21,12]);

//     const c = a1.mul(b1).add(a3);
//     const d = a2.mul(b2).add(a4);

//     const out = c.mul(d);

//     out.backward();

//     assert(`${out}` === `Tensor(data=Matrix([110,768,108], shape=[3]), grad=Matrix([1,1,1], shape=[3]))`);
//     assert(`${a1}` === `Tensor(data=Matrix([1,3,1], shape=[3]), grad=Matrix([77,192,90], shape=[3]))`);
// })

// describe("More ops", () => {
//     // inputs x1,x2
//     const x1 = new Tensor([2]);
//     const x2 = new Tensor([0]);
//     // weights w1,w2
//     const w1 = new Tensor([-3]);
//     const w2 = new Tensor([1]);
//     // bias of the neuron
//     const b = new Tensor([6.8813735870195432]);
//     // x1*w1 + x2*w2 + b
//     const x1w1 = x1.mul(w1);
//     const x2w2 = x2.mul(w2);
//     const x1w1x2w2 = x1w1.add(x2w2);
//     const n = x1w1x2w2.add(b);
//     // ----
//     const e = new Tensor([2]).mul(n).exp();
    
//     // o = (e - 1) / (e + 1)
//     const p = e.sub(1);
//     const q = e.add(1);

//     const o = p.div(q);

//     o.backward();

//     assert(equal(x1w1, TensorFactory({data: [-6], grad: [0.5]})));
//     assert(equal(x1w1, TensorFactory({data: [-6], grad: [0.5]})));
//     assert(equal(x2w2, TensorFactory({data: [0], grad: [0.5]})));
//     assert(equal(x1w1x2w2, TensorFactory({data: [-6], grad: [0.5]})));
//     assert(equal(n, TensorFactory({data: [0.8813735870195432], grad: [0.5]})));
//     assert(equal(e, TensorFactory({data: [5.828427124746192], grad: [0.04289321881345247]})));
//     assert(equal(p, TensorFactory({data: [4.828427124746192], grad: [0.1464466094067262]})));
//     assert(equal(q, TensorFactory({data: [6.828427124746192], grad: [-0.10355339059327374]})));
//     assert(equal(o, TensorFactory({data: [0.7071067811865477], grad: [1]})));
// })

// describe("Matmul backward", () => {
//     const a = new Tensor([[-2.0260, -2.0655, -1.2054], [-0.9122, -1.2502,  0.8032]]);
//     const b = new Tensor([[-0.2071,  0.0544], [ 0.1378, -0.3889], [ 0.5133,  0.3319]]);

//     const out = a.matmul(b);
//     const scalar_out = out.sum();

//     scalar_out.backward();

//     assert(equal(a, TensorFactory({data: [[-2.026,-2.0655,-1.2054],[-0.9122,-1.2502,0.8032]], grad: [[-0.1527,-0.2511,0.8452],[-0.1527,-0.2511,0.8452]]})));
//     assert(equal(b, TensorFactory({data: [[-0.2071,0.0544],[0.1378,-0.3889],[0.5133,0.3319]], grad: [[-2.9381999999999997,-2.9381999999999997],[-3.3157,-3.3157],[-0.4022,-0.4022]]})));
// })