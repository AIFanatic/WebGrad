import { assert, equal, TensorFactory } from "../TestUtils";
import { Tensor } from "../../src/Tensor";
import { TestRunner } from "../run-web";
import { Backend, Device } from "../../src/backend/Backend";
import { NetworkMoonsData } from "./networks/MoonsData/MoonsData.test";
import { TensorBuffer } from "../../src/backend/TensorBuffer";
import { Random } from "../../src";

function TestTest(device: Device) {
    // TestRunner.describe("Split", () => {
    //     // const a = Tensor.arange(0, 10).reshape([5, 2]);
    //     const a = new Tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]);

    //     const b = a.split(2);
    //     console.log(`b`, b);
    //     assert(b.length == 3);
    //     assert(equal(b[0], new Tensor([[0, 1], [2, 3]])));
    //     assert(equal(b[1], new Tensor([[4, 5], [6, 7]])));
    //     assert(equal(b[2], new Tensor([[8, 9]])));

    //     const c = a.split([1,4]);
    //     console.log(`c`, c);
    //     assert(c.length == 2);
    //     assert(equal(c[0], new Tensor([[0, 1]])));
    //     assert(equal(c[1], new Tensor([[2, 3], [4, 5], [6, 7], [8, 9]])));
    // })

    // TestRunner.describe("Split2", () => {
    //     Random.SetRandomSeed(1337);

    //     const n_embd = 48;
    //     const shape = [1,13,144];
    //     // const a = Tensor.zeros(shape, {device: device});
    //     const a = Tensor.rand(shape, {device: device});

    //     const [q,k,v] = a.split(n_embd, 2);
    //     console.log(q.data.getData().flat(Infinity));
    //     // console.log(`q ${q} ${q.shape}`);
    //     // console.log(`k ${k} ${k.shape}`);
    //     // console.log(`v ${v} ${v.shape}`);
    // })

    // // GPT2
    // TestRunner.describe("Slice", () => {
    //     Random.SetRandomSeed(1337);

    //     const shape = [1,1,65];
    //     const strides = [845,65,1];
    //     const data = [[[8.644,8.5229,-1.9436,-15.5243,-12.0605,4.1463,0.9285,3.7983,0.8079,-3.7813,-2.1262,-0.1962,0.0467,-4.5015,-5.1211,-4.4776,-5.6697,-6.3393,-6.6477,-3.2807,-5.1393,-2.4072,-6.9908,-6.8093,-6.7112,-6.8534,-2.8274,-2.2841,-4.9994,-8.6178,-5.0549,-6.1175,-3.6124,-8.5516,-7.2697,-3.4036,-13.2461,-7.0604,-4.2361,-0.8241,0.4147,-0.138,-0.6626,-0.4893,0.2797,-1.1267,-1.4947,0.0741,-4.6398,-3.713,-0.4828,-2.0015,0.0719,-1.8633,-4.9201,-4.0752,-0.3484,0.0239,0.9059,-2.437,-4.1292,0.5238,-5.7331,-3.5339,-7.7956]]];
    //     const a = new Tensor(data, {device: device});
    //     console.log(`a ${a} ${a.shape} ${a.strides}`);

    //     const tb = Backend.CreateFromFloat32Array(device, new Float32Array(data.flat(Infinity)), shape, strides, 0);
    //     const b = new Tensor(tb, {device: device});
    //     console.log(`b ${b.contiguous()} ${b.shape} ${b.strides}`);
    // })

    TestRunner.describe("Slice", () => {
        Random.SetRandomSeed(1337);

        console.log(Device[device]);
        const data = [
            [
                [-0.4617, 2.0794, -0.3864, -11.851, -9.0403, -0.9475, 0.8994, -2.2933, -0.8202, -5.4271],
                [-4.8378, -4.6949, -5.4385, -12.7535, -8.4812, -1.9216, -5.5221, -3.7003, -5.5581, -10.1265],
            ]
        ];
        console.log()
        const a = new Tensor(data, {device: device});
        console.log(`a ${a} ${a.shape} ${a.strides}`);

        const b = a.slice([null, [a.shape[1]-1, a.shape[1]], null]);
        console.log(`b ${b} ${b.shape} ${b.strides} ${b.offset}`);
        
        const c = b.contiguous();
        console.log(`c ${c} ${c.shape} ${c.strides} ${c.offset}`);
    })
}

export const TestTests = {category: "Test", func: TestTest};