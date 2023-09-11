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

    TestRunner.describe("Split2", () => {
        Random.SetRandomSeed(1337);

        const n_embd = 48;
        const shape = [1,13,144];
        // const a = Tensor.zeros(shape, {device: device});
        const a = Tensor.rand(shape, {device: device});

        const [q,k,v] = a.split(n_embd, 2);
        console.log(q.data.getData().flat(Infinity));
        // console.log(`q ${q} ${q.shape}`);
        // console.log(`k ${k} ${k.shape}`);
        // console.log(`v ${v} ${v.shape}`);
    })
}

export const TestTests = {category: "Test", func: TestTest};