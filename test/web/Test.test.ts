import { assert, equal, TensorFactory } from "../TestUtils";
import { Tensor } from "../../src/Tensor";
import { TestRunner } from "../run-web";
import { Device } from "../../src/backend/Backend";
import { NetworkMoonsData } from "./networks/MoonsData/MoonsData.test";

function TestTest(device: Device) {
    // TestRunner.describe("Add", () => {
    //     const a = new Tensor([[1, 1, 1], [2, 2, 2]], {device: device});
    //     const b = new Tensor([[3, 3, 3], [4, 4, 4]], {device: device});
    
    //     const c = a.add(b);
    //     assert(equal(c, new Tensor([[4, 4, 4], [6, 6, 6]])));
    //     assert(equal(c.shape, [2, 3]));
    // })

    TestRunner.describe("Matmul", () => {
        const a = new Tensor([[1,2], [3,4]], {device: device, requires_grad: true});
        const b = new Tensor([[5,6], [7,8]], {device: device, requires_grad: true});
        const c = a.matmul(b);
    
        c.backward();
    
        console.log(`a ${a}`);
        assert(equal(a, TensorFactory({data: [[1,2],[3,4]], grad: [[11,15],[11,15]]})));
        assert(equal(b, TensorFactory({data: [[5,6],[7,8]], grad: [[4,4],[6,6]]})));
        assert(equal(c, TensorFactory({data: [[19,22],[43,50]], grad: [[1,1],[1,1]]})));
    })

    // TestRunner.describe("MoonsData", () => {
    //     NetworkMoonsData(device);
    // })
}

export const TestTests = {category: "Test", func: TestTest};