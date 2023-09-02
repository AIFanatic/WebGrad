import { assert, equal, TensorFactory } from "../TestUtils";
import { Tensor } from "../../src/Tensor";
import { nn } from "../../src";
import { TestRunner } from "../run-web";
import { Device } from "../../src/backend/Backend";

function TestTest(device: Device) {
    TestRunner.describe("Softmax", () => {
        const x = Tensor.arange(0, 10);
    
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
}

export const TestTests = {category: "Test", func: TestTest};