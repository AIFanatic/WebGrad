import { assert, equal, TensorFactory } from "../TestUtils";
import { Tensor } from "../../src/Tensor";
import { nn } from "../../src";
import { TestRunner } from "../run-web";
import { Device } from "../../src/backend/Backend";

function TestTest(device: Device) {
    TestRunner.describe("Softmax", () => {
        const x = new Tensor([[-0.947],[0.2077],[0.0474],[-0.8259]], {device: device});
        const a = new Tensor([[1],[1],[1],[1]], {device: device});

        const b = x.pow(a);

        console.log(`b ${b}`);
    })
}

export const TestTests = {category: "Test", func: TestTest};