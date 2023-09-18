import { GPT2Demo } from ".";
import { Device, Random, Tensor } from "../../../../src";
import { assert, equal, TensorFactory } from "../../../TestUtils";
import { TestRunner } from "../../../run-web";

export function NetworkGPT2(device: Device) {

    TestRunner.describe("GPT2 test", async () => {
        Random.SetRandomSeed(1337);
        
        const gpt2Demo = new GPT2Demo(device);
        const generated = await gpt2Demo.run();

        // console.log(`generated ${generated}`);
    
        assert(equal(generated, TensorFactory({data: [27,1,19,53,42,6,1,27,1,19,53,42,2,0,0,31,47,58,1,40,43,1,61], grad: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]})));
    });
};