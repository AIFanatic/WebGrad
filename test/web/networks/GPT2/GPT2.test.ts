import { GPT2Demo } from ".";
import { describe, assert, equal, TensorFactory } from "../../../TestUtils";

describe("GPT2 test", async () => {
    const gpt2Demo = new GPT2Demo();

    const generated = await gpt2Demo.run();

    assert(equal(generated, TensorFactory({data: [27,1,19,53,42,6,1,27,1,19,53,42,2,0,0,31,47,58,1,40,43,1,61], grad: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]})));
})