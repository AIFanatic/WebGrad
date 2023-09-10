import { NNTests } from "./web/NN.test";
import { NetworkTests } from "./web/Network.test";
import { SumTests } from "./web/Sum.test";
import { TensorGradTests } from "./web/Tensor.Grad.test";
import { TensorTests } from "./web/Tensor.test";
import { TestTests } from "./web/Test.test";
import { NetworksTests } from "./web/networks/Networks.test";

export class TestRunner {
    public static describe(name: string, func: Function) {
        throw Error("Not implemented");
    }

    public static UnitTests = [
        TestTests,
        // SumTests
        
        TensorTests,
        TensorGradTests,
        NNTests,
        // NetworkTests,
        // NetworksTests
    ]
}