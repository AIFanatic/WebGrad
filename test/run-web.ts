import { TensorTests } from "./web/Tensor.test";
import { TestTests } from "./web/Test.test";

export class TestRunner {
    public static describe(name: string, func: Function) {
        throw Error("Not implemented");
    }

    public static UnitTests = [
        TestTests,
        TensorTests
    ]
}