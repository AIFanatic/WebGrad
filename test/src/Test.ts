import { Device } from "../../src";
import { TestRunner } from "../run-web";

export enum TestResult {
    PENDING,
    RUNNING,
    PASSED,
    FAILED
};

export class Test {
    public readonly category: string;
    public readonly name: string;
    public readonly device: Device;
    public readonly func: Function;
    public duration: number;
    public result: TestResult;
    public resultDescription: string;

    constructor(category: string, name: string, device: Device, func: Function) {
        this.category = category;
        this.name = name;
        this.device = device;
        this.func = func;
        this.duration = 0;
        this.result = TestResult.PENDING;
    }

    public async run(): Promise<boolean> {
        return new Promise<boolean>(resolve => {
            TestRunner.describe = async (name, func) => {
                try {
                    const st = performance.now();
                    func();
                    this.duration = performance.now() - st;
                    this.result = TestResult.PASSED;
                    resolve(true);
                    return true;
    
                } catch (error) {
                    console.log(error)
                    this.resultDescription = error;
                    this.result = TestResult.FAILED;
                    resolve(false);
                }
            }
    
            this.func();
        })
    }
}