import { Device } from "../../src";
import { TestRunner } from "../run-web";
import { NNTests } from "../web/NN.test";
import { NetworkTests } from "../web/Network.test";
import { SumTests } from "../web/Sum.test";
import { TensorGradTests } from "../web/Tensor.Grad.test";
import { TensorTests } from "../web/Tensor.test";
import { TestTests } from "../web/Test.test";
import { NetworksTests } from "../web/networks/Networks.test";
import { Table } from "./Table";
import { Test, TestResult } from "./Test";

const UnitTests = [
    // TestTests,
    // SumTests
    
    TensorTests,
    TensorGradTests,
    NNTests,
    NetworkTests,
    // NetworksTests
]
export interface TestEntry {
    categoryElement: HTMLTableRowElement;
    categoryTable: Table;
    testElement: HTMLTableRowElement;
    test: Test;
};

export class WebGradTests {
    private tests: TestEntry[];
    private table: Table;

    constructor(container: HTMLDivElement) {
        this.tests = [];

        this.table = new Table(container);
        this.table.tableHead.addEntry(["Category", "Duration", "Passed", "Failed"]);

        for (let unitTest of UnitTests) {
            const categoryEntry = this.table.tableBody.addEntry([unitTest.category, "-", "-", "-"]);
            categoryEntry.classList.add("category");
            
            const categortTable = this.createCategoryTable(unitTest.category);
            categortTable.container.style.display = "none";
            
            categoryEntry.addEventListener("click", event => {
                categortTable.container.style.display = categortTable.container.style.display === "none" ? "" : "none";
            })
            this.table.tableBody.addEntry(categortTable.container);

            for (let deviceKey of Object.keys(Device)) {
                if (!isNaN(parseInt(deviceKey))) continue;

                
                const device = Device[deviceKey];
                
                TestRunner.describe = (name: string, func: Function) => {
                    const unitTestFunction = () => {
                        TestRunner.describe(name, func);
                    }
                    const test = new Test(unitTest.category, name, device, unitTestFunction);
                    const testElement = categortTable.table.tableBody.addEntry([name, deviceKey, "-", TestResult[TestResult.PENDING]]);
                    this.tests.push({
                        categoryElement: categoryEntry,
                        categoryTable: categortTable.table,
                        testElement: testElement,
                        test: test
                    });
                }

                unitTest.func(device);
            }
        }

        setTimeout(() => {
            this.runTests();
        }, 1000);
    }

    public async runTests() {
        for (let unitTest of this.tests) {
            await new Promise(resolve => setTimeout(resolve, 0)); // Force repaint
            unitTest.categoryTable.tableBody.updateEntry(unitTest.testElement, [
                null,
                null,
                null,
                TestResult[TestResult.RUNNING]
            ]);

            const result = await unitTest.test.run();
            const testResult = result ? TestResult[TestResult.PASSED] : TestResult[TestResult.FAILED];
            unitTest.categoryTable.tableBody.updateEntry(unitTest.testElement, [
                null,
                null,
                `${unitTest.test.duration.toFixed(4)}ms`,
                testResult
            ]);

            if (!result) unitTest.testElement.classList.add("failed");
            else unitTest.testElement.classList.add("passed");

            this.updateCategoryStats(unitTest.categoryElement);
        }
    }

    private updateCategoryStats(category: HTMLTableRowElement) {
        let duration = 0;
        let passed = 0;
        let failed = 0;
        let testsPerCategory = 0;
        for (let unitTest of this.tests) {
            if (unitTest.categoryElement !== category) continue;

            if (unitTest.test.result === TestResult.PASSED) passed++;
            else if (unitTest.test.result === TestResult.FAILED) failed++;
            
            duration += unitTest.test.duration;
            testsPerCategory++;
        }

        if (passed + failed === testsPerCategory) {
            if (failed > 0) category.classList.add("failed");
            else category.classList.add("passed");
        }

        this.table.tableBody.updateEntry(category, [null, `${duration.toFixed(4)}ms`, passed.toString(), failed.toString()]);
    }
    
    private createCategoryTable(category: string): {container: HTMLTableCellElement, table: Table} {
        const testEntrySpanned = document.createElement("td");
        testEntrySpanned.colSpan = 4;
        const testEntryTable = new Table(testEntrySpanned);
        testEntryTable.tableHead.addEntry(["Test", "Device", "Duration", "Result"]);
        return {container: testEntrySpanned, table: testEntryTable};
    }
}