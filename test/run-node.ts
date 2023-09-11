import fs from "fs";
import path from "path";
import { execSync } from "child_process";

function getAllFiles(dirPath: string, arrayOfFiles: string[] = []): string[] {
    const files = fs.readdirSync(dirPath)

    for (let file of files) {
        if (fs.statSync(dirPath + "/" + file).isDirectory()) {
            arrayOfFiles = getAllFiles(dirPath + "/" + file, arrayOfFiles);
        } else {
            arrayOfFiles.push(path.join(dirPath, "/", file));
        }
    }

    return arrayOfFiles;
}

function getFilesByPattern(dirPath: string, pattern: string): string[] {
    const files = getAllFiles(dirPath);
    let matchingFiles: string[] = [];
    for (let file of files) {
        if (file.includes(pattern)) matchingFiles.push(file);
    }
    return matchingFiles;
}

const files = getFilesByPattern("./test/node", "test.ts");

for(let file of files) {
    console.log(`--------`);
    console.log(`Running test: ${file}`);
    execSync(`node -r esbuild-runner/register ${file}`, { stdio: 'inherit' });
}





// import { Device } from "../src";
// import { TestRunner } from "./run-web";
// import { TensorTests } from "./web/Tensor.test";
// import { TensorGradTests } from "./web/Tensor.Grad.test";
// import { NNTests } from "./web/NN.test";
// import { NetworkTests } from "./web/Network.test";
// import { NetworksTests } from "./web/networks/Networks.test";

// const UnitTests = [
//     // TestTests,
//     // SumTests
    
//     // TensorTests,
//     // TensorGradTests,
//     // NNTests,
//     // NetworkTests,
//     NetworksTests
// ]

// TestRunner.describe = async (name: string, func: Function) => {
//     const colors = {"red": "\x1b[31m", "green": "\x1b[32m", "yellow": "\x1b[33m"};

//     let passed = false;
//     try {
//         await func();

//         console.log(`${colors["yellow"]}[${name}] ${colors["green"]} Passed`);

//     } catch (error) {
//         console.error("Error", error);
//         passed = false;
//         console.log(`${colors["yellow"]}[${name}] ${colors["red"]} Failed`);
//         throw Error("Test failed")
//     }
// }

// for (let unitTest of UnitTests) {
//     new Promise(resolve => setTimeout(resolve, 0)).then(value => {
//         console.log(`--------`);
//         console.log(`Running test: ${unitTest.category}`);
//         unitTest.func(Device.CPU);
//     })
// }