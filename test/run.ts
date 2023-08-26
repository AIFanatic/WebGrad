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

const files = getFilesByPattern("./test", "test.ts");

for(let file of files) {
    console.log(`--------`);
    console.log(`Running test: ${file}`);
    execSync(`node -r esbuild-runner/register ${file}`, { stdio: 'inherit' });
}