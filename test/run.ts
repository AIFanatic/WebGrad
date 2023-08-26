const glob = require('glob');
const { execSync } = require('child_process');

// File pattern using wildcard (*.js)
const filePattern = './test/*test.ts';

// Find all matching file paths
glob(filePattern, (err, files) => {
    if (err) throw err;

    // Iterate over the file paths and run each file individually
    files.forEach((file) => {
        console.log(`--------`);
        console.log(`Running test: ${file}`);
        execSync(`node -r esbuild-runner/register ${file}`, { stdio: 'inherit' });
    });
});