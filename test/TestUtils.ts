import { Matrix } from "../src/Matrix";
import { Tensor } from "../src/Tensor";

export async function describe(name: string, func: Function) {
    const colors = {"red": "\x1b[31m", "green": "\x1b[32m", "yellow": "\x1b[33m"};

    let passed = false;
    try {
        await func()

        console.log(`${colors["yellow"]}[${name}] ${colors["green"]} Passed`);

    } catch (error) {
        console.error("Error", error);
        passed = false;
        console.log(`${colors["yellow"]}[${name}] ${colors["red"]} Failed`);
        throw Error("Test failed")
    }
}

export function assert(condition: boolean) {
    if (!condition) throw Error("Ups, condition is false");
}

// function equalTensorBuffer(a: TensorBuffer, b: TensorBuffer, EPS: number = 1e-5): boolean {
//     if (a.shape.length !== b.shape.length ) {
//         console.log(`Shape of a (${a.shape}) doesn't match shape of b (${b.shape})`);
//         return false;
//     }

//     for (let i = 0; i < a.shape.length; i++) {
//         if (a.shape[i] !== b.shape[i]) {
//             console.log(`Shape of a (${a.shape}) doesn't match shape of b (${b.shape})`);
//             return false;
//         }
//     }

//     for (let i = 0; i < a.shape.reduce((p, c) => p * c); i++) {
//         const aV = a.get(i);
//         const bV = b.get(i);
//         if (aV === undefined || bV === undefined) {
//             console.log(`Expected values but got a[${i}]=${aV} and b[${i}]=${bV}`);
//             return false;
//         }
//         else if (aV === null || bV === null) {
//             console.log(`Expected values but got a[${i}]=${aV} and b[${i}]=${bV}`);
//             return false;
//         }
//         if (Math.abs(aV - bV) > EPS) {
//             console.log(`Data of a (${aV}) doesn't match data of b (${bV})`);
//             return false;
//         }
//     }
    
//     return true;
// }

function equalMatrix(a: Matrix, b: Matrix, EPS: number = 1e-5): boolean {
    if (a.shape.length !== b.shape.length ) {
        console.log(`Shape of a (${a.shape}) doesn't match shape of b (${b.shape})`);
        return false;
    }

    for (let i = 0; i < a.shape.length; i++) {
        if (a.shape[i] !== b.shape[i]) {
            console.log(`Shape of a (${a.shape}) doesn't match shape of b (${b.shape})`);
            return false;
        }
    }

    for (let i = 0; i < a.shape.reduce((p, c) => p * c); i++) {
        const aV = a.get(i);
        const bV = b.get(i);
        if (aV === undefined || bV === undefined) {
            console.log(`Expected values but got a[${i}]=${aV} and b[${i}]=${bV}`);
            return false;
        }
        else if (aV === null || bV === null) {
            console.log(`Expected values but got a[${i}]=${aV} and b[${i}]=${bV}`);
            return false;
        }
        if (Math.abs(aV - bV) > EPS) {
            console.log(`Data of a (${aV}) doesn't match data of b (${bV})`);
            return false;
        }
    }
    
    return true;
}

function equalTensor(a: Tensor, b: Tensor, EPS: number = 1e-5): boolean {
    const equalData = equalMatrix(a.data, b.data, EPS);
    const equalGrads = a.grad && b.grad ? equalFloat32Array(a.grad.data, b.grad.data, EPS) : true;
    return equalData && equalGrads;
}

function equalArray(a: number[], b: number[], EPS: number = 1e-5): boolean {
    if (a.length !== b.length) {
        console.log(`Length of a(${a.length}) not equal to length of b(${b.length})`);
        return false;
    }
    
    for (let i = 0; i < a.length; i++) {
        if (Math.abs(a[i] - b[i]) > EPS) {
            console.log(`Value of a[${i}]=${a[i]} not equal to value of b[${i}]=${b[i]}`);
            return false;
        }
    }
    return true;
}

function equalFloat32Array(a: Float32Array, b: Float32Array, EPS: number = 1e-5): boolean {
    if (a.length !== b.length) {
        console.log(`Length of a(${a.length}) not equal to length of b(${b.length})`);
        return false;
    }
    
    for (let i = 0; i < a.length; i++) {
        if (Math.abs(a[i] - b[i]) > EPS) {
            console.log(`Value of a[${i}]=${a[i]} not equal to value of b[${i}]=${b[i]}`);
            return false;
        }
    }
    return true;
}

export function equal(a: number[] | Tensor | Matrix, b: number[] | Tensor | Matrix, EPS: number = 1e-5): boolean {
    if (a instanceof Tensor && b instanceof Tensor) return equalTensor(a,b,EPS);
    else if (a instanceof Matrix && b instanceof Matrix) return equalMatrix(a,b,EPS);
    else if (a instanceof Array && b instanceof Array) return equalArray(a,b,EPS);

    throw Error("Tried to compare unknown type");
    return false;
}

export interface ITensorFactory {
    data: Array<any>;
    grad: Array<any>;
}

// export function TensorFactory(data: Matrix, grad: Matrix): Tensor {
//     const tensor = new Tensor(data);
//     tensor.grad = grad;
//     return tensor;
// }

export function TensorFactory(tensorData: ITensorFactory): Tensor {
    const tensor = new Tensor(tensorData.data);
    tensor.grad = new Matrix(tensorData.grad);
    return tensor;
}