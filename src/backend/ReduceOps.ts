import { MovementOp } from "./MovementOps";
import { TensorBuffer } from "./TensorBuffer";

export enum ReduceOps {
    SUM,
    PROD
};

export class ReduceOp {
    private static combineLocations(outputLoc: number[], reduceLoc: number[], axes: number[]): number[] {
        const rank = outputLoc.length + reduceLoc.length;
        const loc = [];
        let outIdx = 0;
        let reduceIdx = 0;
        for (let dim = 0; dim < rank; dim++) {
            if (axes.indexOf(dim) === -1) {
                loc.push(outputLoc[outIdx++]);
            } else {
                loc.push(reduceLoc[reduceIdx++]);
            }
        }
        return loc;
    }

    private static expandShapeToKeepDim(shape: number[], axes: number[]): number[] {
        const reduceSubShape = axes.map(x => 1);
        return ReduceOp.combineLocations(shape, reduceSubShape, axes);
    }

    private static axesAreInnerMostDims(axes: number[], rank: number): boolean {
        for (let i = 0; i < axes.length; ++i) {
            if (axes[axes.length - i - 1] !== rank - 1 - i) {
                return false;
            }
        }
        return true;
    }

    private static parseAxisParam(axis: number | number[], shape: number[]): number[] | null {
        const rank = shape.length;

        // Normalize input
        axis = axis == null ? shape.map((s, i) => i) : [].concat(axis);

        // Handle negative axis.
        return axis.map(a => a < 0 ? rank + a : a);
    }

    private static getAxesPermutation(axes: number[], rank: number): number[] {
        const result: number[] = [];
        for (let i = 0; i < rank; ++i) {
            if (axes.indexOf(i) === -1) {
                result.push(i);
            }
        }
        axes.forEach(axis => result.push(axis));
        return result;
    }

    private static getInnerMostAxes(numAxes: number, rank: number): number[] {
        const res: number[] = [];
        for (let i = rank - numAxes; i < rank; ++i) {
            res.push(i);
        }
        return res;
    }

    private static reduce_op(x: TensorBuffer, op: ReduceOps, axis: number[] | number | null, keepdim: boolean): TensorBuffer {
        // if (axis < 0) axis = x.shape.length + axis;

        const origAxes = ReduceOp.parseAxisParam(axis, x.shape);
        let axes = origAxes;
        const permutedAxes = ReduceOp.getAxesPermutation(axes, x.shape.length);
        const inputIsTransposed = permutedAxes !== null;

        let input = x;
        if (inputIsTransposed) {
            // console.log(`permutedAxes ${permutedAxes}`);
            // console.log(`input1 ${input}`);
            // if (x.shape.length != permutedAxes.length) {
            //     const bigger = permutedAxes.reduce((p, c) => p > c ? p : c);
            //     console.log("bigger", bigger)
            //     let expandedShape = [];
            //     for (let i = 0; i < bigger; i++) {
            //         expandedShape.push(x.shape[i] !== undefined ? x.shape[i] : 1);
            //     }
                
                
            //     console.log("expandedShape", expandedShape)
            //     input = MovementOp.expand(x, expandedShape);
            //     console.log(`input2 ${input}`);
            //     // throw Error("GERE")
            // }
            input = MovementOp.permute(input, permutedAxes);
            // console.log("input3", input);
            // throw Error("HERE")
            axes = ReduceOp.getInnerMostAxes(axes.length, x.shape.length);
            input = MovementOp.contiguous(input);
        }

        // console.log(`input ${axes} ${input}`);

        let resultShape = [...x.shape];

        // console.log(`TODO:
        //                 keepDim is fucked Tensor.reduce.test.ts.
        //                 test expand backward
        //                 adapt/test webgl reduce to support array (axes)
        // `)
        // resultShape.splice(axis, 1);

        // if (keepdim === true) {
        //     resultShape.splice(axis, 0, 1);
        // }
        // if (axis === null) {
        //     resultShape = keepdim ? x.shape.map(v => 1) : [1];
        // }

        const r = input.reduce_op(op, axes, x.shape, resultShape);

        if (keepdim) {
            // with tfjs when shape is [1] shape is [], account for that
            let shape = r.shape.length === 1 && r.shape[0] === 1 ? [] : r.shape;
            const newShape = ReduceOp.expandShapeToKeepDim(shape, origAxes);
            // console.log(`newShape ${newShape} ${r.shape} ${origAxes}`);
            return MovementOp.reshape(r, newShape);
        }
        // return MovementOp.reshape(r, resultShape);
        return r;
    }

    public static sum(x: TensorBuffer, axis: number[] | number | null, keepdim: boolean) { return ReduceOp.reduce_op(x, ReduceOps.SUM, axis, keepdim); }
    public static prod(x: TensorBuffer, axis: number [] | number | null, keepdim: boolean) { return ReduceOp.reduce_op(x, ReduceOps.PROD, axis, keepdim); }
}