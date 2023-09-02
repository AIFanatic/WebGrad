import { MovementOp } from "./MovementOps";
import { TensorBuffer } from "./TensorBuffer";

export enum ReduceOps {
    SUM,
    PROD
};

export class ReduceOp {
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

    private static reduce_op(x: TensorBuffer, op: ReduceOps, axis: number[] | number | null, keepdim: boolean): TensorBuffer {
        const origAxes = ReduceOp.parseAxisParam(axis, x.shape);
        let axes = origAxes;
        const permutedAxes = ReduceOp.getAxesPermutation(axes, x.shape.length);
        const inputIsTransposed = permutedAxes !== null;

        let input = x;
        if (inputIsTransposed) {
            input = MovementOp.permute(input, permutedAxes);
            axes = ReduceOp.getInnerMostAxes(axes.length, x.shape.length);
            input = MovementOp.contiguous(input);
        }

        const r = input.reduce_op(op, axes, x.shape, []);
        
        if (keepdim) {
            const s = r.shape;
            let shape = s.length === 1 && s[0] === 1 ? [] : s;
            let newShape = ReduceOp.expandShapeToKeepDim(shape, origAxes);
            if (newShape.length === 1 && newShape[0] === undefined) newShape = [s.reduce((p, c) => p * c)];

            return MovementOp.reshape(r, newShape);
        }

        return r;
    }

    public static sum(x: TensorBuffer, axis: number[] | number | null, keepdim: boolean) { return ReduceOp.reduce_op(x, ReduceOps.SUM, axis, keepdim); }
    public static prod(x: TensorBuffer, axis: number [] | number | null, keepdim: boolean) { return ReduceOp.reduce_op(x, ReduceOps.PROD, axis, keepdim); }
}