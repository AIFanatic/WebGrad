import { Device } from "./Backend";
import { TensorBuffer } from "./TensorBuffer";
import { UnaryOps } from "./UnaryOps";
import { BinaryOps } from "./BinaryOps";
import { ReduceOps } from "./ReduceOps";

export class CPUBuffer extends TensorBuffer {
    public readonly data: Float32Array;

    public readonly shape: number[];
    public readonly strides: number[];
    public readonly offset: number;

    public static EPS = 1e-5;

    constructor(data: CPUBuffer | Float32Array, shape: number[], strides: number[], offset: number) {
        super(shape, strides, offset, Device.CPU);

        if (data instanceof Float32Array) this.data = data.slice();
        else if (data instanceof CPUBuffer) this.data = data.data.slice();
    }

    public static CreateFromArray(array: Array<any>): CPUBuffer {
        const data = Float32Array.from(array.flat(Infinity));
        const shape = TensorBuffer.computeShape(array);
        const strides = TensorBuffer.computeStrides(shape);

        return new CPUBuffer(data, shape, strides, 0);
    }

    public static CreateFromFloat32Array(array: Float32Array, shape: number[] = [], strides: number[] = [], offset: number = 0): CPUBuffer {
        const _shape = shape.length === 0 ? [array.length] : shape;
        const _strides = strides.length === 0 ? this.computeStrides(_shape) : strides;

        return new CPUBuffer(array, _shape, _strides, offset);
    }

    public static CreateFromNumber(num: number): CPUBuffer {
        return new CPUBuffer(new Float32Array([num]), [1], [1], 0);
    }

    protected getInternalData(): Float32Array {
        return this.data;
    }

    public copy(): CPUBuffer {
        return new CPUBuffer(this.data.slice(), this.shape, this.strides, this.offset);
    }

    public unary_op(op: UnaryOps): CPUBuffer {
        if (op === UnaryOps.ABS) return new CPUBuffer(new Float32Array(this.data.map(v => Math.abs(v))), this.shape, this.strides, this.offset);
        else if (op === UnaryOps.EXP) return new CPUBuffer(new Float32Array(this.data.map(v => Math.exp(v))), this.shape, this.strides, this.offset);
        else if (op === UnaryOps.TANH) return new CPUBuffer(new Float32Array(this.data.map(v => Math.tanh(v))), this.shape, this.strides, this.offset);
        else if (op === UnaryOps.LOG) return new CPUBuffer(new Float32Array(this.data.map(v => Math.log(v))), this.shape, this.strides, this.offset);
    }

    public binary_op(other: CPUBuffer, op: BinaryOps): CPUBuffer {
        let [_m1b, _m2b] = [this, other];

        let newData = new Float32Array(_m1b.shape.reduce((p, c) => p * c));

        for (let i = 0; i < newData.length; i++) {
            const v1 = _m1b.get(i);
            const v2 = _m2b.get(i);
            let value = 0;
            if (op == BinaryOps.ADD) value = v1 + v2;
            else if (op == BinaryOps.SUB) value = v1 - v2;
            else if (op == BinaryOps.MUL) value = v1 * v2;
            else if (op == BinaryOps.DIV) value = v1 / v2;
            else if (op == BinaryOps.POW) value = v1 ** v2;
            else if (op == BinaryOps.CMPEQ) value = Math.abs(v1 - v2) < CPUBuffer.EPS === true ? 1 : 0;
            else if (op == BinaryOps.MAX) value = v1 > v2 ? v1 : v2;

            newData[i] = isNaN(value) ? 0 : value;
        }

        const shape = _m1b.shape.slice();
        return new CPUBuffer(newData, shape, TensorBuffer.computeStrides(shape), _m1b.offset);
    }

    public reduce_op(op: ReduceOps, axes: number[]): CPUBuffer {
        function computeOutAndReduceShapes(aShape: number[], axes: number[]): [number[], number[]] {
            const outShape = [];
            const rank = aShape.length;
            for (let dim = 0; dim < rank; dim++) {
                if (axes.indexOf(dim) === -1) {
                    outShape.push(aShape[dim]);
                }
            }
            const reduceShape = axes.map(dim => aShape[dim]);
            return [outShape, reduceShape];
        }

        
        let [outShape, reduceShape] = computeOutAndReduceShapes(this.shape, axes);
        outShape = outShape.length === 0 ? [1] : outShape;

        const sp = outShape.reduce((p, c) => p * c);
        let output = new Float32Array(sp);
        if (op === ReduceOps.PROD) output.fill(1); // Initialize all elements to 1 for product operation

        // console.log(`outShape ${outShape} reduceShape ${reduceShape}`)
        const vals = reduceShape.reduce((p, c) => p * c);
        let additionCounter = 0;

        const length = this.shape.reduce((p, c) => p * c);

        for (let i = 0; i < length; i++) {
            for (let index = 0; index < vals; index++) {
                if (op === ReduceOps.SUM) {
                    output[additionCounter] += this.get(i);
                } else if (op === ReduceOps.PROD) {
                    output[additionCounter] *= this.get(i);
                }
                i++;
            }

            additionCounter++;
            i--;
        }

        return new CPUBuffer(output, outShape, TensorBuffer.computeStrides(outShape), 0);
    }

    public contiguous(): CPUBuffer {
        const r = CPUBuffer.CreateFromArray(this.getData());
        // console.log(`r ${r}`);
        return r;
    }

    public toString(): string {
        function fixed(key, val) {
            return val.toFixed ? Number(val.toFixed(4)) : val;
        }
        return `${JSON.stringify(this.getData(), fixed)}`;
    }
}