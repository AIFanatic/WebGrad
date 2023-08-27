import { Device } from "./Backend";
import { BinaryOps } from "./BinaryOps";
import { ReduceOps } from "./ReduceOps";
import { UnaryOps } from "./UnaryOps";

export class TensorBuffer {
    public readonly shape: number[];
    public readonly strides: number[];
    public readonly offset: number;
    public readonly device: Device;

    constructor(shape: number[], strides: number[], offset: number, device: Device) {
        this.shape = shape.slice();
        this.strides = strides.slice();
        this.offset = offset;
        this.device = device;
    }

    public copy(): TensorBuffer { throw Error("Not implemented"); }
    public static CreateFromArray(array: Array<any>): TensorBuffer { throw Error("Not implemented"); }
    public static CreateFromFloat32Array(array: Float32Array, shape: number[] = [], strides: number[] = [], offset: number = 0): TensorBuffer { throw Error("Not implemented"); }
    public static CreateFromNumber(num: number): TensorBuffer { throw Error("Not implemented"); }
    
    public unary_op(op: UnaryOps): TensorBuffer { throw Error("UnaryOp not implemented"); }
    public binary_op(other: TensorBuffer, op: BinaryOps): TensorBuffer { throw Error("BinaryOp not implemented"); }
    public reduce_op(op: ReduceOps, axis: number | null, inputShape: number[], resultShape: number[]): TensorBuffer { throw Error("ReduceOp not implemented"); }
    
    public contiguous(): TensorBuffer { throw Error("ReduceOp not implemented"); }

    public toString(): string { throw Error("toString not implemented."); }

    // TODO: Check for compatibility between data and shape
    public static computeShape(data: Array<any> | Float32Array, shape: number[] = []): number[] {
        if (!data.length || data.length == 0) {
            return shape;
        }

        shape.push(data.length);
        return TensorBuffer.computeShape(data[0], shape);
    }

    public static computeStrides(shape: number[]): number[] {
        let strides: number[] = new Array(shape.length);
        strides[strides.length - 1] = 1;  // last stride is always 1
        for (let i = strides.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        return strides;
    }

    protected getInternalData(): Float32Array { throw Error("Not implemented"); }

    public get(i): number {
        const data = this.getInternalData();

        let idx = 0;
        let totalSize = this.shape.reduce((a, b) => a * b, 1);
        for (let dim = 0; dim < this.shape.length; ++dim) {
            totalSize /= this.shape[dim];
            const coord = Math.floor(i / totalSize);
            i -= coord * totalSize;
            idx += this.strides[dim] * coord;
        }
        return data[(idx + this.offset) % data.length];
    }

    public getValue(data: Float32Array, indices: number[]): number {
        let index = 0;
        for (let i = 0; i < indices.length; i++) {
            index += this.strides[i] * indices[i];
        }
        return data[(index + this.offset) % data.length];
    }

    private getNestedData(data: Float32Array, indices: number[] = []): number | number[] {
        if (indices.length === this.shape.length) {
            return this.getValue(data, indices);
        }

        let dimSize = this.shape[indices.length];
        let nestedData = new Array(dimSize);
        for (let i = 0; i < dimSize; i++) {
            nestedData[i] = this.getNestedData(data, [...indices, i]);
        }

        return nestedData;
    }

    public getData(): number[] {
        const data = this.getInternalData();
        // Need to slice away extra data from potential mismatching shape and data length?
        const finalData = this.getNestedData(data);
        return finalData instanceof Array ? finalData : [finalData];
    }

    public is_contiguous(): boolean {
        let expected_stride = 1;
    
        // Start from the last dimension and move backward
        for (let i = this.strides.length - 1; i >= 0; i--) {
            if (this.strides[i] !== expected_stride) {
                return false;
            }
            expected_stride *= this.shape[i]; // Assuming you have a 'shape' property that holds the sizes of each dimension
        }
        
        return true;
    }
}