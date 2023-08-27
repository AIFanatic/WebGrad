import { Backend } from "./Backend";
import { TensorBuffer } from "./TensorBuffer";

export class MovementOp {
    // TODO: Fix this, should be an op along from FROM, CONTIGUOUS etc
    // This is slow, there are more efficient ways to make it contiguous like updating the data as needed instead of the whole thing
    public static contiguous(x: TensorBuffer): TensorBuffer {
        // return new Tensor(x.data.getData(), {device: x.device});

        return x.contiguous();
    }
    
    // TODO: Error handling
    public static reshape(x: TensorBuffer, shape: number | number[]): TensorBuffer {
        if (!(shape instanceof Array)) shape = [shape];

        const totalSize = x.shape.reduce((acc, val) => acc * val, 1);
        const missingSize = shape.reduce((acc, val) => val >= 0 ? acc * val : acc, 1);
        const inferredShape = shape.map(val => val === -1 ? totalSize / missingSize : val);

        // TODO: Check for valid shapes
        // if (inferredShape.reduce((p, c) => p * c) !== x.data.length) throw Error(`Shape ${inferredShape} is invalid for input of size ${x.data.length}`);

        if (!x.is_contiguous()) {
            return MovementOp.reshape(MovementOp.contiguous(x), inferredShape);
        }

        return Backend.CreateFromDataShapeAndStrides(x, inferredShape, TensorBuffer.computeStrides(inferredShape));
    }

    public static expand(x: TensorBuffer, shape: number[]): TensorBuffer {
        const lenDiff = shape.length - x.shape.length;
        let oldShape = x.shape;
        let oldStrides = x.strides;

        if (lenDiff > 0) { // shape has more dimensions, adjust oldShape and oldStrides
            oldShape = Array(lenDiff).fill(1).concat(x.shape);
            oldStrides = Array(lenDiff).fill(0).concat(x.strides);
        }

        // replace -1 with the corresponding value from the original shape
        for (let i = 0; i < shape.length; i++) {
            if (shape[i] == -1) {
                if (i >= oldShape.length) {
                    throw new Error('Cannot infer dimension for expansion');
                }
                shape[i] = oldShape[i];
            }
        }

        let newStrides = new Array(shape.length).fill(0);
        for (let i = 0; i < shape.length; i++) {
            if (shape[i] == oldShape[i]) {
                newStrides[i] = oldStrides[i];
            } else if (oldShape[i] == 1) {
                newStrides[i] = 0;
            }
        }

        return Backend.CreateFromDataShapeAndStrides(x, shape, newStrides);
    }

    public static permute(m: TensorBuffer, axes: number[] | null = null) {
        if (axes === null) {
            // return new Tensor(m.data, [...m.shape].reverse(), [...m.strides].reverse());
            return Backend.CreateFromDataShapeAndStrides(m, [...m.shape].reverse(), [...m.strides].reverse());

        }

        // Permute the axes according to the axes argument
        let newShape: number[] = [];
        let newStrides: number[] = [];
        for (let i = 0; i < axes.length; i++) {
            let axis = axes[i] < 0 ? m.shape.length + axes[i] : axes[i];

            newShape[i] = m.shape[axis];
            newStrides[i] = m.strides[axis];
        }
        // Call the existing transpose method with the new axes
        // const ret = new Tensor(m.data, newShape, newStrides);
        // return ret;
        return Backend.CreateFromDataShapeAndStrides(m, newShape, newStrides);
    }

    public static transpose(m: TensorBuffer, dim0: number, dim1: number): TensorBuffer {
        // Ensure dim0 and dim1 are positive and within the range of shape's length
        dim0 = dim0 < 0 ? m.shape.length + dim0 : dim0;
        dim1 = dim1 < 0 ? m.shape.length + dim1 : dim1;

        if (dim0 >= m.shape.length || dim1 >= m.shape.length) {
            throw new Error('Transpose dimensions out of range');
        }

        // Generate the original axes
        let axes: number[] = [];
        for (let i = 0; i < m.shape.length; i++) {
            axes[i] = i;
        }

        // Swap the two dimensions
        let tmp = axes[dim0];
        axes[dim0] = axes[dim1];
        axes[dim1] = tmp;

        return MovementOp.permute(m, axes);
    }
}