import { Random } from "./Random";

enum BinaryOp {
    ADD,
    SUB,
    MUL,
    DIV,
    POW
};

export class Matrix {
    public readonly data: Float32Array;
    public readonly shape: number[];
    public readonly strides: number[];
    public readonly offset: number;

    constructor(data: Array<any> | Float32Array, shape?: number[], strides?: number[], offset?: number) {
        this.data = data instanceof Float32Array ? data : new Float32Array(data.flat(Infinity));

        this.shape = shape === undefined || shape.length === 0 ? this.computeShape(data, []) : [...shape];
        this.strides = strides ? [...strides] : Matrix.computeStrides(this.shape);
        this.offset = offset ? offset : 0;
    }

    // Getter function for data
    public get(i): number {
        let idx = 0;
        let totalSize = this.shape.reduce((a, b) => a * b, 1);
        for (let dim = 0; dim < this.shape.length; ++dim) {
            totalSize /= this.shape[dim];
            const coord = Math.floor(i / totalSize);
            i -= coord * totalSize;
            idx += this.strides[dim] * coord;
        }
        return this.data[(idx + this.offset) % this.data.length];
    }

    private get1D(index: number): number {
        return this.data[this.offset + this.strides[0] * index];
    }

    private get2D(index: number): number {
        // calculate the row and column indices
        let row = Math.floor(index / this.shape[1]);
        let col = index % this.shape[1];

        // calculate the index into the data array
        let dataIndex = this.offset + this.strides[0] * row + this.strides[1] * col;

        return this.data[dataIndex];
    }

    public getValue(indices: number[]): number {
        // Map the indices based on the strides
        let index = 0;
        for (let i = 0; i < indices.length; i++) {
            index += this.strides[i] * indices[i];
        }
        return this.data[(index + this.offset) % this.data.length];
    }

    private getNestedData(indices: number[] = []): number | number[] {
        if (indices.length === this.shape.length) {
            return this.getValue(indices);
        }

        let dimSize = this.shape[indices.length];
        let nestedData = new Array(dimSize);
        for (let i = 0; i < dimSize; i++) {
            nestedData[i] = this.getNestedData([...indices, i]);
        }

        return nestedData;
    }

    public getData(): number | number[] {
        return this.getNestedData();
    }

    // TODO: Check for compatibility between data and shape
    private computeShape(data: any, shape: number[]): number[] {
        if (!data.length || data.length == 0) {
            return shape;
        }

        shape.push(data.length);
        return this.computeShape(data[0], shape);
    }

    public static computeStrides(shape): number[] {
        let strides: number[] = new Array(shape.length);
        strides[strides.length - 1] = 1;  // last stride is always 1
        for (let i = strides.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    public static slice(m: Matrix, indices: (number | null | number[])[]): Matrix {
        if (indices.length != m.shape.length) throw Error(`Indices [${indices}] must match matrix shape ${m.shape}`);

        let offset = m.offset;
        let newShape: number[] = [];
        let newStrides: number[] = [];
        for (let i = 0; i < indices.length; i++) {
            const index = indices[i];

            if (Array.isArray(index)) { // Handle slices
                const [start, stop] = index;
                if (start < 0 || start >= m.shape[i] || stop < 0 || stop > m.shape[i]) {
                    throw Error(`Slice ${start}:${stop} out of bounds for axis ${i} with size ${m.shape[i]}`);
                }
                offset += start * m.strides[i];
                newShape.push(stop - start);
                newStrides.push(m.strides[i]);
            } else if (index !== null) { // Handle integer indices
                if (index < 0 || index >= m.shape[i]) {
                    throw Error(`Index ${index} out of bounds for axis ${i} with size ${m.shape[i]}`);
                }
                offset += index * m.strides[i];
            } else { // Handle null (ellipsis) indices
                newShape.push(m.shape[i]);
                newStrides.push(m.strides[i]);
            }
        }
        return new Matrix(m.data, newShape, newStrides, offset);
    }

    public static multinomial(matrix: Matrix, num_samples: number, normalized: boolean = false): Matrix {
        const origRank = matrix.shape.length;
        const logits2D = origRank === 1 ? matrix.reshape([1, -1]) : matrix;

        const probabilities = normalized ? logits2D : Matrix.softmax(logits2D, -1);
        const batchSize = probabilities.shape[0];
        const numEvents = probabilities.shape[1];
        const probVals = probabilities.data;
        const resShape = [batchSize, num_samples];
        const resVals = new Float32Array(resShape.reduce((p, c) => p * c));

        for (let b = 0; b < batchSize; ++b) {
            const offset = b * numEvents;
            // The cdf won't include the last event. It will be implicit if no other
            // event happened.
            const cdf = new Float32Array(numEvents - 1);
            cdf[0] = probVals[offset];
            for (let event = 1; event < cdf.length; ++event) {
                cdf[event] = cdf[event - 1] + probVals[offset + event];
            }

            const outOffset = b * num_samples;
            for (let sampleId = 0; sampleId < num_samples; ++sampleId) {
                const r = Random.Random();

                // Assume last event happened by default.
                resVals[outOffset + sampleId] = cdf.length;

                for (let event = 0; event < cdf.length; event++) {
                    if (r < cdf[event]) {
                        resVals[outOffset + sampleId] = event;
                        break;
                    }
                }
            }
        }

        return new Matrix(resVals, resShape);
    }

    public static split(matrix: Matrix, split_sizes: number | number[], dim: null | number = null): Matrix[] {
        if (Array.isArray(split_sizes)) throw Error("Split split_sizes as array not supported");
        if (dim !== null) throw Error("Split dim not supported");

        const chunkSize = split_sizes;
        const stride = matrix.shape[matrix.shape.length - 1];
        if (stride % chunkSize !== 0) {
            throw new Error('Invalid chunk size, not evently divisible into last tensor dimension')
        }

        // Setup the output chunks
        const out: Matrix[] = [];
        const chunks = stride / chunkSize;

        for (let c = 0; c < chunks; c++) {
            out.push(Matrix.zeros([...matrix.shape.slice(0, matrix.shape.length - 1), chunkSize]));
        }
        const outOffsets = out.map(_ => 0);
        let sourceOffset = 0;

        // Split up the actual data
        const macroChunks = matrix.data.length / stride;
        for (let i = 0; i < macroChunks; i++) {
            for (let j = 0; j < chunks; j++) {
                out[j].data.set(matrix.data.slice(sourceOffset, sourceOffset + chunkSize), outOffsets[j])
                outOffsets[j] += chunkSize;
                sourceOffset += chunkSize
            }
        }

        return out;
    }

    // TODO: Handle negative axis and error checking
    public static concatenateArrays(arrays: Array<any>, axis: number | null = 0): number[] {
        if (axis < 0) {
            throw new Error(`Invalid axis value ${axis} ${arrays[0].length}`);
        }

        if (axis === 0) {
            // Vertical concatenation (stacking rows)
            return [].concat(...arrays);
        } else {
            // Concatenate along the specified axis
            const result: Array<any> = [];

            for (let i = 0; i < arrays[0].length; i++) {
                const newArrays = arrays.map(arr => arr[i]);
                result.push(Matrix.concatenateArrays(newArrays, axis - 1));
            }

            return result;
        }
    }

    public static cat(matrices: Matrix[], dim: number | null = 0): Matrix {
        if (dim === null) {
            const flattenData = matrices.map(v => Array.from(v.data)).flat(Infinity);
            return new Matrix(flattenData);
        }
        const data = matrices.map(m => m.getData());
        return new Matrix(Matrix.concatenateArrays(data, dim));
    }

    public static permute(m: Matrix, axes: number[] | null = null) {
        if (axes === null) {
            return new Matrix(m.data, [...m.shape].reverse(), [...m.strides].reverse());
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
        const ret = new Matrix(m.data, newShape, newStrides);
        return ret;
    }

    public static transpose(m: Matrix, dim0: number, dim1: number): Matrix {
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

        return Matrix.permute(m, axes);
    }

    public static isContiguous(m: Matrix): boolean {
        let stride = 1;
        for (let i = m.shape.length - 1; i >= 0; --i) {
            if (m.strides[i] !== stride) {
                return false;
            }
            stride *= m.shape[i];
        }
        return true;
    }

    public static reshape(m: Matrix, newShape: number | number[]): Matrix {
        if (newShape === -1) newShape = [m.data.length];
        if (!(newShape instanceof Array)) newShape = [newShape];

        const totalSize = m.shape.reduce((acc, val) => acc * val, 1);
        const missingSize = newShape.reduce((acc, val) => val >= 0 ? acc * val : acc, 1);
        const inferredShape = newShape.map(val => val === -1 ? totalSize / missingSize : val);

        // TODO: Check for valid shapes
        // if (inferredShape.reduce((p, c) => p * c) !== m.data.length) throw Error(`Shape ${inferredShape} is invalid for input of size ${m.data.length}`);

        const newStrides = new Array(inferredShape.length).fill(0);
        let stride = 1;
        for (let i = inferredShape.length - 1; i >= 0; --i) {
            newStrides[i] = stride;
            stride *= inferredShape[i];
        }

        let newData = m.data;

        const notContiguous = m.strides[m.strides.length - 1] !== 1 || !this.isContiguous(m);

        // check if copy is necessary
        if (m.strides[m.strides.length - 1] !== 1 || notContiguous) {
            // console.log("COPYING");
            newData = new Float32Array(totalSize);
            const coords = new Array(m.shape.length).fill(0);

            for (let i = 0; i < totalSize; ++i) {
                let index = 0;
                for (let j = 0; j < coords.length; ++j) {
                    index += m.strides[j] * coords[j];
                }
                newData[i] = m.data[index];
                ++coords[coords.length - 1];
                for (let j = coords.length - 1; j >= 0; --j) {
                    if (coords[j] >= m.shape[j] && j > 0) {
                        coords[j] = 0;
                        ++coords[j - 1];
                    }
                }
            }
        }

        return new Matrix(newData, inferredShape);
    }

    public static reshape2d(shape: number[]): number[] {
        let shape2d = [shape[0], 1];
        for (let i = 1; i < shape.length; i++) {
            shape2d[1] *= shape[i];
        }
        return shape2d;
    }

    public static reshape1d(shape: number[]): number {
        return shape.reduce((p, c) => p * c);
    }

    private static equalArrays(first: number[], second: number[]): boolean {
        if (first.length != second.length) return false;
        for (let i = 0; i < first.length; i++) {
            if (first[i] !== second[i]) return false;
        }
        return true;
    }

    public static broadcast(x: Matrix | number, y: Matrix | number): [Matrix, Matrix] {
        if (!(x instanceof Matrix)) x = new Matrix([x]);
        if (!(y instanceof Matrix)) y = new Matrix([y]);
        
        if (Matrix.equalArrays(x.shape, y.shape)) return [x, y];

        // Calculate the final shape after broadcasting
        let finalShape: number[] = [];
        let maxLength = Math.max(x.shape.length, y.shape.length);
        for (let i = 0; i < maxLength; i++) {
            finalShape.push(Math.max(x.shape[x.shape.length - i - 1] || 1, y.shape[y.shape.length - i - 1] || 1));
        }
        finalShape = finalShape.reverse(); // reverse because we filled the array from the end

        return [x.expand(finalShape), y.expand(finalShape)];
    }

    public broadcast(other: Matrix | number): [Matrix, Matrix] {
        return Matrix.broadcast(this, other);
    }

    public broadcast_to(shape: number[]): Matrix {
        return Matrix.broadcast(this, Matrix.zeros(shape))[0];
    }

    // Helpers
    private static MatrixFromNumber(value: number, shape: number[]): Matrix {
        const desiredElements = shape.reduce((acc, val) => acc * val, 1);
        if (value === 0) return new Matrix(new Float32Array(desiredElements), shape);
        return new Matrix(new Float32Array(desiredElements).fill(value), shape);
    }

    public static arange(start: number, stop: number, step: number = 1): Matrix {
        let data: Float32Array = new Float32Array(Math.floor((stop - start) / step));
        let s = 0;
        for (let i = start; i < stop; i += step) {
            data[s] = i;
            s++;
        }
        return new Matrix(data);
    }

    public static rand(shape: number[]): Matrix {
        let data: Float32Array = new Float32Array(shape.reduce((prev, curr) => prev * curr));
        for (let i = 0; i < data.length; i++) {
            data[i] = Random.Random();
        }
        return new Matrix(data, shape);
    }

    public static uniform(low: number, high: number, shape: number[]): Matrix {
        const m = Matrix.zeros(shape);

        for (let i = 0; i < m.data.length; i++) {
            m.data[i] = Random.RandomRange(low, high);
        }
        return m;
    }

    public static expand_dims(m: Matrix, axis: number | number[]): Matrix {
        if (!Array.isArray(axis)) axis = [axis];

        let newShape = [...m.shape];
        for (let i = 0; i < axis.length; i++) {
            newShape.splice(axis[i] < 0 ? newShape.length + axis[i] + 1 : axis[i], 0, 1);
        }

        return this.reshape(m, newShape);
    }

    public static unsqueeze(m: Matrix, dim: number): Matrix {
        if (dim < 0) dim = m.shape.length + dim + 1;
        return m.reshape([...m.shape.slice(0, dim), 1, ...m.shape.slice(dim)]);
    }

    public expand(shape: number[]): Matrix {
        const lenDiff = shape.length - this.shape.length;
        let oldShape = this.shape;
        let oldStrides = this.strides;

        if (lenDiff > 0) { // shape has more dimensions, adjust oldShape and oldStrides
            oldShape = Array(lenDiff).fill(1).concat(this.shape);
            oldStrides = Array(lenDiff).fill(0).concat(this.strides);
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

        const newShape = shape;
        return new Matrix(this.data, newShape, newStrides);
    }

    private static _tri(r: number, c: number, k: number = 0): Matrix {
        let a = Matrix.arange(0, r).unsqueeze(1).expand([r, c]);
        let b = Matrix.arange(-k, c - k).unsqueeze(0).expand([r, c]);

        return a.lte(b);
    }

    public static triu(m: Matrix, k: number = 0): Matrix {
        const a = Matrix._tri(m.shape[m.shape.length - 2], m.shape[m.shape.length - 1], k);
        return Matrix.where(a, m, Matrix.zeros(m.shape));
    }

    public static tril(m: Matrix, k: number = 0): Matrix {
        const a = Matrix._tri(m.shape[m.shape.length - 2], m.shape[m.shape.length - 1], k + 1);
        return Matrix.where(a, Matrix.zeros(m.shape), m);
    }

    public static masked_fill(m: Matrix, mask: Matrix, value: number): Matrix {
        const [mb, maskb] = Matrix.broadcast(m, mask);

        const fillMatrix = Matrix.full(mb.shape, value);
        const filled = Matrix.where(maskb, fillMatrix, mb);
        return filled;
    }

    public copy(): Matrix {
        return new Matrix(this.data.slice(), this.shape.slice());
    }

    public static zeros(size: number[]): Matrix {
        return Matrix.MatrixFromNumber(0, size);
    }

    public static ones(size: number[]): Matrix {
        return Matrix.MatrixFromNumber(1, size);
    }

    public static full(size: number[], value: number): Matrix {
        return Matrix.MatrixFromNumber(value, size);
    }

    private static binary_op(m1: Matrix, m2: Matrix, op: BinaryOp): Matrix {
        let newData = new Float32Array(m1.shape.reduce((p, c) => p * c));

        // Normal
        for (let i = 0; i < newData.length; i++) {

            let value = 0
            if (op == BinaryOp.ADD) value = m1.get(i) + m2.get(i);
            else if (op == BinaryOp.SUB) value = m1.get(i) - m2.get(i);
            else if (op == BinaryOp.MUL) value = m1.get(i) * m2.get(i);
            else if (op == BinaryOp.DIV) value = m1.get(i) / m2.get(i);
            else if (op == BinaryOp.POW) value = m1.get(i) ** m2.get(i);

            newData[i] = isNaN(value) ? 0 : value;
        }

        // console.timeEnd(`binary_op ${op}`);

        return new Matrix(newData, m1.shape);
    }

    public static add(m1: Matrix | number, m2: Matrix | number): Matrix {
        // return Matrix.binary_op(m1, m2, BinaryOp.ADD);
        return Matrix.binary_op(...Matrix.broadcast(m1, m2), BinaryOp.ADD);
    }

    public static sub(m1: Matrix | number, m2: Matrix | number): Matrix {
        // return Matrix.binary_op(m1, m2, BinaryOp.SUB);
        return Matrix.binary_op(...Matrix.broadcast(m1, m2), BinaryOp.SUB);
    }

    public static mul(m1: Matrix | number, m2: Matrix | number): Matrix {
        // return Matrix.binary_op(m1, m2, BinaryOp.MUL);
        return Matrix.binary_op(...Matrix.broadcast(m1, m2), BinaryOp.MUL);
    }

    public static div(m1: Matrix | number, m2: Matrix | number): Matrix {
        // return Matrix.binary_op(m1, m2, BinaryOp.DIV);
        return Matrix.binary_op(...Matrix.broadcast(m1, m2), BinaryOp.DIV);
    }

    public static pow(m1: Matrix | number, m2: Matrix | number): Matrix {
        // return Matrix.binary_op(m1, m2, BinaryOp.POW);
        return Matrix.binary_op(...Matrix.broadcast(m1, m2), BinaryOp.POW);
    }

    public static dot(m1: Matrix, m2: Matrix): Matrix {
        const x = m1.reshape([...m1.shape.slice(0, m1.shape.length - 1), 1, ...m1.shape.slice(m1.shape.length - 1, m1.shape.length)]);
        let w = m2.reshape([...m2.shape.slice(0, m2.shape.length - 2), 1, ...m2.shape.slice(m2.shape.length - 2, m2.shape.length - 1), ...m2.shape.slice(m2.shape.length - 1, m2.shape.length)]);
        w = w.transpose(-1, -2);

        let r = x.mul(w)
        r = r.sum(-1);

        if (m1.shape.length == 1) {
            r = r.reshape([...r.shape.slice(0, r.shape.length - 3), ...r.shape.slice(r.shape.length - 2, r.shape.length - 1)]);
        }

        return r;
    }

    public static reduce(m: Matrix, func: (m1: Matrix, m2: Matrix) => Matrix, axis: number | null = null, initialValue: number = 0, keepdims: boolean = false) {
        if (axis === null) {
            const result = m.data.reduce((a, b) => func(new Matrix(new Float32Array([a])), new Matrix(new Float32Array([b]))).data[0], initialValue);
            return new Matrix([result], keepdims ? m.shape.map(() => 1) : [1]);
        }

        if (axis < 0) axis += m.shape.length;

        const resultShape = [...m.shape];
        resultShape.splice(axis, 1);

        const numSlices = m.shape[axis];
        const sliceSize = m.data.length / numSlices;
        const stride = m.strides[axis];

        let resultData = new Float32Array();
        for (let slice = 0; slice < numSlices; slice++) {
            let sliceData = new Float32Array(sliceSize);
            for (let i = 0; i < sliceSize; i++) {
                const idx = Math.floor(i / stride) * stride * numSlices + i % stride + stride * slice;
                // sliceData.push(m.data[idx]);
                sliceData[i] = m.data[idx]
            }

            let sliceMatrix = new Matrix(sliceData, [sliceSize]);
            if (slice === 0) {
                resultData = sliceMatrix.data;
            } else {
                resultData = func(new Matrix(resultData, [resultData.length]), sliceMatrix).data;
            }
        }

        if (keepdims) {
            resultShape.splice(axis, 0, 1);
        }

        return new Matrix(resultData, resultShape);
    }

    public static reduce2(m: Matrix, func: (a: number, b: number) => number, axis: number | null = null, keepdims: boolean = false): Matrix {
        if (axis === null) {
            const data = [m.data.reduce((p, c) => func(p, c))];
            if (keepdims) {
                return new Matrix(data, m.shape.map(v => 1));
            }
            return new Matrix(data);
        }

        if (axis < 0) axis = m.shape.length + axis;

        function parseAxisParam(axis: number | number[], shape: number[]): number[] {
            const rank = shape.length;

            // Normalize input
            axis = axis == null ? shape.map((s, i) => i) : [].concat(axis);

            // Handle negative axis.
            return axis.map(a => a < 0 ? rank + a : a);
        }

        function getAxesPermutation(axes: number[], rank: number): number[] {
            const result: number[] = [];
            for (let i = 0; i < rank; ++i) {
                if (axes.indexOf(i) === -1) {
                    result.push(i);
                }
            }
            axes.forEach(axis => result.push(axis));
            return result;
        }

        const axes = parseAxisParam(axis, m.shape);
        const pa = getAxesPermutation(axes, m.shape.length);
        let p = m.permute(pa);





        const resultShape = [...m.shape];
        resultShape.splice(axis, 1);

        if (keepdims === true) {
            resultShape.splice(axis, 0, 1);
        }

        const sp = resultShape.reduce((p, c) => p * c);
        let output = new Float32Array(sp);

        const vals = m.shape[axis];
        let additionCounter = 0;

        const l = p.shape.reduce((p, c) => p * c);

        // reshape to 2d so that get2d can be used, faster!
        // p = p.reshape(Matrix.reshape1d(p.shape));
        p = p.reshape(Matrix.reshape2d(p.shape));

        for (let i = 0; i < l; i++) {
            for (let index = 0; index < vals; index++) {
                // output[additionCounter] = func(output[additionCounter], p.get(i));
                // output[additionCounter] = func(output[additionCounter], p.get1D(i));
                output[additionCounter] = func(output[additionCounter], p.get2D(i));
                i++;
            }

            additionCounter++;
            i--;
        }

        return new Matrix(output, resultShape);
    }
    
    public static sum(m: Matrix, axis: number | null = null, keepdims: boolean = false) {
        const sumOp = (a: number, b: number) => a + b;
        return Matrix.reduce2(m, sumOp, axis, keepdims);
    }

    public static prod(m: Matrix, axis: number | null = null): Matrix {
        // const mulOp = (a: number, b: number) => a * b;
        // return Matrix.reduce2(m, mulOp, axis, false);

        return Matrix.reduce(m, Matrix.mul, axis, 1);
    }

    public static mean(m: Matrix, axis: number | null = null, keepdims: boolean = false): Matrix {
        if (axis < 0) axis = m.shape.length + axis;

        const sum = Matrix.sum(m, axis, keepdims);
        if (axis === null) return sum.div(m.data.length);
        return sum.div(m.shape[axis]);
    }

    public static var(m: Matrix, axis: number | null = null, keepdims: boolean = false): Matrix {
        const x = Matrix.abs(m.sub(Matrix.mean(m, axis, true))).pow(2);
        return Matrix.mean(x, axis, keepdims);
    }

    public static softmax(m: Matrix, dim: number): Matrix {
        return m.exp().div(m.exp().sum(dim, true));
    }

    // TODO: No data manipulation
    public static abs(m: Matrix): Matrix {
        return new Matrix(m.data.map(v => v < 0 ? v * -1 : v), m.shape);
    }

    // TODO: No data manipulation
    public static exp(x: Matrix): Matrix {
        return new Matrix(x.data.map(v => Math.exp(v)), x.shape);
    }

    // TODO: No data manipulation
    public static tanh(x: Matrix): Matrix {
        return new Matrix(x.data.map(v => Math.tanh(v)), x.shape);
    }

    private compare(m: Matrix | number, sign: "<" | "<=" | ">" | ">=" | "==") {
        m = m instanceof Matrix ? m : new Matrix([m]);
        const [m1, m2] = Matrix.broadcast(this, m);

        const data = new Float32Array(m1.shape.reduce((p, c) => p * c));

        for (let i = 0; i < m1.shape.reduce((p, c) => p * c); i++) {
            if (sign == "<") data[i] = m1.get(i) < m2.get(i) ? 1 : 0;
            else if (sign == "<=") data[i] = m1.get(i) <= m2.get(i) ? 1 : 0;
            else if (sign == ">") data[i] = m1.get(i) > m2.get(i) ? 1 : 0;
            else if (sign == ">=") data[i] = m1.get(i) >= m2.get(i) ? 1 : 0;
            else if (sign == "==") data[i] = m1.get(i) == m2.get(i) ? 1 : 0;
        }
        return new Matrix(data, m1.shape);
    }

    public gt(value: number | Matrix): Matrix {
        return this.compare(value, ">");
    }

    public gte(value: number | Matrix): Matrix {
        return this.compare(value, ">=");
    }

    public lt(value: number | Matrix): Matrix {
        return this.compare(value, "<");
    }

    public lte(value: number | Matrix): Matrix {
        return this.compare(value, "<=");
    }

    public equal(value: number | Matrix): Matrix {
        return this.compare(value, "==");
    }

    public static where(condition: Matrix, x: Matrix, y: Matrix) {
        return new Matrix(x.data.map((v, i) => {
            return condition.get(i) ? x.get(i) : y.get(i);
        }), x.shape);

    }

    public static maximum(x1: Matrix, x2: Matrix) {
        return Matrix.where(x1.gte(x2), x1, x2);
    }





    // Fillers
    public add(m: Matrix | number): Matrix { return Matrix.add(this, m); };
    public sub(m: Matrix | number): Matrix { return Matrix.sub(this, m); };
    public mul(m: Matrix | number): Matrix { return Matrix.mul(this, m); };
    public div(m: Matrix | number): Matrix { return Matrix.div(this, m); };
    public pow(m: Matrix | number): Matrix { return Matrix.pow(this, m); };

    public exp(): Matrix { return Matrix.exp(this) };
    public dot(m: Matrix): Matrix { return Matrix.dot(this, m) };
    public split(split_sizes: number | number[], dim: null | number = null): Matrix[] { return Matrix.split(this, split_sizes, dim) };
    public masked_fill(mask: Matrix, value: number): Matrix { return Matrix.masked_fill(this, mask, value) };
    public unsqueeze(dim: number): Matrix { return Matrix.unsqueeze(this, dim); }
    public sum(axis: number | null = null, keepdims: boolean = false): Matrix { return Matrix.sum(this, axis, keepdims) };
    public mean(axis: number | null = null, keepdims: boolean = false): Matrix { return Matrix.mean(this, axis, keepdims) };
    public var(axis: number | null = null, keepdims: boolean = false): Matrix { return Matrix.var(this, axis, keepdims) };
    public reshape(shape: number | number[]): Matrix { return Matrix.reshape(this, shape) };
    public permute(axes: number[] | null = null): Matrix { return Matrix.permute(this, axes) };
    public transpose(dim0: number, dim1: number): Matrix { return Matrix.transpose(this, dim0, dim1) };


    public prod(axis?: number): Matrix { return Matrix.prod(this, axis) };
    public tril(k?: number): Matrix { return Matrix.tril(this, k) };

    public get T(): Matrix { return Matrix.permute(this); }

    toString() {
        function fixed(key, val) {
            return val.toFixed ? Number(val.toFixed(4)) : val;
        }
        return `Matrix(${JSON.stringify(this.getData(), fixed)}, shape=[${this.shape}])`;
    }
}