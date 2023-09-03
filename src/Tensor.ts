import { Operations, Random } from ".";
import { Operation } from "./Operations";
import { Backend, Device } from "./backend/Backend";
import { BinaryOp } from "./backend/BinaryOps";
import { MovementOp } from "./backend/MovementOps";
import { TensorBuffer } from "./backend/TensorBuffer";

export type TensorDataTypes = Tensor | TensorBuffer | Array<any> | Float32Array | number;

export interface TensorOptions {
    _children?: Tensor[];
    _op?: Operation;
    device?: Device;
    requires_grad?: boolean;
};

const DefaultTensorOptions: TensorOptions = {
    _children: [],
    _op: null,
    device: Device.CPU,
    requires_grad: false,
};

export class Tensor {
    public id: string;

    public data: TensorBuffer;
    public grad: TensorBuffer;
    public _children: Tensor[];
    public _prev: Set<Tensor>;
    public _op: Operation;
    public device: Device;
    public requires_grad: boolean;
    private options: TensorOptions;

    public get shape(): number[] { return this.data.shape; }
    public get strides(): number[] { return this.data.strides; }
    public get offset(): number { return this.data.offset; }

    constructor(data: TensorDataTypes, options?: TensorOptions) {
        this.id = "P" + Math.floor(Math.random() * 1000000).toString().padStart(6, "0");

        const _options: TensorOptions = Object.assign({}, DefaultTensorOptions, options);

        if (_options._children.length !== 0) {
            _options.requires_grad = _options._children[0].requires_grad;
        }

        if (data instanceof Tensor) {
            this.data = data.data;
            _options.device = options && options.device ? options.device : data.device;

            // _options._op = options && options._op !== null ? options._op : data._op;
            // _options._children = options && options._children && options._children.length !== 0 ? options._children : data._children;
        }
        else if (data instanceof TensorBuffer) {
            this.data = data;
            _options.device = options && options.device ? options.device : data.device;
        }
        else if (data instanceof Array) this.data = Backend.CreateFromArray(_options.device, data);
        else if (data instanceof Float32Array) this.data = Backend.CreateFromFloat32Array(_options.device, data);
        else if (!isNaN(data)) this.data = Backend.CreateFromNumber(_options.device, data);

        this.grad = Backend.CreateFromFloat32Array(_options.device, new Float32Array([0]), this.shape, TensorBuffer.computeStrides(this.shape));
        this.device = _options.device;
        this.requires_grad = _options.requires_grad;
        this._op = _options._op;
        this._prev = new Set(_options._children);
        this._children = _options._children;

        this.options = _options;
        // console.warn(_options.device);
    }

    public backward() {
        let topo: Tensor[] = [];
        let visited = new Set<Tensor>();

        function build_topo(v: Tensor) {
            if (!visited.has(v)) {
                visited.add(v);
                for (let child of v._prev) {
                    build_topo(child);
                }
                topo.push(v);
            }
        }

        build_topo(this);

        this.grad = Tensor.ones(this.data.shape, {device: this.device}).data;

        for (let v of topo.reverse()) {
            if (v._op === null) continue;
            const grads = v._op.backward(v.grad);
            if (grads) {
                for (let i = 0; i < grads.length; i++) {
                    if (grads[i] !== null) {
                        if (v._children[i].grad) {
                            v._children[i].grad = BinaryOp.add(v._children[i].grad, grads[i]);
                        } else {
                            v._children[i].grad = grads[i];
                        }
                    }
                }
            }
        }
    }

    // Helpers

    // TODO: Should use strides and shape to do this
    private static TensorFromNumber(value: number, shape: number[], options?: TensorOptions): Tensor {
        const desiredElements = shape.reduce((acc, val) => acc * val, 1);
        const device = options && options.device ? options.device : Device.CPU;
        if (value === 0) return new Tensor(Backend.CreateFromFloat32Array(device, new Float32Array(desiredElements), shape), options);
        return new Tensor(Backend.CreateFromFloat32Array(device, new Float32Array(desiredElements).fill(value), shape), options);
        // if (value === 0) return new Tensor(new Float32Array(desiredElements), options).reshape(shape);
        // return new Tensor(new Float32Array(desiredElements).fill(value), options).reshape(shape);
    }

    public static zeros(size: number[], options?: TensorOptions): Tensor {
        return Tensor.TensorFromNumber(0, size, options);
    }

    public static ones(size: number[], options?: TensorOptions): Tensor {
        return Tensor.TensorFromNumber(1, size, options);
    }

    public static full(size: number[], value: number, options?: TensorOptions): Tensor {
        return Tensor.TensorFromNumber(value, size, options);
    }

    public static arange(start: number, stop: number, step: number = 1, options?: TensorOptions): Tensor {
        let data: Float32Array = new Float32Array(Math.floor((stop - start) / step));
        let s = 0;
        for (let i = start; i < stop; i += step) {
            data[s] = i;
            s++;
        }
        return new Tensor(data, options);
    }

    public static rand(shape: number[], options?: TensorOptions): Tensor {
        let data: Float32Array = new Float32Array(shape.reduce((prev, curr) => prev * curr));
        for (let i = 0; i < data.length; i++) {
            data[i] = Random.Random();
        }
        return new Tensor(data, options).reshape(shape);
    }

    public static uniform(low: number, high: number, shape: number[], options?: TensorOptions): Tensor {
        let data: Float32Array = new Float32Array(shape.reduce((prev, curr) => prev * curr));
        for (let i = 0; i < data.length; i++) {
            data[i] = Random.RandomRange(low, high);
        }
        // return new Tensor(data, options).reshape(shape);
        const device = options.device ? options.device : Device.CPU;
        const tb = Backend.CreateFromFloat32Array(device, data, shape);
        return new Tensor(tb, options);
    }

    public is_contiguous(): boolean {
        return this.data.is_contiguous();
    }

    public contiguous(): Tensor {
        return new Tensor(MovementOp.contiguous(this.data), {device: this.device, requires_grad: this.requires_grad});
    }
    
    // Movement ops
    public get T(): Tensor { return this.permute(); }
    public expand(shape: number[]): Tensor { return new Operations.Expand().forward(this, shape); }

    private static equalArrays(first: number[], second: number[]): boolean {
        if (first.length != second.length) return false;
        for (let i = 0; i < first.length; i++) {
            if (first[i] !== second[i]) return false;
        }
        return true;
    }

    public broadcast(other: Tensor): [Tensor, Tensor] {
        const [x,y] = [this, other];
        if (Tensor.equalArrays(x.shape, y.shape)) return [x, y];

        // Calculate the final shape after broadcasting
        let finalShape: number[] = [];
        let maxLength = Math.max(x.shape.length, y.shape.length);
        for (let i = 0; i < maxLength; i++) {
            finalShape.push(Math.max(x.shape[x.shape.length - i - 1] || 1, y.shape[y.shape.length - i - 1] || 1));
        }
        finalShape = finalShape.reverse(); // reverse because we filled the array from the end

        // return [
        //     new Tensor(x.expand(finalShape), {device: this.device, requires_grad: this.requires_grad}),
        //     new Tensor(y.expand(finalShape), {device: this.device, requires_grad: this.requires_grad})
        // ]
        return [
            x.expand(finalShape),
            y.expand(finalShape)
        ]
    }

    public slice(indices: (number | null | number[])[]): Tensor {
        if (indices.length != this.shape.length) throw Error(`Indices [${indices}] must match tensor shape ${this.shape}`);

        let offset = this.offset;
        let newShape: number[] = [];
        let newStrides: number[] = [];
        for (let i = 0; i < indices.length; i++) {
            const index = indices[i];

            if (Array.isArray(index)) { // Handle slices
                const [start, stop] = index;
                if (start < 0 || start >= this.shape[i] || stop < 0 || stop > this.shape[i]) {
                    throw Error(`Slice ${start}:${stop} out of bounds for axis ${i} with size ${this.shape[i]}`);
                }
                offset += start * this.strides[i];
                newShape.push(stop - start);
                newStrides.push(this.strides[i]);
            } else if (index !== null) { // Handle integer indices
                if (index < 0 || index >= this.shape[i]) {
                    throw Error(`Index ${index} out of bounds for axis ${i} with size ${this.shape[i]}`);
                }
                offset += index * this.strides[i];
            } else { // Handle null (ellipsis) indices
                newShape.push(this.shape[i]);
                newStrides.push(this.strides[i]);
            }
        }
        const tb = Backend.CreateFromDataShapeAndStrides(this.data, newShape, newStrides, offset);
        return new Tensor(tb, {device: this.device, requires_grad: this.requires_grad});
    }

    public split(split_sizes: number | number[], dim: null | number = null): Tensor[] {
        if (Array.isArray(split_sizes)) throw Error("Split split_sizes as array not supported");
        if (dim !== null) throw Error("Split dim not supported");
    
        const chunkSize = split_sizes;
        const lastDim = this.shape[this.shape.length - 1];
        if (lastDim % chunkSize !== 0) {
            throw new Error('Invalid chunk size, not evenly divisible into last tensor dimension');
        }
    
        const numChunks = lastDim / chunkSize;
        const out: Tensor[] = [];
    
        let start = 0;
        for (let i = 0; i < numChunks; i++) {
            let end = start + chunkSize;
    
            const sliceIndices = this.shape.map((dimSize, idx) => idx === this.shape.length - 1 ? [start, end] : [0, dimSize]);
            const chunk = this.slice(sliceIndices);
            out.push(chunk);
    
            start = end;  // Update the start index for the next iteration
        }
    
        return out;
    }

    // UnaryOps
    public abs(): Tensor {
        return this.relu().add(this.mul(-1).relu())
    }

    public log(): Tensor {
        return new Operations.Log().forward(this);
    }

    public masked_fill(mask: Tensor, value: number): Tensor {
        const [mb, maskb] = this.broadcast(mask);

        const fillTensor = Tensor.full(mb.shape, value, {device: this.device, requires_grad: this.requires_grad});
        const filled = Tensor.where(maskb, fillTensor, mb);
        return filled;
    }

    public softmax(dim: number): Tensor {
        return this.exp().div(this.exp().sum(dim, true));
    }

    private static _tri(r: number, c: number, k: number = 0, options?: TensorOptions): Tensor {
        let a = Tensor.arange(0, r, 1, options).unsqueeze(1).expand([r, c]);
        let b = Tensor.arange(-k, c - k, 1, options).unsqueeze(0).expand([r, c]);

        return a.lte(b);
    }

    public triu(k: number = 0): Tensor {
        const a = Tensor._tri(this.shape[this.shape.length - 2], this.shape[this.shape.length - 1], k, {device: this.device});
        return Tensor.where(a, this, Tensor.zeros(this.shape, {device: this.device}));
    }

    public tril(k: number = 0): Tensor {
        const a = Tensor._tri(this.shape[this.shape.length - 2], this.shape[this.shape.length - 1], k + 1, {device: this.device});
        return Tensor.where(a, Tensor.zeros(this.shape, {device: this.device}), this);
    }

    // BinaryOps
    public add(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, {device: this.device, requires_grad: this.requires_grad});
        return new Operations.Add().forward(...this.broadcast(otherTensor));
    }

    public sub(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, {device: this.device, requires_grad: this.requires_grad});
        return this.add(otherTensor.mul(-1));
    }

    public mul(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, {device: this.device, requires_grad: this.requires_grad});
        return new Operations.Mul().forward(...this.broadcast(otherTensor));
    }
    
    public div(other: Tensor | number) {
        if (!(other instanceof Tensor)) other = new Tensor(other, {device: this.device, requires_grad: this.requires_grad});
        return this.mul(other.pow(-1));
    }

    public pow(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, {device: this.device, requires_grad: this.requires_grad});
        return new Operations.Pow().forward(...this.broadcast(otherTensor));
    }

    public sqrt(): Tensor {
        return this.pow(0.5);
    }

    public rsqrt(): Tensor {
        return this.pow(-0.5);
    }

    public _matmul(other: Tensor | number): Tensor {
        const [m1, m2] = [this, other instanceof Tensor ? other : Tensor.full(this.shape, other)];
        const x = m1.reshape([...m1.shape.slice(0, m1.shape.length - 1), 1, ...m1.shape.slice(m1.shape.length - 1, m1.shape.length)]);
        let w = m2.reshape([...m2.shape.slice(0, m2.shape.length - 2), 1, ...m2.shape.slice(m2.shape.length - 2, m2.shape.length - 1), ...m2.shape.slice(m2.shape.length - 1, m2.shape.length)]);
        w = w.transpose(-1, -2);

        let r = x.mul(w);
        r = r.sum(-1);

        if (m1.shape.length == 1) {
            r = r.reshape([...r.shape.slice(0, r.shape.length - 3), ...r.shape.slice(r.shape.length - 2, r.shape.length - 1)]);
        }

        return r;
    }

    public matmul(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, {device: this.device, requires_grad: this.requires_grad});
        return new Operations.Matmul().forward(this, otherTensor);
    }

    public sum(axis: number | null = null, keepdims: boolean = false): Tensor {
        return new Operations.Sum().forward(this, axis, keepdims);
    }
        
    public mean(axis: number | null = null, keepdims: boolean = false): Tensor {
        const out = this.sum(axis, keepdims);
        const outShapeLen = out.shape.reduce((p, c) => p * c);
        const thisShapeLen = this.shape.reduce((p, c) => p * c);
        return out.mul(outShapeLen/thisShapeLen);
    }

    public var(axis: number | null = null, keepdims: boolean = false): Tensor {
        const x = this.sub(this.mean(axis, true)).abs().pow(2);
        return x.mean(axis, keepdims);
    }

    public reshape(shape: number | number[]): Tensor {
        return new Operations.Reshape().forward(this, shape);
    }

    public exp(): Tensor {
        return new Operations.Exp().forward(this);
    }

    public relu(): Tensor {
        return new Operations.Relu().forward(this);
    }

    public reciprocal(): Tensor {
        return Tensor.ones(this.data.shape, {device: this.device, requires_grad: this.requires_grad}).div(this);
    }

    public sigmoid(): Tensor {
        return Tensor.ones(this.data.shape, {device: this.device, requires_grad: this.requires_grad}).add((this.mul(-1)).exp()).reciprocal();
    }

    public tanh(): Tensor {
        const one = new Tensor(1, {device: this.device, requires_grad: this.requires_grad});
        const two1 = new Tensor(2, {device: this.device, requires_grad: this.requires_grad});
        const two2 = new Tensor(2, {device: this.device, requires_grad: this.requires_grad});
        return two1.mul((two2.mul(this).sigmoid())).sub(one);
    }

    public permute(axes: number[] | null = null): Tensor {
        return new Operations.Permute().forward(this, axes);
    }

    public transpose(dim0: number, dim1: number): Tensor {
        return new Operations.Transpose().forward(this, dim0, dim1);
    }

    public zero_grad() {
        this.grad = Tensor.zeros(this.data.shape, {device: this.device, requires_grad: this.requires_grad}).data;
    }

    public toString() {
        return `Tensor(data=${this.data}, grad=${this.grad})`;
    }





    // New ops
    public maximum(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, {device: this.device});
        return new Operations.Maximum().forward(...this.broadcast(otherTensor));
    }

    public eq(other: Tensor | number) {
        const otherTensor: Tensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, {device: this.device});
        return new Operations.Equal().forward(...this.broadcast(otherTensor));
    }

    public ne(other: Tensor | number) {
        const one = new Tensor(1, {device: this.device});
        return one.sub(this.eq(other));
    }

    public gte(other: Tensor | number): Tensor { return this.maximum(other).eq(this); }
    public lte(other: Tensor | number): Tensor { return this.maximum(other).eq(other); }

    public gt(other: Tensor | number): Tensor {
        const one = new Tensor([1], {device: this.device});
        return one.sub(this.lte(other));
    }

    public lt(other: Tensor | number): Tensor {
        const one = new Tensor([1], {device: this.device});
        return one.sub(this.gte(other));
    }

    public static where(condition: Tensor, x: Tensor, y: Tensor): Tensor {
        const one = new Tensor([1], {device: x.device, requires_grad: x.requires_grad});
        return condition.mul(x).add(one.sub(condition).mul(y));
    }

    public unsqueeze(dim: number): Tensor {
        if (dim < 0) dim = this.shape.length + dim + 1;
        return this.reshape([...this.shape.slice(0, dim), 1, ...this.shape.slice(dim)]);
    }

    public assign(tensor: Tensor): Tensor {
        this.data = new Tensor(tensor.data.getData(), tensor.options).data;
        this.grad = new Tensor(tensor.grad.getData(), tensor.options).data;
        this.options = Object.assign({}, tensor.options);

        return this;
    }

    public to(device: Device): Tensor {
        this.device = device;
        this.options.device = device;
        return this.assign(this);
    }
}