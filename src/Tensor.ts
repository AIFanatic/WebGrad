import { Operations, Random } from ".";
import { Matrix } from "./Matrix";
import { Operation } from "./Operations";
import { Backend, Device } from "./backend/Backend";
import { TensorBuffer } from "./backend/TensorBuffer";



// TODO: Temp
export function TensorBufferToMatrix(tensorBuffer: TensorBuffer): Matrix {
    // return new Matrix(tensorBuffer.getData(), tensorBuffer.shape, tensorBuffer.strides, tensorBuffer.offset);
    return new Matrix(tensorBuffer.getData());
}

export function MatrixToTensorBuffer(device: Device, matrix: Matrix): TensorBuffer {
    const tf = Backend.CreateFromFloat32Array(device, matrix.data);
    const tfShaped = Backend.CreateFromDataShapeAndStrides(tf, matrix.shape, matrix.strides, matrix.offset);
    return tfShaped;
}




export type TensorDataTypes = Matrix | Tensor | TensorBuffer | Array<any> | Float32Array | number;

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
    public data: TensorBuffer;
    public grad: Matrix;
    public _children: Tensor[];
    public _prev: Set<Tensor>;
    public _op: Operation;
    public device: Device;
    public requires_grad: boolean;

    public id: string;

    public get shape(): number[] {
        return this.data.shape;
    }

    constructor(data: TensorDataTypes, options?: TensorOptions) {
        this.id = "P" + Math.floor(Math.random() * 1000000).toString().padStart(6, "0");

        const _options: TensorOptions = Object.assign({}, DefaultTensorOptions, options);

        if (_options._children.length !== 0) {
            _options.device = _options._children[0].device;
            _options.requires_grad = _options._children[0].requires_grad;
        }

        if (data instanceof Tensor) {
            this.data = data.data;
            _options.device = options && options.device ? options.device : data.device;
        }
        else if (data instanceof TensorBuffer) {
            this.data = data;
            _options.device = options && options.device ? options.device : data.device;
        }
        else if (data instanceof Array) this.data = Backend.CreateFromArray(_options.device, data);
        else if (data instanceof Float32Array) this.data = Backend.CreateFromFloat32Array(_options.device, data);
        // TODO: Temp
        else if (data instanceof Matrix) {
            this.data = MatrixToTensorBuffer(_options.device, data);
        }
        else if (!isNaN(data)) this.data = Backend.CreateFromNumber(_options.device, data);

        this.grad = _options.requires_grad ? Matrix.zeros(this.shape) : null;
        this.device = _options.device;
        this.requires_grad = _options.requires_grad;
        this._op = _options._op;
        this._prev = new Set(_options._children);
        this._children = _options._children;
    }

    public backward() {
        let topo: Tensor[] = [];
        let visited = new Set<Tensor>();

        const thisShape = this.shape.slice();
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

        this.grad = Matrix.ones(this.data.shape);

        for (let v of topo.reverse()) {
            if (v._op === null) continue;
            const grads = v._op.backward(v.grad);
            if (grads) {
                for (let i = 0; i < grads.length; i++) {
                    if (grads[i] !== null) {
                        if (v._children[i].grad) {
                            v._children[i].grad = v._children[i].grad.add(grads[i]); // accumulate gradients
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
        if (value === 0) return new Tensor(new Float32Array(desiredElements), options).reshape(shape);
        return new Tensor(new Float32Array(desiredElements).fill(value), options).reshape(shape);
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
        return new Tensor(data, options).reshape(shape);
    }
    
    public add(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, {device: this.device, requires_grad: this.requires_grad});
        return new Operations.Add().forward(this, otherTensor);
    }

    public sub(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, {device: this.device, requires_grad: this.requires_grad});
        return this.add(otherTensor.mul(-1));
    }

    public mul(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, {device: this.device, requires_grad: this.requires_grad});
        return new Operations.Mul().forward(this, otherTensor);
    }
    
    public div(other: Tensor | number) {
        if (!(other instanceof Tensor)) other = new Tensor(other);
        return this.mul(other.pow(-1));
    }

    public pow(other: Tensor | number): Tensor {
        // if ((other instanceof Tensor)) throw Error("Pow only supports scalars");

        const otherTensor: Tensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, {device: this.device, requires_grad: this.requires_grad});

        // const otherTensor: Tensor = new Tensor(Matrix.full(this.data.shape, other));
        return new Operations.Pow().forward(this, otherTensor);
    }

    public sqrt(): Tensor {
        return this.pow(0.5);
    }

    public rsqrt(): Tensor {
        return this.pow(-0.5);
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
        const two1 = new Tensor(2, {device: this.device, requires_grad: this.requires_grad});
        const two2 = new Tensor(2, {device: this.device, requires_grad: this.requires_grad});
        return two1.mul((two2.mul(this).sigmoid())).sub(1);
    }

    public permute(axes: number[] | null = null): Tensor {
        return new Operations.Permute().forward(this, axes);
    }

    public transpose(dim0: number, dim1: number): Tensor {
        return new Operations.Tranpose().forward(this, dim0, dim1);
    }

    public zero_grad() {
        this.grad = Matrix.zeros(this.data.shape);
    }

    public __neg__() {
        return this.mul(this.mul(-1));
    }

    public __radd__(other: Tensor) {
        return this.add(other);
    }

    public __rsub__(other: Tensor | number) {
        const o = other instanceof Tensor ? other : new Tensor(other);
        return o.add(this.mul(-1));
    }

    public __rmul__(other: Tensor | number) {
        return this.mul(other);
    }

    public __rtruediv__(other: Tensor) {
        return other.mul(this.pow(-1));
    }

    public toString() {
        return `Tensor(data=${this.data}, grad=${this.grad})`;
    }

    public assign(other: Tensor): Tensor {
        this.data = other.data.copy();
        return this;
    }

    public copy(): Tensor {
        return new Tensor(this.data.copy());
    }
}