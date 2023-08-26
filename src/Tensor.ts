import { Operations } from ".";
import { Matrix } from "./Matrix";
import { Operation } from "./Operations";

export class Tensor {
    public data: Matrix;
    public grad: Matrix;
    public _children: Tensor[];
    public _prev: Set<Tensor>;
    public _op: Operation;

    public id: string;

    public get shape(): number[] {
        return this.data.shape;
    }

    constructor(data: Tensor | Matrix | any[] | number, _children: Tensor[] = [], _op: Operation = null) {
        this.id = "P" + Math.floor(Math.random() * 1000000).toString().padStart(6, "0");

        if (data instanceof Matrix) this.data = data;
        else if (data instanceof Tensor) this.data = data.data.copy();
        else if (Array.isArray(data)) this.data = new Matrix(data);
        else this.data = new Matrix([data]);

        this.grad = Matrix.zeros(this.data.shape);
        this._prev = new Set(_children);
        this._children = _children;
        this._op = _op;
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
    
    public add(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : new Tensor(Matrix.full(this.data.shape, other));
        return new Operations.Add().forward(this, otherTensor);
    }

    public sub(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : new Tensor(Matrix.full(this.data.shape, other));
        return this.add(otherTensor.mul(-1));
    }

    public mul(other: Tensor | number): Tensor {
        const otherTensor: Tensor = other instanceof Tensor ? other : new Tensor(Matrix.full(this.data.shape, other));
        return new Operations.Mul().forward(this, otherTensor);
    }
    
    public div(other: Tensor | number) {
        if (!(other instanceof Tensor)) other = new Tensor(other);
        return this.mul(other.pow(-1));
    }

    public pow(other: Tensor | number): Tensor {
        // if ((other instanceof Tensor)) throw Error("Pow only supports scalars");

        const otherTensor: Tensor = other instanceof Tensor ? other : new Tensor(Matrix.full(this.data.shape, other));

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
        const otherTensor: Tensor = other instanceof Tensor ? other : new Tensor(Matrix.full(this.data.shape, other));
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
        return new Tensor(Matrix.ones(this.data.shape)).div(this);
    }

    public sigmoid(): Tensor {
        return new Tensor(Matrix.ones(this.data.shape)).add((this.mul(-1)).exp()).reciprocal();
    }

    public tanh(): Tensor {
        const two1 = new Tensor(2);
        const two2 = new Tensor(2);
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
        return this.mul(new Tensor(this.data.mul(-1)));
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