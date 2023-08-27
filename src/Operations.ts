import { Matrix } from "./Matrix";
import { Tensor } from "./Tensor";

export class Operation {
    forward(...args): Tensor { throw Error("Not implemented") };
    backward(grad: Matrix): [Matrix, Matrix] { throw Error("Not implemented") };
}

function get_repeat_axis(left_shape: number[], right_shape: number[]): [number, number] {
    const len1 = left_shape.length;
    const len2 = right_shape.length;
    const left_not_repeat = len1 - len2;
    const repeat_axis = Matrix.arange(0, Math.abs(len1 - len2), 1);

    if (repeat_axis.data.length > 1) throw Error("Multiple axis repeated, sum wont work");
    
    return [left_not_repeat, repeat_axis.data[0]];
}

export class Add extends Operation {
    private x: Tensor;
    private y: Tensor;
    public forward(x: Tensor, y: Tensor): Tensor {
        this.x = x;
        this.y = y;
        return new Tensor(x.data.add(y.data), {_children: [x, y], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        // return [grad, grad];

        const [self_not_repeat, self_repeat_axis] = get_repeat_axis(this.x.grad.shape, grad.shape);
        const [other_not_repeat, other_repeat_axis] = get_repeat_axis(this.y.grad.shape, grad.shape);
    
        return [
            self_not_repeat < 0 ? grad.sum(self_repeat_axis) : grad,
            other_not_repeat < 0 ? grad.sum(other_repeat_axis) : grad
        ]
    }
}

export class Mul extends Operation {
    private x: Tensor;
    private y: Tensor;

    public forward(x: Tensor, y: Tensor): Tensor {
        this.x = x;
        this.y = y;

        return new Tensor(x.data.mul(y.data), {_children: [x, y], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [this.y.data.mul(grad), this.x.data.mul(grad)];
    }
}

export class Pow extends Operation {
    private x: Tensor;
    private y: Tensor;

    public forward(x: Tensor, y: Tensor): Tensor {
        this.x = x;
        this.y = y;
        return new Tensor(x.data.pow(y.data), {_children: [x], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        const a = this.y.data.sub(1);
        const b = this.x.data.pow(a);
        const c = this.y.data.mul(b);
        const d = c.mul(grad);
        
        return [d, null];
    }
}

export class Matmul extends Operation {
    private x: Tensor;
    private y: Tensor;

    public forward(x: Tensor, y: Tensor): Tensor {
        this.x = x;
        this.y = y;
        return new Tensor(Matrix.dot(x.data, y.data), {_children: [x, y], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [Matrix.dot(grad, this.y.data.T), Matrix.dot(this.x.data.T, grad)];
    }
}

export class Sum extends Operation {
    private shape: number[];

    public forward(x: Tensor, axis: number | null = null, keepdims: boolean = false): Tensor {
        this.shape = x.shape.slice();
        return new Tensor(Matrix.sum(x.data, axis, keepdims), {_children: [x], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [grad.expand(this.shape), null];
    }
}

export class Reshape extends Operation {
    private shape: number | number[];
    public forward(x: Tensor, shape: number | number[]): Tensor {
        this.shape = x.shape;
        return new Tensor(Matrix.reshape(x.data, shape), {_children: [x], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [grad.reshape(this.shape), null];
    }
}

export class Exp extends Operation {
    private out: Tensor;
    public forward(x: Tensor): Tensor {
        this.out = new Tensor(Matrix.exp(x.data), {_children: [x], _op: this});
        return this.out;
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [grad.mul(this.out.data), null];
    }
}

export class Relu extends Operation {
    private out: Tensor;
    public forward(x: Tensor): Tensor {
        this.out = new Tensor(Matrix.maximum(Matrix.zeros(x.shape), x.data), {_children: [x], _op: this});
        return this.out;
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [Matrix.where(this.out.data.gt(0), grad, Matrix.zeros(this.out.data.shape)), null];
    }
}

export class Permute extends Operation {
    private axes: number[];
    public forward(x: Tensor, axes: number[]): Tensor {
        this.axes = axes;
        return new Tensor(Matrix.permute(x.data, axes), {_children: [x], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [Matrix.permute(grad, this.axes), null];
    }
}

export class Tranpose extends Operation {
    private dim0: number;
    private dim1: number;
    public forward(x: Tensor, dim0: number, dim1: number): Tensor {
        this.dim0 = dim0;
        this.dim1 = dim1;
        return new Tensor(Matrix.transpose(x.data, dim0, dim1), {_children: [x], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [Matrix.transpose(grad, this.dim0, this.dim1), null]
    }
}