import { Matrix } from "./Matrix";
import { MatrixToTensorBuffer, Tensor, TensorBufferToMatrix } from "./Tensor";
import { BinaryOp } from "./backend/BinaryOps";
import { MovementOp } from "./backend/MovementOps";
import { ReduceOp } from "./backend/ReduceOps";
import { UnaryOp } from "./backend/UnaryOps";

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
        return new Tensor(BinaryOp.add(x.data, y.data), {_children: [x, y], _op: this}); // Fails
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
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

        return new Tensor(BinaryOp.mul(x.data, y.data), {_children: [x, y], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [TensorBufferToMatrix(this.y.data).mul(grad), TensorBufferToMatrix(this.x.data).mul(grad)];
    }
}

export class Pow extends Operation {
    private x: Tensor;
    private y: Tensor;

    public forward(x: Tensor, y: Tensor): Tensor {
        this.x = x;
        this.y = y;
        return new Tensor(BinaryOp.pow(x.data, y.data), {_children: [x], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        const a = TensorBufferToMatrix(this.y.data).sub(1);
        const b = TensorBufferToMatrix(this.x.data).pow(a);
        const c = TensorBufferToMatrix(this.y.data).mul(b);
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
        return new Tensor(x._matmul(y), {_children: [x, y], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [
            TensorBufferToMatrix(new Tensor(grad)._matmul(this.y.T).data),
            TensorBufferToMatrix(this.x.T._matmul(new Tensor(grad)).data),
        ];
    }
}

export class Sum extends Operation {
    private shape: number[];

    public forward(x: Tensor, axis: number | null = null, keepdims: boolean = false): Tensor {
        this.shape = x.shape.slice();
        return new Tensor(ReduceOp.sum(x.data, axis, keepdims), {_children: [x], _op: this}); // Fails
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [
            TensorBufferToMatrix(MovementOp.expand(MatrixToTensorBuffer(0, grad), this.shape)),
            null
        ];
    }
}

export class Exp extends Operation {
    private out: Tensor;
    public forward(x: Tensor): Tensor {
        this.out = new Tensor(UnaryOp.exp(x.data), {_children: [x], _op: this});
        return this.out;
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [
            grad.mul(TensorBufferToMatrix(this.out.data)),
            null
        ];
    }
}

export class Relu extends Operation {
    private out: Tensor;
    public forward(x: Tensor): Tensor {
        this.out = new Tensor(Matrix.maximum(Matrix.zeros(x.shape), TensorBufferToMatrix(x.data)), {_children: [x], _op: this});
        return this.out;
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [Matrix.where(TensorBufferToMatrix(this.out.data).gt(0), grad, Matrix.zeros(this.out.data.shape)), null];
    }
}

export class Expand extends Operation {
    private shape: number[];
    public forward(x: Tensor, shape: number[]): Tensor {
        // console.log(`Expand forward ${x}, ${x.shape}`)
        this.shape = x.shape.slice();
        return new Tensor(MovementOp.expand(x.data, shape), {_children: [x], _op: this});
    }

    public backward(grad1: Matrix): [Matrix, Matrix] {

        let gradTensor = new Tensor(grad1);

        const gradShape = gradTensor.shape.slice();
        const originalShape = this.shape.slice();

        if (originalShape.length > gradShape.length) {
            throw Error(`Gradient shape ${gradShape} needs to be bigger or equal to the input shape ${originalShape}`);
        }


        // console.log(`originalShape ${originalShape} gradShape ${gradShape}`);
        if (originalShape.length === gradShape.length) {
            for (let i = 0; i < gradShape.length; i++) {
                const g = gradShape[i];
                const o = originalShape[i];

                if (g !== o) {
                    if (o !== 1) throw Error(`Broadcasted went wrong? Shape lengths match but non matching dimension is expected to have dim 1 and has dim ${o}`);
                    
                    gradTensor = gradTensor.sum(i);
                }
            }
        }
        else {
            const sumsRequired = gradShape.length - originalShape.length;

            for (let i = 0; i < sumsRequired; i++) {
                gradTensor = gradTensor.sum(0);
            }
        }

        return [
            TensorBufferToMatrix(gradTensor.data),
            null
        ];
    }
}

export class Reshape extends Operation {
    private shape: number | number[];
    public forward(x: Tensor, shape: number | number[]): Tensor {
        this.shape = x.shape;
        return new Tensor(MovementOp.reshape(x.data, shape), {_children: [x], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [
            TensorBufferToMatrix(MovementOp.reshape(MatrixToTensorBuffer(0, grad), this.shape)),
            null
        ];
    }
}

export class Permute extends Operation {
    private axes: number[];
    public forward(x: Tensor, axes: number[]): Tensor {
        this.axes = axes;
        return new Tensor(MovementOp.permute(x.data, axes), {_children: [x], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [
            TensorBufferToMatrix(MovementOp.permute(MatrixToTensorBuffer(0, grad), this.axes)),
            null
        ];
    }
}

export class Transpose extends Operation {
    private dim0: number;
    private dim1: number;
    public forward(x: Tensor, dim0: number, dim1: number): Tensor {
        this.dim0 = dim0;
        this.dim1 = dim1;
        return new Tensor(MovementOp.transpose(x.data, dim0, dim1), {_children: [x], _op: this});
    }

    public backward(grad: Matrix): [Matrix, Matrix] {
        return [
            TensorBufferToMatrix(MovementOp.transpose(MatrixToTensorBuffer(0, grad), this.dim0, this.dim1)),
            null
        ];
    }
}