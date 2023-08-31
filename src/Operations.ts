import { Tensor } from "./Tensor";
import { BinaryOp } from "./backend/BinaryOps";
import { MovementOp } from "./backend/MovementOps";
import { ReduceOp } from "./backend/ReduceOps";
import { TensorBuffer } from "./backend/TensorBuffer";
import { UnaryOp } from "./backend/UnaryOps";

export class Operation {
    forward(...args): Tensor { throw Error("Not implemented") };
    backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] { throw Error("Not implemented") };
}

export class Add extends Operation {
    private x: Tensor;
    private y: Tensor;
    public forward(x: Tensor, y: Tensor): Tensor {
        this.x = x;
        this.y = y;
        return new Tensor(BinaryOp.add(x.data, y.data), {_children: [x, y], _op: this}); // Fails
    }

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        return [grad, grad];
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

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        return [
            BinaryOp.mul(this.y.data, grad),
            BinaryOp.mul(this.x.data, grad),
        ];
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

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        const a = BinaryOp.sub(this.y.data, new Tensor([1]).data);
        const b = BinaryOp.pow(this.x.data, a);
        const c = BinaryOp.mul(this.y.data, b);
        const d = BinaryOp.mul(c, grad);
        
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

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        return [
            new Tensor(grad)._matmul(this.y.T).data,
            this.x.T._matmul(new Tensor(grad)).data
        ];
    }
}

export class Sum extends Operation {
    private shape: number[];

    public forward(x: Tensor, axis: number | null = null, keepdims: boolean = false): Tensor {
        this.shape = x.shape.slice();
        return new Tensor(ReduceOp.sum(x.data, axis, keepdims), {_children: [x], _op: this}); // Fails
    }

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        return [
            MovementOp.expand(grad, this.shape),
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

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        return [
            BinaryOp.mul(grad, this.out.data),
            null
        ];
    }
}

export class Relu extends Operation {
    private out: Tensor;
    public forward(x: Tensor): Tensor {
        this.out = new Tensor(Tensor.zeros(x.shape).maximum(x), {_children: [x], _op: this});
        return this.out;
    }

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        return [
            Tensor.where(this.out.gt(0), new Tensor(grad), Tensor.zeros(this.out.shape)).data,
            null
        ];
    }
}

export class Expand extends Operation {
    private shape: number[];
    public forward(x: Tensor, shape: number[]): Tensor {
        // console.log(`Expand forward ${x}, ${x.shape}`)
        this.shape = x.shape.slice();
        return new Tensor(MovementOp.expand(x.data, shape), {_children: [x], _op: this});
    }

    public backward(grad1: TensorBuffer): [TensorBuffer, TensorBuffer] {

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
            gradTensor.data,
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

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        return [
            MovementOp.reshape(grad, this.shape),
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

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        return [
            MovementOp.permute(grad, this.axes),
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

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        return [
            MovementOp.transpose(grad, this.dim0, this.dim1),
            null
        ];
    }
}

export class Maximum extends Operation {
    private x: Tensor;
    private y: Tensor;
    public forward(x: Tensor, y: Tensor): Tensor {
        this.x = x;
        this.y = y;
        return new Tensor(BinaryOp.maximum(x.data, y.data), {_children: [x, y], _op: this});
    }

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        return [
            this.x.gte(this.y).data,
            this.y.gt(this.x).data
        ];
    }
}

export class Equal extends Operation {
    public forward(x: Tensor, y: Tensor): Tensor {
        return new Tensor(BinaryOp.equal(x.data, y.data), {_children: [x, y], _op: this});
    }

    public backward(grad: TensorBuffer): [TensorBuffer, TensorBuffer] {
        return [grad, null];
    }
}