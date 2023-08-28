import { Matrix } from "../Matrix";
import { Module } from "../Module";
import { Tensor, TensorBufferToMatrix } from "../Tensor";

// TODO: Enable dropout when training
export class Dropout extends Module {
    private pScalar: number;
    private p: Tensor;

    // p = probability of an element to be zeroed.
    constructor(p: number = 0.5) {
        super();
        this.pScalar = p;
        this.p = new Tensor(p);
    }
        
    public forward(x: Tensor): Tensor {
        if (this.pScalar === 0) return x;
        const mask = new Tensor(Matrix.rand(x.shape).gte(TensorBufferToMatrix(this.p.data).reshape(x.shape)));
        return x.mul(mask).mul(new Tensor(1).div(new Tensor(1).sub(this.p)));
    }

    public parameters(): Tensor[] {
        return [];
    }

    public toString(): string {
        return `Dropout(p=${this.pScalar.toFixed(2)})`;
    }
}