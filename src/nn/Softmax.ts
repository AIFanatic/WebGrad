import { Module } from "../Module";
import { Tensor, TensorBufferToMatrix } from "../Tensor";

export class Softmax extends Module {
    private dim: number;

    constructor(dim: number) {
        super();
        this.dim = dim;
    }

    public forward(x: Tensor): Tensor {
        // return x.exp().div(x.exp().sum(this.dim, true));

        const e = x.exp();
        const eM = TensorBufferToMatrix(e.data);
        const es = eM.sum(this.dim, true);
        const eT = new Tensor(es, {device: x.device, requires_grad: x.requires_grad});
        return x.exp().div(eT);
        
    }

    public parameters(): Tensor[] {
        return [];
    }

    public toString(): string {
        return `SoftMax(dim=${this.dim})`;
    }
}