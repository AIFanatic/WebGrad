import { Module } from "../Module";
import { Tensor } from "../Tensor";

export class Softmax extends Module {
    private dim: number;

    constructor(dim: number) {
        super();
        this.dim = dim;
    }

    public forward(x: Tensor): Tensor {
        return x.exp().div(x.exp().sum(this.dim, true));
    }

    public parameters(): Tensor[] {
        return [];
    }

    public toString(): string {
        return `SoftMax(dim=${this.dim})`;
    }
}