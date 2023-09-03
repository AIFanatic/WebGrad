import { Module } from "../Module";
import { Tensor } from "../Tensor";

export class Dropout extends Module {
    private p: number; // probability of an element to be zeroed.

    constructor(p: number = 0.5) {
        super();
        this.p = p;
    }
        
    public forward(x: Tensor): Tensor {
        if (this.p === 0) return x;
        const mask = Tensor.rand(x.shape).gte(this.p).reshape(x.shape);
        return x.mul(mask).mul(new Tensor(1).div(new Tensor(1).sub(this.p)));
    }

    public parameters(): Tensor[] {
        return [];
    }

    public toString(): string {
        return `Dropout(p=${this.p.toFixed(2)})`;
    }
}