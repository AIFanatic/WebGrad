import { Module } from "../Module";
import { Tensor } from "../Tensor";

export class LayerNorm extends Module {
    private eps: number;
    private weight: Tensor;
    private bias: Tensor | null;
    private elementwise_affine: boolean;

    constructor(normalized_shape: number[], eps: number = 1e-5, elementwise_affine: boolean = true) {
        super();
        this.eps = eps;
        this.elementwise_affine = elementwise_affine;
        this.weight = Tensor.ones(normalized_shape, {requires_grad: true}); // learnable weights
        this.bias = elementwise_affine ? Tensor.zeros(normalized_shape, {requires_grad: true}) : null; // learnable bias
    }

    private layernorm(self: Tensor, axis: number = -1, eps: number = 1e-5): Tensor {
        const a = self.mean(axis, true);
        const y = self.sub(a);
        const b = y.mul(y);
        const c = b.mean(axis, true);
        const d = c.add(eps);
        const e = d.rsqrt();
        const f = y.mul(e);
        return f;

    }

    public forward(x: Tensor): Tensor {
        const axis = -1;

        x = this.layernorm(x, axis, this.eps);
        if (!this.elementwise_affine) return x;
        return x.mul(this.weight).add(this.bias);
    }

    public parameters(): Tensor[] {
        return [this.weight, this.bias];
    }

    public toString(): string {
        return `LayerNorm(eps=${this.eps})`;
    }
}