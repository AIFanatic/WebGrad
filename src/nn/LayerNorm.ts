import { Module } from "../Module";
import { Matrix } from "../Matrix";
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
        this.weight = new Tensor(Matrix.ones(normalized_shape), {requires_grad: true}); // learnable weights
        this.bias = elementwise_affine ? new Tensor(Matrix.zeros(normalized_shape), {requires_grad: true}) : null; // learnable bias
    }

    private numel(tensor: Tensor): number {
        // return new Tensor(Matrix.prod(new Matrix(tensor.shape)));
        return tensor.shape.reduce((acc, val) => acc * val, 1);
    }

    // def layernorm(self, axis=-1, eps:float=1e-5) -> Tensor:
        // y = (self - self.mean(axis, keepdim=True))
        // return y.mul((y*y).mean(axis, keepdim=True).add(eps).rsqrt())
    
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