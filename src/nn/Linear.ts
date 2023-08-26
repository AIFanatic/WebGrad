import { Module } from "../Module";
import { Matrix } from "../Matrix";
import { Tensor } from "../Tensor";

export class Linear extends Module {
    public weight: Tensor;
    public bias: Tensor;
    
    private in_features: number;
    private out_features: number;

    constructor(in_features: number, out_features: number) {
        super();
        this.weight = new Tensor(Matrix.uniform(-1, 1, [out_features, in_features]));
        this.bias = new Tensor(Matrix.zeros([out_features]));

        this.in_features = in_features;
        this.out_features = out_features;
    }

    public forward(x: Tensor): Tensor {
        const wt = this.weight.permute();
        x = wt.shape.length === 1 ? x.mul(wt) : x.matmul(wt);
        return this.bias ? x.add(this.bias) : x;
    }

    public parameters(): Tensor[] {
        return [this.weight, this.bias];
    }

    public toString(): string {
        return `Linear(in_features=${this.in_features}, out_features=${this.out_features})`;
    }
}