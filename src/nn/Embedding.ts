import { Module } from "../Module";
import { Tensor } from "../Tensor";

export class Embedding extends Module {
    public weight: Tensor;
    
    private num_embeddings: number;
    private embedding_dim: number;

    constructor(num_embeddings: number, embedding_dim: number) {
        super();
        this.weight = Tensor.uniform(-1, 1, [num_embeddings, embedding_dim], {requires_grad: true});

        this.num_embeddings = num_embeddings;
        this.embedding_dim = embedding_dim;
    }

    public forward(x: Tensor): Tensor {
        const va = Tensor.arange(0, this.num_embeddings, 1, {device: x.device});
        const vb = va.reshape([1,1,this.num_embeddings]);
        const vc = vb.expand([...x.shape, this.num_embeddings]);
        const vocab_counter = vc;
        const a = x.unsqueeze(2);
        const b = vocab_counter.eq(a);
        const c = b.expand([...x.shape, this.num_embeddings]);
        const d = c.matmul(this.weight);
        
        return new Tensor(d);
    }

    public parameters(): Tensor[] {
        return [this.weight];
    }

    public toString(): string {
        return `Embedding(num_embeddings=${this.num_embeddings}, embedding_dim=${this.embedding_dim})`;
    }
}