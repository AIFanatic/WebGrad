import { Module } from "../Module";
import { Matrix } from "../Matrix";
import { Tensor, TensorBufferToMatrix } from "../Tensor";

export class Embedding extends Module {
    public weight: Tensor;
    
    private num_embeddings: number;
    private embedding_dim: number;

    constructor(num_embeddings: number, embedding_dim: number) {
        super();
        this.weight = new Tensor(Matrix.uniform(-1, 1, [num_embeddings, embedding_dim]), {requires_grad: true});

        this.num_embeddings = num_embeddings;
        this.embedding_dim = embedding_dim;
    }

    public getFirsts(v: Tensor | Matrix): number[] {
        const data = v instanceof Tensor ? v.data.getData() : v.getData();
        return [data[0][0][0], data[0][0][1], data[0][0][2]];
    }

    public forward(x: Tensor): Tensor {
        const va = Matrix.arange(0, this.num_embeddings);
        const vb = va.reshape([1,1,this.num_embeddings]);
        const vc = vb.expand([...x.shape, this.num_embeddings]);
        const vocab_counter = vc;
        const a = TensorBufferToMatrix(x.data).unsqueeze(2);
        const b = vocab_counter.equal(a);
        const c = b.expand([...x.shape, this.num_embeddings]);
        const d = c.dot(TensorBufferToMatrix(this.weight.data));
        
        return new Tensor(d);
    }

    public parameters(): Tensor[] {
        return [this.weight];
    }

    public toString(): string {
        return `Embedding(num_embeddings=${this.num_embeddings}, embedding_dim=${this.embedding_dim})`;
    }
}