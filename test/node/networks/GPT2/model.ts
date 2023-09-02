import { Module, nn, Random, Tensor } from "../../../../src";
import { Backend } from "../../../../src/backend/Backend";
import { TensorBuffer } from "../../../../src/backend/TensorBuffer";

Random.SetRandomSeed(1337);

// TODO: Handle negative axis and error checking
function concatenateArrays(arrays: Array<any>, axis: number = 0): number[] {
    if (axis < 0) {
        throw new Error(`Invalid axis value ${axis} ${arrays[0].length}`);
    }

    if (axis === 0) {
        // Vertical concatenation (stacking rows)
        return [].concat(...arrays);
    } else {
        // Concatenate along the specified axis
        const result: Array<any> = [];

        for (let i = 0; i < arrays[0].length; i++) {
            const newArrays = arrays.map(arr => arr[i]);
            result.push(concatenateArrays(newArrays, axis - 1));
        }

        return result;
    }
}

function cat(tensors: Tensor[], dim: number | null = 0): Tensor {
    if (dim === null) {
        const flattenData = tensors.map(v => Array.from(v.data.getData())).flat(Infinity);
        return new Tensor(flattenData);
    }
    const data = tensors.map(m => m.data.getData());
    return new Tensor(concatenateArrays(data, dim));
}

function multinomial(tensor: Tensor, num_samples: number, normalized: boolean = false): Tensor {
    const origRank = tensor.shape.length;
    const logits2D = origRank === 1 ? tensor.reshape([1, -1]) : tensor;

    const probabilities = normalized ? logits2D : logits2D.softmax(-1);
    const batchSize = probabilities.shape[0];
    const numEvents = probabilities.shape[1];
    const probVals = probabilities.data.getData().flat(Infinity);
    const resShape = [batchSize, num_samples];
    const resVals = new Float32Array(resShape.reduce((p, c) => p * c));

    for (let b = 0; b < batchSize; ++b) {
        const offset = b * numEvents;
        // The cdf won't include the last event. It will be implicit if no other
        // event happened.
        const cdf = new Float32Array(numEvents - 1);
        cdf[0] = probVals[offset];
        for (let event = 1; event < cdf.length; ++event) {
            cdf[event] = cdf[event - 1] + probVals[offset + event];
        }

        const outOffset = b * num_samples;
        for (let sampleId = 0; sampleId < num_samples; ++sampleId) {
            const r = Random.Random();

            // Assume last event happened by default.
            resVals[outOffset + sampleId] = cdf.length;

            for (let event = 0; event < cdf.length; event++) {
                if (r < cdf[event]) {
                    resVals[outOffset + sampleId] = event;
                    break;
                }
            }
        }
    }

    const tb = Backend.CreateFromFloat32Array(tensor.device, resVals, resShape, TensorBuffer.computeStrides(resShape), 0);
    return new Tensor(tb);
}

export class NewGELU extends Module {
    constructor() {
        super();
    }

    public forward(x: Tensor): Tensor {
        const t1 = new Tensor(0.5);
        const t2 = new Tensor(1.0);
        const t4 = new Tensor(0.044715);

        const a = x.pow(3);
        const b = x.add(t4.mul(a));
        const c = new Tensor(Math.sqrt(2.0 / Math.PI))
        const d = c.mul(b);
        const e = t2.add(d.tanh());
        const f = t1.mul(x).mul(e);
        return f
    }
}

export interface NetworkConfig {
    n_head: number;
    n_embd: number;
    
    attn_pdrop: number;
    resid_pdrop: number;
    embd_pdrop: number;

    vocab_size: number;
    block_size: number;

    n_layer: number;
};

export class CausalSelfAttention extends Module {
    private c_attn: nn.Linear;
    private c_proj: nn.Linear;

    private attn_dropout: nn.Dropout;
    private resid_dropout: nn.Dropout;

    private n_head: number;
    private n_embd: number;

    private bias: Tensor;

    constructor(config: NetworkConfig) {
        super();
        if (config.n_embd % config.n_head !== 0) throw Error(`Wrong n_head(${config.n_head}) and n_embd(${config.n_embd})`);

        // key, query, value projections for all heads, but in a batch
        this.c_attn = new nn.Linear(config.n_embd, 3 * config.n_embd);

        // output projection
        this.c_proj = new nn.Linear(config.n_embd, config.n_embd);

        // regularization
        this.attn_dropout = new nn.Dropout(config.attn_pdrop);
        this.resid_dropout = new nn.Dropout(config.resid_pdrop);

        // causal mask to ensure that attention is only applied to the left in the input sequence
        this.bias = Tensor.ones([config.block_size, config.block_size]).tril().reshape([1,1,config.block_size,config.block_size]);

        this.n_head = config.n_head;
        this.n_embd = config.n_embd;
    }

    public forward(x: Tensor): Tensor {
        const [B,T,C] = x.shape; // batch size, sequence length, embedding dimensionality (n_embd)

        // calculate query, key, values for all heads in batch and move head forward to be the batch dim

        const c_attn_f = this.c_attn.forward(x);
        let [q, k, v] = c_attn_f.split(this.n_embd);

        k = k.reshape([B, T, this.n_head, Math.floor(C / this.n_head)]).transpose(1,2);
        q = q.reshape([B, T, this.n_head, Math.floor(C / this.n_head)]).transpose(1,2);
        v = v.reshape([B, T, this.n_head, Math.floor(C / this.n_head)]).transpose(1,2);

        // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        const t1 = new Tensor(1);
        let att = q.matmul(k.transpose(-2, -1)).mul(t1.div( Math.sqrt(k.shape[k.shape.length-1]) ))
        
        const biasSliced = this.bias.slice([null, null, [0, T], [0, T]]);
        let maskedAttn = att.masked_fill(biasSliced.eq(0), 0);

        att = this.attn_dropout.forward(maskedAttn.softmax(-1));

        let y = att.matmul(v)
        y = y.transpose(1,2).contiguous().reshape([B,T,C]) // re-assemble all head outputs side by side

        // output projection
        y = this.resid_dropout.forward(this.c_proj.forward(y));

        return y;
    }
}

export class MLP extends Module {
    public c_fc: nn.Linear;
    public c_proj: nn.Linear;
    public act: NewGELU;
    public dropout: nn.Dropout;

    constructor(config: NetworkConfig) {
        super();
        
        this.c_fc = new nn.Linear(config.n_embd, 4 * config.n_embd),
        this.c_proj = new nn.Linear(4 * config.n_embd, config.n_embd),
        this.act = new NewGELU(),
        this.dropout = new nn.Dropout(config.resid_pdrop)
    }
}

export class Block extends Module {
    private ln_1: nn.LayerNorm;
    private ln_2: nn.LayerNorm;

    private attn: CausalSelfAttention;
    private mlp: MLP;
    private mlpf: (x: Tensor) => Tensor;

    constructor(config: NetworkConfig) {
        super();
        this.ln_1 = new nn.LayerNorm([config.n_embd]);
        this.attn = new CausalSelfAttention(config);
        this.ln_2 = new nn.LayerNorm([config.n_embd]);

        this.mlp = new MLP(config);
        this.mlpf = (x: Tensor): Tensor => {
            const c_fc_f = this.mlp.c_fc.forward(x);
            const act_f = this.mlp.act.forward(c_fc_f);
            const c_proj_f = this.mlp.c_proj.forward(act_f);
            const drop_f = this.mlp.dropout.forward(c_proj_f);
            return drop_f;
        }
    }

    public forward(x: Tensor): Tensor {
        x = x.add(this.attn.forward(this.ln_1.forward(x)));
        x = x.add(this.mlpf(this.ln_2.forward(x)));
        return x;
    }
}

export class Transformer extends Module {
    public wte: nn.Embedding;
    public wpe: nn.Embedding;
    public drop: nn.Dropout;
    public h: Block[];
    public ln_f: nn.LayerNorm;

    constructor(config: NetworkConfig) {
        super();

        this.wte = new nn.Embedding(config.vocab_size, config.n_embd);
        this.wpe = new  nn.Embedding(config.block_size, config.n_embd);
        this.drop = new nn.Dropout(config.embd_pdrop);
        this.h = [];
        this.ln_f = new nn.LayerNorm([config.n_embd]);

        for (let i = 0; i < config.n_layer; i++) {
            this.h.push(new Block(config));
        }
    }
}

function numel(tensor: Tensor): Tensor {
    return new Tensor(tensor.shape.reduce((p, c) => p * c));
}

export class GPT extends Module {
    private block_size: number;
    
    public transformer: Transformer;
    private lm_head: nn.Linear;

    constructor(config: NetworkConfig) {
        super();
        if (config.vocab_size === undefined) throw Error("config.vocab_size cannot be null");
        if (config.block_size === undefined) throw Error("config.block_size cannot be null");

        this.block_size = config.block_size;

        this.transformer = new Transformer(config);

        // report number of parameters (note we don't count the decoder parameters in lm_head)
        this.lm_head = new nn.Linear(config.n_embd, config.vocab_size); // bias = False

        let n_params = 0;

        for (let p of this.parameters()) {
            n_params += numel(p).data.get(0);
        }
    }

    public forward(idx: Tensor, targets=null): Tensor {
        const [b, t] = idx.shape;
        const pos = Tensor.arange(0, t).unsqueeze(0);

        const tok_emb = this.transformer.wte.forward(idx);
        const pos_emb = this.transformer.wpe.forward(pos);

        let x = this.transformer.drop.forward(tok_emb.add(pos_emb));

        for (let i = 0; i < this.transformer.h.length; i++) {
            const block = this.transformer.h[i];
            x = block.forward(x);
        }

        x = this.transformer.ln_f.forward(x);
        const logits = this.lm_head.forward(x);

        const loss = new Tensor(0);
        if (targets) {
            throw Error("cross_entropy not implemented");
        }

        return logits;
    }

    public generate(idx: Tensor, max_new_tokens: number, temperature: number = 1.0, do_sample: boolean = false, top_k: number | null = null) {
        for (let i = 0; i < max_new_tokens; i++) {
            const idx_cond = idx.shape[1] < this.block_size ? idx : new Tensor(idx.data);

            let logits = this.forward(idx_cond);

            // pluck the logits at the final step and scale by desired temperature
            const logits_temp = logits.slice([null, [logits.shape[1]-1, logits.shape[1]], null]).contiguous(); // TODO: Why contiguous
            logits = logits_temp.div(temperature).reshape([1,65]);
            
            const probs = logits.softmax(-1);

            let idx_next: Tensor

            if (do_sample) {
                idx_next = multinomial(probs, 1, true);
            }
            else {
                // _, idx_next = torch.topk(probs, k=1, dim=-1)
                idx_next = new Tensor(0);
            }

            // # append sampled index to the running sequence and continue
            idx = cat([idx, idx_next], 1);

        }
        return idx;
    }
}

