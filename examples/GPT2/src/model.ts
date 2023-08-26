import { Tensor, Matrix, nn, Module, Random } from "micrograd-ts";

Random.SetRandomSeed(1337);

function arrayCompare(a: number[], b: number[]) {
    const af = a.flat(Infinity);
    const bf = b.flat(Infinity);

    if (af.length != bf.length) throw Error(`Length of a ${af.length} != length of b ${bf.length}`);

    const EPS = 10e-3;
    for (let i = 0; i < af.length; i++) {
        if (Math.abs(af[i] - bf[i]) > EPS) {
            throw Error(`[${i}] Data for a ${af[i]} doesn't match data for b ${bf[i]}`);
        }
    }
}

function softmax(att: Tensor, dim: number): Tensor {
    const softmax = new nn.Softmax(dim);
    return softmax.forward(att);
}

export class NewGELU extends Module {
    constructor() {
        super();
    }

    public forward(x: Tensor): Tensor {
        const t1 = new Tensor(0.5);
        const t2 = new Tensor(1.0);
        const t4 = new Tensor(0.044715);

        // return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
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
        this.bias = new Tensor(Matrix.tril(Matrix.ones([config.block_size, config.block_size]))).reshape([1,1,config.block_size,config.block_size]);
        // this.bias = Matrix.ones([config.block_size, config.block_size]).tril().reshape([1,1,config.block_size,config.block_size]);

        this.n_head = config.n_head;
        this.n_embd = config.n_embd;
    }

    private contiguous(tensor: Tensor): Tensor {
        return new Tensor(tensor.data.getData());
    }

    public forward(x: Tensor): Tensor {
        const [B,T,C] = x.shape; // batch size, sequence length, embedding dimensionality (n_embd)

        // console.log(`c_attn x ${x.shape}, ${x.sum()}`);

        // calculate query, key, values for all heads in batch and move head forward to be the batch dim
        // console.log(`c_attn.weight ${this.c_attn.weight.shape}, ${this.c_attn.weight.sum()}`);
        // console.log(`c_attn.bias ${this.c_attn.bias.shape}, ${this.c_attn.bias.sum()}`);

        const c_attn_f = this.c_attn.forward(x);
        // console.log(`c_attn_f ${c_attn_f.shape}, ${c_attn_f.sum()}`);

        // q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        // let [q, k ,v]  = this.c_attn.call(x).split(this.n_embd, dim=2)
        // let q = new Tensor(c_attn_f.data.split([this.n_embd], 2));
        // console.log(`this.n_embd ${this.n_embd}`);
        const chunks = c_attn_f.data.split(this.n_embd);
        // console.log("chunks", chunks);
        // throw Error("HGERE")
        let [q, k, v] = [new Tensor(chunks[0]), new Tensor(chunks[1]), new Tensor(chunks[2])];
        // console.log(`q reshape ${q.data.data} ${q.data.shape} ${q.data.strides}`);
        // console.log(`k reshape ${k.data.data} ${k.data.shape} ${k.data.strides}`);
        // console.log(B, T, this.n_head, Math.floor(C / this.n_head))
        // throw Error("ewfewfwef")

        k = k.reshape([B, T, this.n_head, Math.floor(C / this.n_head)]).transpose(1,2); // TODO: Why no transpose(1,2)?
        q = q.reshape([B, T, this.n_head, Math.floor(C / this.n_head)]).transpose(1,2); // Why no transpose(1,2)?
        v = v.reshape([B, T, this.n_head, Math.floor(C / this.n_head)]).transpose(1,2); // Why no transpose(1,2)?

        // console.log(`v reshape ${v.data}`);
        
        k = new Tensor(k.data)
        q = new Tensor(q.data)
        v = new Tensor(v.data)

        // console.log(`k ${k.shape} ${k.sum()}`);
        // console.log(`q ${q.shape} ${q.sum()}`);
        // console.log(`v ${v.shape} ${v.sum()}`);

        
        // console.log("qm", qm.shape, qm.data.strides)
        // console.log(`qm ${qm}`)
        // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        const t1 = new Tensor(1);
        let att = q.matmul(k.transpose(-2, -1)).mul(t1.div( Math.sqrt(k.shape[k.shape.length-1]) ))
        // console.log(`att ${att.shape} ${att.sum()} ${t1.div( Math.sqrt(k.shape[k.shape.length-1]))} ${k.shape[k.shape.length-1]}`);
        // console.log(`att ${att.shape} ${att.sum()}`);
        // console.log(`bias ${this.bias.data}`);

        // console.log("T", T)
        
        
        // att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        const biasSliced = Matrix.slice(this.bias.data, [null, null, [0, T], [0, T]]);
        // console.log(`self.bias[:,:,:T,:T] ${biasSliced}`)
        // throw Error("wefewfwe")
        // console.log(`biasSliced ${biasSliced}`)
        let maskedAttn = att.data.masked_fill(biasSliced.equal(0), 0);
        // console.log(`maskedAttn ${maskedAttn.shape} ${maskedAttn.sum()}`)

        // throw Error("ATT")
        att = new Tensor(maskedAttn.getData());
        // att = maskedAttn;
        // console.log(`att mask ${att.shape} ${att.sum()}`)
        // throw Error("ATT")
        
        att = softmax(att, -1);
        // console.log(`att softmax ${att.shape} ${att.sum()}`)


        att = this.attn_dropout.forward(att);
        // console.log(`att dropout ${att.shape} ${att.sum()}`)
        // console.log(`v ${v.shape} ${v.sum()}`)

        // let y = new Tensor(att.dot(v)); // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        let y = att.matmul(v)
        // console.log(`y ${y.shape} ${y.sum()}`)
        
        y = this.contiguous(y.transpose(1,2)).reshape([B,T,C]) // re-assemble all head outputs side by side
        // y = y.transpose(1,2)
        // y = new Tensor(y.data.getData()).reshape([B,T,C])
        // console.log(`y tranpose ${y.shape} ${y.data.strides} ${y.sum()}`)


        // output projection
        // console.log(`y ${y.data}`)
        // const c_proj_f = this.c_proj.forward(y);
        // console.log(`c_proj_f ${c_proj_f.shape} ${c_proj_f.sum()}`)

        y = this.resid_dropout.forward(this.c_proj.forward(y));
        // console.log(`out proj ${y.shape} ${y.sum()}`)

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
            // console.log("---------MLP------");
            // console.log(`x ${x.shape} ${x.sum()}`);
            const c_fc_f = this.mlp.c_fc.forward(x);
            // console.log(`c_fc_f ${c_fc_f.shape} ${c_fc_f.sum()}`);
            const act_f = this.mlp.act.forward(c_fc_f);
            // console.log(`act_f ${act_f.shape} ${act_f.sum()}`);
            const c_proj_f = this.mlp.c_proj.forward(act_f);
            // console.log(`c_proj_f ${c_proj_f.shape} ${c_proj_f.sum()}`);
            const drop_f = this.mlp.dropout.forward(c_proj_f);
            // console.log(`drop_f ${drop_f.shape} ${drop_f.sum()}`);
            return drop_f;
        }
    }

    public forward(x: Tensor): Tensor {
        const ln_1_f = this.ln_1.forward(x);
        // console.log(`self.ln_1(x) ${ln_1_f.shape} ${ln_1_f.sum()}`)
        x = x.add(this.attn.forward(this.ln_1.forward(x)));
        // console.log(`x + self.attn(self.ln_1(x)) ${x.sum()} ${x.shape}`)
        x = x.add(this.mlpf(this.ln_2.forward(x)));
        // console.log(`x + self.attn(self.ln_2(x)) ${x.sum()} ${x.shape}`)

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
    return new Tensor(new Matrix(tensor.shape).prod());
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

    public forward(idx: Tensor, targets=null): [Tensor, Tensor] {
        // console.log("-----forward----");
        const [b, t] = idx.shape;
        // const pos = Tensor.arange(0, t).unsqueeze(0);
        const pos = new Tensor(Matrix.arange(0, t).unsqueeze(0));

        // console.log(`self.transformer.wte.weight ${this.transformer.wte.weight.shape} ${this.transformer.wte.weight.sum()}`);
        // console.log(`self.transformer.wpe.weight ${this.transformer.wpe.weight.shape} ${this.transformer.wpe.weight.sum()}`);

        // console.log(`idx ${idx.shape} ${idx.sum()}`);
        // console.log(`pos ${pos.shape} ${pos.sum()}`);

        const tok_emb = this.transformer.wte.forward(idx);
        const pos_emb = this.transformer.wpe.forward(pos);

        // console.log(`tok_emb ${tok_emb.shape} ${tok_emb.sum()}`);
        // console.log(`pos_emb ${pos_emb.shape} ${pos_emb.sum()}`);

        let x = this.transformer.drop.forward(tok_emb.add(pos_emb));
        const a = tok_emb.add(pos_emb);
        // console.log(`a ${a.shape} ${a.sum()}`);
        // console.log(`x ${x.shape} ${x.sum()}`);
        for (let i = 0; i < this.transformer.h.length; i++) {
            const block = this.transformer.h[i];
            // console.log(`Doing block ${i} forward`);
            x = block.forward(x);
            // console.log(`block(x) ${x.shape} ${x.sum()}`)
        }
        // console.log(`h ${x.shape} ${x.sum()}`);

        x = this.transformer.ln_f.forward(x);
        // console.log(`self.transformer ${x.shape} ${x.sum()}`);
        const logits = this.lm_head.forward(x);
        // console.log(`logits ${logits.shape} ${logits.sum()}`);

        const loss = new Tensor(0);
        if (targets) {
            // loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            throw Error("cross_entropy not implemented");
        }

        return [logits, loss];
    }

    public async load_weights_from_json(url: string) {
        const json = await fetch(url).then(response => response.json()).then(json => json);
        await this.load_state_dict(json);
        return;
    }

    public generate(idx: Tensor, max_new_tokens: number, temperature: number = 1.0, do_sample: boolean = false, top_k: number | null = null) {
        for (let i = 0; i < max_new_tokens; i++) {
            const idx_cond = idx.shape[1] < this.block_size ? idx : new Tensor(idx.data);
            // console.log(`idx ${idx} ${idx.shape}`);
            // console.log(`idx_cond ${idx_cond} ${idx_cond.shape}`);

            let [logits, _] = this.forward(idx_cond);
            console.log(`logits after forward ${logits.shape} ${logits.sum()}`);

            // pluck the logits at the final step and scale by desired temperature
            // if (logits.shape[0] != 1 || logits.shape[1] != 13 || logits.shape[2] != 65) {
            //     throw Error(`Expected logits shape to be [1, 13, 65] it is [${logits.shape}], the next slice wont work`);
            // }

            // logits = new Tensor(Matrix.slice(logits.data, [null, [1, 2], null])).reshape([1,65]);
            // console.log(`logits11 temp ${logits.shape} ${logits.sum()}`);

            // logits = new Tensor(Matrix.slice(logits.data, [[0, 1], [12, 13], [0, 65]]).div(temperature)).reshape([1,65]);

            // TODO: Fix getdata
            const logits_temp = Matrix.slice(logits.data, [null, [logits.shape[1]-1, logits.shape[1]], null]);
            logits = new Tensor(new Matrix(logits_temp.getData()).div(temperature)).reshape([1,65]);
            
            // logits = new Tensor(Matrix.slice(logits.data, [null, [logits.shape[1]-1, logits.shape[1]], null]));

            // console.log(`logits temp ${logits.shape} ${logits.sum()}`);

            const probs = softmax(logits, -1);
            // console.log(`probs ${probs}`);
            // console.log(`probs ${probs.shape} ${probs.mean()}`);

            let idx_next: Tensor

            if (do_sample) {
                // idx_next = torch.multinomial(probs, num_samples=1)
                // idx_next = Tensor.multinomial(probs.data, 1, true);
                idx_next = new Tensor(Matrix.multinomial(probs.data, 1, true));
                // console.log(`idx_next multi ${idx_next.shape} ${idx_next.sum()}`);
            }
            else {
                // _, idx_next = torch.topk(probs, k=1, dim=-1)
                // const [_, _idx_next] = 
                idx_next = new Tensor(0);
            }
            // throw Error("1")

            // # append sampled index to the running sequence and continue
            // console.log(`idx ${idx}`);
            // console.log(`idx_next ${idx_next}`);

            // idx = Tensor.cat([idx.data, idx_next.data], 1);
            idx = new Tensor(Matrix.cat([idx.data, idx_next.data], 1));

        }
        return idx;
    }
}

