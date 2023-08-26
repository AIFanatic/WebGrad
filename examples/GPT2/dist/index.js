// src/index.ts
import { Tensor as Tensor2, Matrix as Matrix2 } from "micrograd-ts";

// src/model.ts
import { Tensor, Matrix, nn, Module, Random } from "micrograd-ts";
Random.SetRandomSeed(1337);
function softmax(att, dim) {
  const softmax2 = new nn.Softmax(dim);
  return softmax2.forward(att);
}
var NewGELU = class extends Module {
  constructor() {
    super();
  }
  forward(x) {
    const t1 = new Tensor(0.5);
    const t2 = new Tensor(1);
    const t4 = new Tensor(0.044715);
    const a = x.pow(3);
    const b = x.add(t4.mul(a));
    const c = new Tensor(Math.sqrt(2 / Math.PI));
    const d = c.mul(b);
    const e = t2.add(d.tanh());
    const f = t1.mul(x).mul(e);
    return f;
  }
};
var CausalSelfAttention = class extends Module {
  constructor(config) {
    super();
    if (config.n_embd % config.n_head !== 0)
      throw Error(`Wrong n_head(${config.n_head}) and n_embd(${config.n_embd})`);
    this.c_attn = new nn.Linear(config.n_embd, 3 * config.n_embd);
    this.c_proj = new nn.Linear(config.n_embd, config.n_embd);
    this.attn_dropout = new nn.Dropout(config.attn_pdrop);
    this.resid_dropout = new nn.Dropout(config.resid_pdrop);
    this.bias = new Tensor(Matrix.tril(Matrix.ones([config.block_size, config.block_size]))).reshape([1, 1, config.block_size, config.block_size]);
    this.n_head = config.n_head;
    this.n_embd = config.n_embd;
  }
  contiguous(tensor) {
    return new Tensor(tensor.data.getData());
  }
  forward(x) {
    const [B, T, C] = x.shape;
    const c_attn_f = this.c_attn.forward(x);
    const chunks = c_attn_f.data.split(this.n_embd);
    let [q, k, v] = [new Tensor(chunks[0]), new Tensor(chunks[1]), new Tensor(chunks[2])];
    k = k.reshape([B, T, this.n_head, Math.floor(C / this.n_head)]).transpose(1, 2);
    q = q.reshape([B, T, this.n_head, Math.floor(C / this.n_head)]).transpose(1, 2);
    v = v.reshape([B, T, this.n_head, Math.floor(C / this.n_head)]).transpose(1, 2);
    k = new Tensor(k.data);
    q = new Tensor(q.data);
    v = new Tensor(v.data);
    const t1 = new Tensor(1);
    let att = q.matmul(k.transpose(-2, -1)).mul(t1.div(Math.sqrt(k.shape[k.shape.length - 1])));
    const biasSliced = Matrix.slice(this.bias.data, [null, null, [0, T], [0, T]]);
    let maskedAttn = att.data.masked_fill(biasSliced.equal(0), 0);
    att = new Tensor(maskedAttn.getData());
    att = softmax(att, -1);
    att = this.attn_dropout.forward(att);
    let y = att.matmul(v);
    y = this.contiguous(y.transpose(1, 2)).reshape([B, T, C]);
    y = this.resid_dropout.forward(this.c_proj.forward(y));
    return y;
  }
};
var MLP = class extends Module {
  constructor(config) {
    super();
    this.c_fc = new nn.Linear(config.n_embd, 4 * config.n_embd), this.c_proj = new nn.Linear(4 * config.n_embd, config.n_embd), this.act = new NewGELU(), this.dropout = new nn.Dropout(config.resid_pdrop);
  }
};
var Block = class extends Module {
  constructor(config) {
    super();
    this.ln_1 = new nn.LayerNorm([config.n_embd]);
    this.attn = new CausalSelfAttention(config);
    this.ln_2 = new nn.LayerNorm([config.n_embd]);
    this.mlp = new MLP(config);
    this.mlpf = (x) => {
      const c_fc_f = this.mlp.c_fc.forward(x);
      const act_f = this.mlp.act.forward(c_fc_f);
      const c_proj_f = this.mlp.c_proj.forward(act_f);
      const drop_f = this.mlp.dropout.forward(c_proj_f);
      return drop_f;
    };
  }
  forward(x) {
    const ln_1_f = this.ln_1.forward(x);
    x = x.add(this.attn.forward(this.ln_1.forward(x)));
    x = x.add(this.mlpf(this.ln_2.forward(x)));
    return x;
  }
};
var Transformer = class extends Module {
  constructor(config) {
    super();
    this.wte = new nn.Embedding(config.vocab_size, config.n_embd);
    this.wpe = new nn.Embedding(config.block_size, config.n_embd);
    this.drop = new nn.Dropout(config.embd_pdrop);
    this.h = [];
    this.ln_f = new nn.LayerNorm([config.n_embd]);
    for (let i = 0; i < config.n_layer; i++) {
      this.h.push(new Block(config));
    }
  }
};
function numel(tensor) {
  return new Tensor(new Matrix(tensor.shape).prod());
}
var GPT = class extends Module {
  constructor(config) {
    super();
    if (config.vocab_size === void 0)
      throw Error("config.vocab_size cannot be null");
    if (config.block_size === void 0)
      throw Error("config.block_size cannot be null");
    this.block_size = config.block_size;
    this.transformer = new Transformer(config);
    this.lm_head = new nn.Linear(config.n_embd, config.vocab_size);
    let n_params = 0;
    for (let p of this.parameters()) {
      n_params += numel(p).data.get(0);
    }
  }
  forward(idx, targets = null) {
    const [b, t] = idx.shape;
    const pos = new Tensor(Matrix.arange(0, t).unsqueeze(0));
    const tok_emb = this.transformer.wte.forward(idx);
    const pos_emb = this.transformer.wpe.forward(pos);
    let x = this.transformer.drop.forward(tok_emb.add(pos_emb));
    const a = tok_emb.add(pos_emb);
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
    return [logits, loss];
  }
  async load_weights_from_json(url) {
    const json = await fetch(url).then((response) => response.json()).then((json2) => json2);
    await this.load_state_dict(json);
    return;
  }
  generate(idx, max_new_tokens, temperature = 1, do_sample = false, top_k = null) {
    for (let i = 0; i < max_new_tokens; i++) {
      const idx_cond = idx.shape[1] < this.block_size ? idx : new Tensor(idx.data);
      let [logits, _] = this.forward(idx_cond);
      console.log(`logits after forward ${logits.shape} ${logits.sum()}`);
      const logits_temp = Matrix.slice(logits.data, [null, [logits.shape[1] - 1, logits.shape[1]], null]);
      logits = new Tensor(new Matrix(logits_temp.getData()).div(temperature)).reshape([1, 65]);
      const probs = softmax(logits, -1);
      let idx_next;
      if (do_sample) {
        idx_next = new Tensor(Matrix.multinomial(probs.data, 1, true));
      } else {
        idx_next = new Tensor(0);
      }
      idx = new Tensor(Matrix.cat([idx.data, idx_next.data], 1));
    }
    return idx;
  }
};

// models/gpt-nano/model_weights.json
var model_weights_default = "./model_weights-P5FDERYG.json";

// models/input.txt
var input_default = "./input-3T5GKPYH.txt";

// src/index.ts
var CharDataset = class {
  constructor(config, data) {
    this.config = config;
    const chars = Array.from(new Set(data)).sort();
    this.data_size = data.length;
    this.vocab_size = chars.length;
    this.stoi = {};
    this.itos = {};
    for (let i = 0; i < chars.length; i++) {
      const ch = chars[i];
      this.stoi[ch] = i;
      this.itos[i] = ch;
    }
  }
};
function numel2(tensor) {
  return new Tensor2(new Matrix2(tensor.shape).prod());
}
var GPT2Demo = class {
  constructor(lossElement, networkElement, trainingElement) {
    const gptNanoConfig = {
      n_head: 3,
      n_embd: 48,
      attn_pdrop: 0,
      // disable dropouts for testing due to rng = 0.1
      resid_pdrop: 0,
      // disable dropouts for testing due to rng = 0.1
      embd_pdrop: 0,
      // disable dropouts for testing due to rng = 0.1
      vocab_size: 65,
      // 50257
      block_size: 128,
      n_layer: 3
    };
    fetch(input_default).then((response) => response.text()).then((text) => {
      this.trainDataset = new CharDataset(gptNanoConfig, text);
      const model = new GPT(gptNanoConfig);
      console.log(`${model}`);
      console.log(model.named_parameters());
      let s = 0;
      for (let p of model.parameters()) {
        s += numel2(p).data.data[0];
      }
      console.log(s);
      model.load_weights_from_json(model_weights_default).then(() => {
        console.log("LOADED MODEL");
        const context = "O God, O God!";
        const x = new Tensor2(context.split("").map((v) => this.trainDataset.stoi[v])).reshape([1, -1]);
        ;
        const y = model.generate(x, 30, 1, true, 10).reshape([-1]);
        console.log(`y ${y}`);
        let completion = "";
        for (let ich of y.data.data) {
          completion += this.trainDataset.itos[ich];
        }
        console.log("completion", completion);
      });
    });
  }
};
export {
  GPT2Demo
};
