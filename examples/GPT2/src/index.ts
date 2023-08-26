import { Tensor, Matrix } from "micrograd-ts";
import { GPT, NetworkConfig } from "./model";

import modelWeightsURL from '../models/gpt-nano/model_weights.json';
import textURL from '../models/input.txt';

class CharDataset {
    private config: NetworkConfig;
    private data: string;
    private data_size: number;
    private vocab_size: number;
    
    // public readonly stoi: Map<string, number>;
    // public readonly itos: Map<number, string>;

    public readonly stoi: { [k: string]: number };
    public readonly itos: { [k: number]: string };

    constructor(config: NetworkConfig, data: string) {
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
}


function numel(tensor: Tensor): Tensor {
    return new Tensor(new Matrix(tensor.shape).prod());
}

export class GPT2Demo {
    private lossElement: HTMLSpanElement;
    private networkElement: HTMLDivElement;
    private trainingElement: HTMLDivElement;

    private trainDataset: CharDataset;

    constructor(lossElement: HTMLSpanElement, networkElement: HTMLDivElement, trainingElement: HTMLDivElement) {
        // 'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
        const gptNanoConfig: NetworkConfig = {
            n_head: 3,
            n_embd: 48,

            attn_pdrop: 0.0, // disable dropouts for testing due to rng = 0.1
            resid_pdrop: 0.0, // disable dropouts for testing due to rng = 0.1
            embd_pdrop: 0.0, // disable dropouts for testing due to rng = 0.1

            vocab_size: 65, // 50257
            block_size: 128,

            n_layer: 3
        }

        fetch(textURL).then(response => response.text()).then(text => {
            this.trainDataset = new CharDataset(gptNanoConfig, text);
            
            const model = new GPT(gptNanoConfig);
    
            console.log(`${model}`)
    
            console.log(model.named_parameters())
    
            // const context = "O God, O God!"
            // const x = new Tensor(context.split("").map(v => this.trainDataset.stoi[v])).reshape([1, -1]);
            // console.log("x", x.shape)

    
            // throw Error("ergerg")
    
            let s = 0;
            for (let p of model.parameters()) {
                s += numel(p).data.data[0];
            }
    
            console.log(s)
    
            model.load_weights_from_json(modelWeightsURL).then(() => {
                console.log("LOADED MODEL")
    
                // const x = new Tensor([[27,  1, 19, 53, 42,  6,  1, 27,  1, 19, 53, 42,  2]]);
                const context = "O God, O God!"
                // const context = "O "
                const x = new Tensor(context.split("").map(v => this.trainDataset.stoi[v])).reshape([1, -1]);;

                const y = model.generate(x, 30, 1.0, true, 10).reshape([-1]);
                console.log(`y ${y}`)

                let completion = "";

                for (let ich of y.data.data) {
                    completion += this.trainDataset.itos[ich];
                }
                console.log("completion", completion)
                // const emb = new nn.Embedding(5000, config.n_embd);
            })
        })


    }
}