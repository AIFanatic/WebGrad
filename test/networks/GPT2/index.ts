import { Tensor } from "../../../src";
import { StateDict } from "../../../src/Module";

import { GPT, NetworkConfig } from "./model";

import fs from "fs";
import path from "path";

const modelWeightsPath = './models/gpt-nano/model_weights.json';
const textPath = './models/input.txt';

class CharDataset {
    public readonly stoi: { [k: string]: number };
    public readonly itos: { [k: number]: string };

    constructor(data: string) {
        const chars = Array.from(new Set(data)).sort();

        this.stoi = {};
        this.itos = {};
        for (let i = 0; i < chars.length; i++) {
            const ch = chars[i];
            this.stoi[ch] = i;
            this.itos[i] = ch;
        }
    }
}

export class GPT2Demo {
    private trainDataset: CharDataset;
    private model: GPT;

    constructor() {
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

        const text = fs.readFileSync(path.join(__dirname, "/", textPath), "utf-8");
        this.trainDataset = new CharDataset(text);
        
        this.model = new GPT(gptNanoConfig);
    }

    public async run(): Promise<Tensor> {
        const weights = await this.get_weights_from_file(modelWeightsPath);
        this.model.load_state_dict(weights);
        
        const context = "O God, O God!"
        const x = new Tensor(context.split("").map(v => this.trainDataset.stoi[v])).reshape([1, -1]);;
        const y = this.model.generate(x, 10, 1.0, true, 10).reshape([-1]);

        let completion = "";
        for (let ich of y.data.data) {
            completion += this.trainDataset.itos[ich];
        }

        return y;
    }

    public async get_weights_from_file(weightsPath: string): Promise<StateDict> {
        const json = JSON.parse(fs.readFileSync(path.join(__dirname, "/", weightsPath), "utf-8"));
        return json;
    }
}