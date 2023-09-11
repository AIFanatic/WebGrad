import { Tensor } from "./Tensor";
import { Device } from "./backend/Backend";

export type StateDict = { [key: string]: number[][] };
export type NameParametersDict = { [key: string]: Tensor };

export class Module {
    private _total_params: number;

    constructor() {
        this._total_params = 0;
    }

    public forward(x: Tensor): Tensor {
        throw Error("Not implemented");
    }

    public zero_grad() {
        for (let p of this.parameters()) {
            p.zero_grad();
        }
    }

    public get total_params(): number {
        return this._total_params;
    }

    public named_parameters(prefix: string = ''): NameParametersDict {
        let result: NameParametersDict = {};

        for (let key of Object.keys(this)) {
            let fullPath = prefix + key;
            
            // Traverse into submodules
            if (this[key] instanceof Module) {
                let childParameters = this[key].named_parameters(fullPath + '.');
                for (let childKey in childParameters) {
                    result[childKey] = childParameters[childKey];
                }
            }
            // Traverse arrays
            else if (Array.isArray(this[key])) {
                this[key].forEach((item, index) => {
                    if (item instanceof Module) {
                        let childParameters = item.named_parameters(fullPath + '.' + index + '.');
                        for (let childKey in childParameters) {
                            result[childKey] = childParameters[childKey];
                        }
                    }
                    // Register tensor parameters
                    else if (item instanceof Tensor) {
                        result[`${fullPath}[${index}]`] = item;
                    }
                });
            }
            // Register tensor parameters
            else if (this[key] instanceof Tensor) {
                result[fullPath] = this[key];
            }
        }

        return result;
    }

    public parameters(): Tensor[] {
        let params: Tensor[] = [];

        const keys = Object.keys(this);
        for (let key of keys) {
            const property = this[key];
            if (property instanceof Module) {
                const module = property;
                params.push(...module.parameters());
            }
            else if (property instanceof Array) {
                for(let ak of property) {
                    if (ak instanceof Module) {
                        const module = ak;
                        params.push(...module.parameters());
                    }
                }
            }
        }

        this._total_params = params.length;
        return params;
    }

    public get_name(): string {
        return this.constructor.name;
    }

    public toString(): string {
        
        function _addindent(s_: string, numSpaces: number): string {
            let s = s_.split("\n");
            if (s.length == 1) {
                return s_;
            }

            const first = s.shift();
            s = s.map(line => new Array(numSpaces).fill(" ").join("") + line)
            let str = "";
            for (let line of s) {
                str += "\n" + line;
            }
            str = first  + str;
            return str;
        }

        let child_lines: string[] = [];

        const keys = Object.keys(this);
        for (let key of keys) {
            if (this[key] instanceof Module || this[key]["parameters"]) {
                const module = this[key];
                let mod_str = `${module}`;
                mod_str = _addindent(mod_str, 2);
                child_lines.push('(' + key + '): ' + mod_str);
            }
            else if (this[key] instanceof Array && key !== "layers") {
                for(let ak of this[key]) {
                    const module = ak;
                    let mod_str = `${module}`;
                    mod_str = _addindent(mod_str, 2);
                    child_lines.push('(' + key + '): ' + mod_str);
                }
            }
        }

        const lines = child_lines;

        let main_str = this.get_name() + "(";

        if (lines) {
            main_str += '\n  ';
            for (let line of lines) {
                main_str += line + '\n  ';
            }
        }

        main_str += ")";
        return main_str;
    }

    public load_state_dict(stateDict: StateDict) {
        const namedParameters = this.named_parameters();
        const entries = Object.entries(stateDict);

        for (let state of entries) {
            const path = state[0];
            const tensor = state[1];
            
            if (!namedParameters[path]) throw Error(`Layer ${path} not found`);

            const modelTensor = namedParameters[path];

            const t = new Tensor(tensor, {device: modelTensor.device});
            const stateTensorShape = t.shape.reduce((p, c) => p * c);
            const modelParameterShape = modelTensor.shape.reduce((p, c) => p * c);
            if (stateTensorShape != modelParameterShape) throw Error(`State tensor shape (${stateTensorShape}) doesn't match model tensor shape (${modelParameterShape})`);
            
            modelTensor.assign(t);
        }
    }

    public to<T extends Module>(this: T, device: Device): T {
        for (let parameter of this.parameters()) {
            // parameter.assign(parameter.to(device));
            parameter.to(device);
        }
        return this;
    }
}