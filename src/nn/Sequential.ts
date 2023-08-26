import { Module } from "../Module";
import { Tensor } from "../Tensor";

export class Sequential extends Module {
    public modules: Module[];

    constructor(...modules: Module[]) {
        super();
        this.modules = modules;
    }


    public forward(x: Tensor): Tensor {
        for (let module of this.modules) {
            x = module.forward(x);
        }
        return x;
    }
}