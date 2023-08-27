import { TensorBuffer } from "./TensorBuffer";

export enum UnaryOps {
    ABS,
    EXP,
    TANH,
    LOG
};

export class UnaryOp {
    private static unary_op(x: TensorBuffer, op: UnaryOps): TensorBuffer {
        return x.unary_op(op);
    }

    public static abs(x: TensorBuffer) { return UnaryOp.unary_op(x, UnaryOps.ABS); }
    public static exp(x: TensorBuffer) { return UnaryOp.unary_op(x, UnaryOps.EXP); }
    public static tanh(x: TensorBuffer) { return UnaryOp.unary_op(x, UnaryOps.TANH); }
    public static log(x: TensorBuffer) { return UnaryOp.unary_op(x, UnaryOps.LOG); }
       
}