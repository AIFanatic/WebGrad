import { Device } from "./Backend";
import { MovementOp } from "./MovementOps";
import { TensorBuffer } from "./TensorBuffer";

export enum BinaryOps {
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    CMPEQ,
    MAX
};

export class BinaryOp {
    private static binary_op(x: TensorBuffer, y: TensorBuffer, op: BinaryOps): TensorBuffer {
        if (x.device !== y.device) {
            throw Error(`Cannot perform binary op since TensorBuffer 1 is on ${Device[x.device]} and TensorBuffer 2 is on ${Device[y.device]}`);
        }

        // console.log("x contiguous", x.is_contiguous(), x.data.data, x.shape, x.strides);
        // console.log("y contiguous", y.is_contiguous(), y.data.data, y.shape, y.strides)
        if (!x.is_contiguous()) x = MovementOp.contiguous(x);
        if (!y.is_contiguous()) y = MovementOp.contiguous(y);
        // x = MovementOp.contiguous(x);
        // y = MovementOp.contiguous(y);

        return x.binary_op(y, op);
    }

    public static add(x: TensorBuffer, y: TensorBuffer) { return BinaryOp.binary_op(x, y, BinaryOps.ADD); }
    public static sub(x: TensorBuffer, y: TensorBuffer) { return BinaryOp.binary_op(x, y, BinaryOps.SUB); }
    public static mul(x: TensorBuffer, y: TensorBuffer) { return BinaryOp.binary_op(x, y, BinaryOps.MUL); }
    public static div(x: TensorBuffer, y: TensorBuffer) { return BinaryOp.binary_op(x, y, BinaryOps.DIV); }
    public static pow(x: TensorBuffer, y: TensorBuffer) { return BinaryOp.binary_op(x, y, BinaryOps.POW); }
    public static equal(x: TensorBuffer, y: TensorBuffer) { return BinaryOp.binary_op(x, y, BinaryOps.CMPEQ); }
    public static maximum(x: TensorBuffer, y: TensorBuffer) { return BinaryOp.binary_op(x, y, BinaryOps.MAX); }
}