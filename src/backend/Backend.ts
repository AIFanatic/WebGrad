import { TensorBuffer } from "./TensorBuffer";
import { CPUBuffer } from "./CPU";
import { WEBGLBuffer } from "./WebGL";

export enum Device {
    CPU,
    WEBGL
};

export class Backend {
    public static CreateFromArray(device: Device, array: Array<any>): TensorBuffer {
        if (device === Device.CPU) return CPUBuffer.CreateFromArray(array);
        else if (device === Device.WEBGL) return WEBGLBuffer.CreateFromArray(array);

        throw Error(`Unable to call CreateFromArray for device ${Device[device]}`);
    }

    public static CreateFromFloat32Array(device: Device, array: Float32Array, shape: number[] = [], strides: number[] = [], offset: number = 0): TensorBuffer {
        if (device === Device.CPU) return CPUBuffer.CreateFromFloat32Array(array, shape, strides, offset)
        else if (device === Device.WEBGL) return WEBGLBuffer.CreateFromFloat32Array(array, shape, strides, offset)
        
        throw Error(`Unable to call CreateFromFloat32Array for device ${Device[device]}`);
    }

    public static CreateFromNumber(device: Device, num: number): TensorBuffer {
        if (device === Device.CPU) return CPUBuffer.CreateFromNumber(num);
        else if (device === Device.WEBGL) return WEBGLBuffer.CreateFromNumber(num);

        throw Error(`Unable to call CreateFromNumber for device ${Device[device]}`);
    }

    public static CreateFromDataShapeAndStrides(data: TensorBuffer, shape: number[], strides: number[], offset: number = 0): TensorBuffer {
        if (data instanceof CPUBuffer) return new CPUBuffer(data, shape, strides, offset);
        else if (data instanceof WEBGLBuffer) return new WEBGLBuffer(data, shape, strides, offset);

        throw Error(`Unable to call CreateFromDataShapeAndStrides`);
    }
}