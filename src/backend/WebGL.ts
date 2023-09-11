import { Device } from "./Backend";
import { TensorBuffer } from "./TensorBuffer";
import { UnaryOps } from "./UnaryOps";
import { BinaryOps } from "./BinaryOps";
import { ReduceOps } from "./ReduceOps";
import { MovementOp } from "./MovementOps";

function equalArrays(a: number[], b: number[]): boolean {
    if (a.length !== b.length) return false;

    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

enum UniformType {
    FLOAT,
    FLOAT_VEC2,
    FLOAT_VEC3,
    FLOAT_VEC4,

    INT,
    INT_VEC2,
    INT_VEC3,
    INT_VEC4,

    FLOAT_ARRAY,
    INT_ARRAY,

    SAMPLER2D,
    SAMPLERCUBE,
};

interface WEBGLKernelUniform {
    name: string;
    value: number | number[];
    type: UniformType;
};

interface ITextureInfo {
    width: number;
    height: number;
    internalFormat: number;
    format: number;
    type: number;
    originalShape: number[];
}

export class Texture {
    public data: Float32Array | null;
    public readonly width: number;
    public readonly height: number;
    public readonly internalFormat: number;
    public readonly format: number;
    public readonly type: number;
    public readonly texture: WebGLTexture;
    public readonly originalShape: number[];

    public creator: string;

    constructor(data: Float32Array | null, info: ITextureInfo) {
        if (data === undefined) throw Error("Got undefined data");

        this.data = data;
        this.width = info.width;
        this.height = info.height;
        this.internalFormat = info.internalFormat;
        this.format = info.format;
        this.type = info.type;
        this.originalShape = info.originalShape;

        const gl = WEBGLContext.gl;

        if (data instanceof Texture) {
            if (!data.texture) throw Error("Passed texture but no data.texture found");
            gl.bindTexture(gl.TEXTURE_2D, data.texture);
            this.texture = data.texture;
        }
        else {
            const texture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texImage2D(gl.TEXTURE_2D, 0, info.internalFormat, info.width, info.height, 0, info.format, info.type, data);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

            this.texture = texture;
        }

        try {
            throw Error("creator");
        } catch (error) {
            this.creator = error;
        }
    }

    public toString() {
        return `Texture(
            data=[${this.data}],
            width=${this.width},
            height=${this.height},
            internalFormat=${WEBGLContext.glCodeToStr(this.internalFormat)}
            format=${WEBGLContext.glCodeToStr(this.format)},
            type=${WEBGLContext.glCodeToStr(this.type)}
        )`
    }

    private decodeMatrixFromUnpackedArray(unpackedArray: Float32Array, matrix: Float32Array, channelsPerTexture: number) {
        function getMatrixSizeFromUnpackedArraySize(unpackedSize: number, channelsPerTexture: number): number {
            if (unpackedSize % channelsPerTexture !== 0) {
                throw new Error(`unpackedSize (${unpackedSize}) must be a multiple of ${channelsPerTexture}`);
            }
            return unpackedSize / channelsPerTexture;
        }

        const requiredSize = getMatrixSizeFromUnpackedArraySize(unpackedArray.length, channelsPerTexture);
        if (matrix.length < requiredSize) throw new Error(`matrix length (${matrix.length}) must be >= ${requiredSize}`);

        let dst = 0;
        for (let src = 0; src < unpackedArray.length; src += channelsPerTexture) {
            matrix[dst++] = unpackedArray[src];
        }
    }

    public read(): Float32Array {
        if (this.data) return this.data;

        this.data = WEBGLContext.readTextureData(this);

        return this.data;
    }

    public static shapeTo2d(shape: number[]): [number, number] {
        let shape2d: [number, number] = [shape[0], 1];
        for (let i = 1; i < shape.length; i++) {
            shape2d[1] *= shape[i];
        }
        return shape2d;

        // let shape2d: [number, number] = [1, shape[shape.length - 1]];
        // for (let i = shape.length - 1; i >= 0; i--) {
        //     shape2d[0] *= shape[i];
        // }
        // return shape2d;
    }

    // Computes the width and height necessary to fit in an RGBA texture.
    // For example if the shape is [3,3] the data size is 3*3=9.
    // If the texture is w=2,h=1, this only 2*1*4=8pixels, we need 9.
    // So the pixels need to be w=2,h=2 which is 2*2*4=16 (16-9=7 pixels are ignored).
    //
    // TODO: The height is being increased if the pixels still dont fit the data.
    //       Is this enought or are there any exceptions?
    // public static calculateWidthAndHeightToFitShape(shape: number[], channels: number): [number, number] {
    //     const shape2D = Texture.shapeTo2d(shape);
    //     const prodShape = shape2D.reduce((p, c) => p * c);

    //     const width = Math.ceil(Math.sqrt(prodShape / channels));
    //     let height = Math.floor(Math.sqrt(prodShape / channels));

    //     if (width * height * channels < prodShape) height++;
    //     if (width * height * channels < prodShape) throw Error("Couldnt get enough pixels to compute.");

    //     return [width, height];
    // }
    public static calculateWidthAndHeightToFitShape(shape: number[], channels: number): [number, number] {
        const shape2D = Texture.shapeTo2d(shape);
        const prodShape = shape2D.reduce((p, c) => p * c);
    
        let width = Math.ceil(Math.sqrt(prodShape / channels));
        let height = Math.floor(Math.sqrt(prodShape / channels));
    
        while (width * height * channels < prodShape) {
            height++;
            if (width * height * channels < prodShape) width++;
        }
    
        return [width, height];
    }

    public static createUnpackedFromShape(data: Float32Array | null, shape: number[]): Texture {
        const gl = WEBGLContext.gl;
        // const [width, height] = Texture.shapeTo2d(shape);
        const [width, height] = Texture.calculateWidthAndHeightToFitShape(shape, 1);
        if (data && data.length < width * height) {
            const _data = new Float32Array(width * height);
            _data.set(data);
            data = _data;
        }
        return new Texture(data, { width: width, height: height, internalFormat: gl.R32F, format: gl.RED, type: gl.FLOAT, originalShape: shape});
    }

    public static createUnpackedFromDimensions(data: Float32Array | null, width: number, height: number): Texture {
        const gl = WEBGLContext.gl;
        return new Texture(data, { width: width, height: height, internalFormat: gl.R32F, format: gl.RED, type: gl.FLOAT, originalShape: [width, height] });
    }
}

interface WEBGLCacheShader {
    key: number;
    code: string;
    shader: WebGLShader;
};

export class WEBGLCache {
    private static shaderCache: Map<number, WEBGLCacheShader> = new Map();
    private static programCache: Map<number, WebGLProgram> = new Map();

    private static hashCode(s: string): number {
        let h = 0;
        for (let i = 0; i < s.length; i++) {
            h = Math.imul(31, h) + s.charCodeAt(i) | 0;
        }

        return h;
    }

    public static getShader(code: string): WebGLShader | null {
        const key = WEBGLCache.hashCode(code);
        return WEBGLCache.shaderCache.has(key) ? WEBGLCache.shaderCache.get(key).shader : null;
    }

    public static setShader(code: string, shader: WebGLShader) {
        const key = WEBGLCache.hashCode(code);
        WEBGLCache.shaderCache.set(key, {
            key: key,
            code: code,
            shader: shader
        })
    }

    public static hasShader(code: string): boolean {
        const key = WEBGLCache.hashCode(code);
        return WEBGLCache.shaderCache.has(key);
    }

    public static getProgram(fragmentCode: string): WebGLProgram {
        const key = WEBGLCache.hashCode(fragmentCode);
        return WEBGLCache.programCache.has(key) ? WEBGLCache.programCache.get(key) : null;
    }

    public static setProgram(fragmentCode: string, program: WebGLProgram) {
        const key = WEBGLCache.hashCode(fragmentCode);
        WEBGLCache.programCache.set(key, program);
    }

    public static hasProgram(code: string): boolean {
        const key = WEBGLCache.hashCode(code);
        return WEBGLCache.programCache.has(key);
    }
}

export class WEBGLContext {
    private static defaultVertexShader = `#version 300 es
    precision highp float;
    in vec3 clipSpacePos;
    in vec2 uv;
    out vec2 resultUV;

    void main() {
        gl_Position = vec4(clipSpacePos, 1);
        resultUV = uv;
    }`;

    private static glCodesTable;

    private static _gl: WebGL2RenderingContext;
    public static get gl(): WebGL2RenderingContext {
        if (!WEBGLContext._gl) WEBGLContext._gl = WEBGLContext.setup();
        return WEBGLContext._gl;
    }

    constructor() {
        throw Error("Cannot call WEBGLContext with new.");
    }

    private static hashCode(s) {
        let h = 0;
        for (let i = 0; i < s.length; i++)
            h = Math.imul(31, h) + s.charCodeAt(i) | 0;

        return h;
    }

    private static setup(): WebGL2RenderingContext {
        if (typeof window === "undefined") throw Error("Window not found, WebGL2 is only supported in browsers.");
        if (typeof document === "undefined") throw Error("Document not found, WebGL2 is only supported in browsers.");

        const canvas = document.createElement("canvas");
        // canvas.width = 512;
        // canvas.height = 512;

        const gl = canvas.getContext("webgl2");
        if (!gl) throw Error("Could not setup WebGL2");
        if (!gl.getExtension("EXT_color_buffer_float")) throw Error("EXT_color_buffer_float not supported");

        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        document.body.appendChild(canvas);

        return gl;
    }

    private static compileShader(gl: WebGL2RenderingContext, shaderType: WebGLRenderingContextBase["VERTEX_SHADER"] | WebGLRenderingContextBase["FRAGMENT_SHADER"], shaderCode: string): WebGLShader {
        const shader = gl.createShader(shaderType);
        gl.shaderSource(shader, shaderCode);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.log(`${shaderCode}`);
            throw new Error('Could not compile shader: ' + gl.getShaderInfoLog(shader));
        }

        return shader;
    }

    private static createShaderProgram(gl: WebGL2RenderingContext, vertexShader: WebGLShader, fragmentShader: WebGLShader): WebGLProgram {
        const shaderProgram = gl.createProgram();
        gl.attachShader(shaderProgram, vertexShader);
        gl.attachShader(shaderProgram, fragmentShader);
        gl.linkProgram(shaderProgram);

        if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
            const linkErrLog = gl.getProgramInfoLog(shaderProgram);
            throw Error(`Shader program did not link successfully. Error: ${linkErrLog}`);
        }

        return shaderProgram;
    }

    private static createQuad(gl: WebGL2RenderingContext, program: WebGLProgram): WebGLBuffer {
        const vertex_data = new Float32Array([-1, 1, 0, 0, 1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, 0]);
        const vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertex_data, gl.STATIC_DRAW);

        const posOffset = 0;               // x is the first buffer element
        const uvOffset = 3 * 4;            // uv comes after [x y z]
        const stride = (3 * 4) + (2 * 4);  // xyz + uv, each entry is 4-byte float.

        const clipSpacePosLoc = gl.getAttribLocation(program, "clipSpacePos");
        gl.vertexAttribPointer(clipSpacePosLoc, 3, gl.FLOAT, false, stride, posOffset);
        gl.enableVertexAttribArray(clipSpacePosLoc);

        const uvLoc = gl.getAttribLocation(program, "uv");
        gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, stride, uvOffset);
        gl.enableVertexAttribArray(uvLoc);

        return vertexBuffer;
    }

    private static bindTexture(gl: WebGL2RenderingContext, texture: WebGLTexture, program: WebGLProgram, location: number): WebGLUniformLocation {
        gl.activeTexture(gl.TEXTURE0 + location);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        const textureLocation = gl.getUniformLocation(program, `u_tex${location}`);
        gl.uniform1i(textureLocation, location); // Texture unit 0
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.uniform1i(textureLocation, location);

        return textureLocation;
    }

    private static processUniforms(gl: WebGL2RenderingContext, program: WebGLProgram, uniforms: WEBGLKernelUniform[]) {
        if (!uniforms) return;

        for (let i = 0; i < uniforms.length; i++) {
            const name = uniforms[i].name;
            const value = uniforms[i].value;
            const type = uniforms[i].type;

            const location = gl.getUniformLocation(program, name);
            if (location === null) throw Error(`Got null uniform location ${name} ${value}`)

            if (value instanceof Array) {
                if (type === UniformType.FLOAT_ARRAY) gl.uniform1fv(location, value);
                else if (type === UniformType.INT_ARRAY) gl.uniform1iv(location, value);
                else if (type === UniformType.FLOAT_VEC2) gl.uniform2fv(location, value);
                else if (type === UniformType.FLOAT_VEC3) gl.uniform3fv(location, value);
                else if (type === UniformType.FLOAT_VEC4) gl.uniform4fv(location, value);
                else if (type === UniformType.INT_VEC2) gl.uniform2iv(location, value);
                else if (type === UniformType.INT_VEC3) gl.uniform3iv(location, value);
                else if (type === UniformType.INT_VEC4) gl.uniform4iv(location, value);
            }
            else {
                if (type === UniformType.FLOAT) gl.uniform1f(location, value);
                if (type === UniformType.INT) gl.uniform1i(location, value);
            }
        }
    }

    public static runKernel(shader: string, inputs: Texture[], output: Texture, uniforms?: WEBGLKernelUniform[]): void {
        if (inputs.length === 0) throw Error("Cannot run kernel without any buffers.");

        const gl = WEBGLContext.gl;

        let vertexShader = WEBGLCache.getShader(WEBGLContext.defaultVertexShader);
        let fragmentShader = WEBGLCache.getShader(shader);

        if (vertexShader === null) {
            vertexShader = WEBGLContext.compileShader(gl, gl.VERTEX_SHADER, WEBGLContext.defaultVertexShader);
            WEBGLCache.setShader(WEBGLContext.defaultVertexShader, vertexShader);
        }
        if (fragmentShader === null) {
            fragmentShader = WEBGLContext.compileShader(gl, gl.FRAGMENT_SHADER, shader);
            WEBGLCache.setShader(shader, fragmentShader);
        }

        let shaderProgram = WEBGLCache.getProgram(shader);
        if (shaderProgram === null) {
            shaderProgram = WEBGLContext.createShaderProgram(gl, vertexShader, fragmentShader);
            WEBGLCache.setProgram(shader, shaderProgram);
        }

        const quadVertexBuffer = WEBGLContext.createQuad(gl, shaderProgram);

        gl.useProgram(shaderProgram);

        WEBGLContext.processUniforms(gl, shaderProgram, uniforms);

        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.viewport(0, 0, output.width, output.height);

        WEBGLContext.bindTexture(gl, output.texture, shaderProgram, 0);

        // Create and bind the framebuffer
        const fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, output.texture, 0);

        for (let i = 0; i < inputs.length; i++) {
            const bufferLocation = WEBGLContext.bindTexture(gl, inputs[i].texture, shaderProgram, i);
        }

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    public static readTextureData(texture: Texture): Float32Array {
        const gl = WEBGLContext.gl;
        const framebuffer = gl.createFramebuffer();

        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture.texture, 0);

        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw Error("Framebuffer incomplete");

        const data = new Float32Array(texture.width * texture.height * 1);
        gl.readBuffer(gl.COLOR_ATTACHMENT0);
        gl.readPixels(0, 0, texture.width, texture.height, gl.RED, gl.FLOAT, data);
        // console.log("readPixels", data);

        // Clean up
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(framebuffer);

        return data;
    }

    public static glCodeToStr(code) {
        if (!WEBGLContext.glCodesTable) {
            WEBGLContext.glCodesTable = {};
            for (var key in WEBGLContext.gl) {
                WEBGLContext.glCodesTable[WEBGLContext.gl[key]] = key;
            }
        }
        return "gl." + WEBGLContext.glCodesTable[code];
    }
}

export class WEBGLBuffer extends TensorBuffer {
    public readonly shape: number[];
    public readonly strides: number[];
    public readonly offset: number;

    public data: Float32Array;
    public texture: Texture;

    public creator: string;

    constructor(data: WEBGLBuffer | Texture | Float32Array, shape: number[], strides: number[], offset: number) {
        if (!data || data === null) throw Error("Cannot create buffer with no data");

        super(shape, strides, offset, Device.WEBGL);

        // if (data instanceof Texture) {
        //     if (!equalArrays(data.originalShape, shape)) {
        //         console.warn("Passed texture", data.originalShape, shape);
        //     }
        // }
        
        if (data instanceof Float32Array) this.data = data;
        if (data instanceof Texture) this.texture = data;
        if (data instanceof WEBGLBuffer) {
            if (!data.data && !data.texture) throw Error("Tried to create WEBGLBuffer with no data or texture.");
            if (data.data) this.data = data.data;
            else if (data.texture) this.texture = data.texture;
        }

        try {
            throw Error("creator");
        } catch (error) {
            this.creator = error;
        }
    }

    public static CreateFromArray(array: Array<any>): WEBGLBuffer {
        const data = Float32Array.from(array.flat(Infinity));
        const shape = TensorBuffer.computeShape(array);
        const strides = TensorBuffer.computeStrides(shape);
        return new WEBGLBuffer(data, shape, strides, 0);
    }

    public static CreateFromFloat32Array(data: Float32Array, shape: number[] = [], strides: number[] = [], offset: number = 0): WEBGLBuffer {
        const _shape = shape.length === 0 ? [data.length] : shape;
        const _strides = strides.length === 0 ? this.computeStrides(_shape) : strides;
        return new WEBGLBuffer(data, _shape, _strides, offset);
    }

    public static CreateFromNumber(num: number): WEBGLBuffer {
        return new WEBGLBuffer(new Float32Array([num]), [1], [1], 0);
    }

    protected getInternalData(): Float32Array {
        return this.data ? this.data : this.texture.read();
    }

    public createUnpackedTexture(): Texture {
        if (!this.data && !this.texture) throw Error("Tried to create unpacked texture without a data or texture field");

        if (this.texture) {
            if (!equalArrays(this.shape, this.texture.originalShape)) {
                // this.texture = Texture.createUnpackedFromShape(this.texture.read(), this.shape);
                this.texture = this.copyToShape(this.shape).texture;
            }
            return this.texture;
        }
        else if (this.data) {
            this.texture = Texture.createUnpackedFromShape(this.data, this.shape);
        }
        return this.texture;
    }

    public createUnpackedTextureFromDimensions(width: number, height: number): Texture {
        if (!this.data && !this.texture) throw Error("Tried to create unpacked texture without a data or texture field");

        if (this.texture) {
            if (this.texture.data) {
                this.texture = Texture.createUnpackedFromDimensions(this.texture.data, width, height);
            }
            return this.texture;
        }
        else if (this.data) {
            this.texture = Texture.createUnpackedFromDimensions(this.data, width, height);
        }
        return this.texture;
    }

    public unary_op(op: UnaryOps): WEBGLBuffer {
        function processOp(op: UnaryOps): string {
            if (op === UnaryOps.ABS) return "abs(t1)";
            else if (op === UnaryOps.EXP) return "exp(t1)";
            else if (op === UnaryOps.TANH) return "tanh(t1)";
            else if (op === UnaryOps.LOG) return "log(t1)";
        }

        const inputTexture = this.createUnpackedTexture();
        const outputTexture = Texture.createUnpackedFromShape(null, this.shape);

        WEBGLContext.runKernel(`#version 300 es
        precision highp int;
        precision highp float;
        precision highp sampler2D;
        
        uniform sampler2D u_tex0;

        out vec4 result;
        
        void main() {
            ivec2 coords = ivec2(gl_FragCoord.xy);
            vec4 t1 = texelFetch(u_tex0, coords, 0);

            result = ${processOp(op)};
        }`, [inputTexture], outputTexture);

        return new WEBGLBuffer(outputTexture, this.shape, this.strides, this.offset);
    }

    public binary_op(other: WEBGLBuffer, op: BinaryOps): WEBGLBuffer {
        function processOp(op: BinaryOps): string {
            if (op === BinaryOps.ADD) return "t1 + t2";
            else if (op === BinaryOps.SUB) return "t1 - t2";
            else if (op === BinaryOps.MUL) return "t1 * t2";
            else if (op === BinaryOps.DIV) return "t1 / t2";
            else if (op === BinaryOps.POW) return "myPow(t1, t2)";
            else if (op === BinaryOps.CMPEQ) return "vec4(t1.r == t2.r, t1.g == t2.g, t1.b == t2.b, t1.a == t2.a)"; // Probably needs epsilon
            else if (op === BinaryOps.MAX) return "max(t1, t2)";
        }

        const inputTextureX = this.createUnpackedTexture();
        const inputTextureY = other.createUnpackedTexture();
        const outputTexture = Texture.createUnpackedFromShape(null, this.shape);

        WEBGLContext.runKernel(`#version 300 es
        precision highp int;
        precision highp float;
        precision highp sampler2D;

        uniform sampler2D u_tex0;
        uniform sampler2D u_tex1;

        out vec4 result;

        // Pow behaves differently in WEBGL, pow(-2, 3) = 8 instead of -8
        // This still doesn't work for cases where the base is negative and exponent is fractional,
        // this should return 0 to match js
        vec4 myPow(vec4 base, vec4 exponent) {
            vec4 absBase = abs(base); // Absolute value of base
            vec4 rawPow = pow(absBase, exponent); // Compute pow using absolute values
            vec4 isOdd = mod(exponent, 2.0); // Check if exponents are odd
            vec4 signBase = sign(base); // Get the sign of each base component
            return mix(rawPow, signBase * rawPow, isOdd); // Mix based on odd/even exponent
        }

        void main() {
            ivec2 coords = ivec2(gl_FragCoord.xy);
            vec4 t1 = texelFetch(u_tex0, coords, 0);
            vec4 t2 = texelFetch(u_tex1, coords, 0);
        
            result = ${processOp(op)};
        }`, [inputTextureX, inputTextureY], outputTexture);

        return new WEBGLBuffer(outputTexture, this.shape, this.strides, this.offset);
    }

    public reduce_op(op: ReduceOps, axes: number[]): WEBGLBuffer {
        function prod(array: number[]): number {
            return array.reduce((p, c) => p * c);
        }

        // console.log(`input ${this.texture.read()} ${this.shape} ${this.strides} ${axes}`);
        // console.log(`axes ${axes}`);

        const axisLength = axes.length === this.shape.length ? prod(this.shape) : this.shape[this.shape.length - 1];

        function sumDim(input: Texture, shape: number[], stride: number): Texture {
            const outputTexture = Texture.createUnpackedFromShape(null, shape);

            const uniforms: WEBGLKernelUniform[] = [
                { name: "width", value: outputTexture.width, type: UniformType.INT },
                { name: "u_stride", value: stride, type: UniformType.INT },
                { name: "u_axisLength", value: axisLength, type: UniformType.INT },
                { name: "u_op", value: op, type: UniformType.INT },
            ];

            WEBGLContext.runKernel(`#version 300 es
            precision highp int;
            precision highp float;
            precision highp int;
            
            uniform sampler2D u_tex0;

            uniform int width;
            uniform int u_stride;
            uniform int u_axisLength;
            uniform int u_op;

            const float EPS = 0.001;
            
            out float result;

            ivec2 getIndexCoords(int index) {
                return ivec2(index % width, index / width);
            }

            int getIndexAxis(int index) {
                float v = float(index) / float(u_axisLength);
                return int(floor(v + EPS));
            }
            
            void main() {
                int index = int(gl_FragCoord.x) + int(gl_FragCoord.y) * width;

                int t1Index = index;
                int t2Index = index + u_stride;
                int t1Axis = getIndexAxis(t1Index);
                int t2Axis = getIndexAxis(t2Index);

                ivec2 t1 = getIndexCoords(t1Index);
                ivec2 t2 = getIndexCoords(t2Index);

                if (t1Axis != t2Axis) {
                    float value1 = texelFetch(u_tex0, t1, 0).r;
                    result = value1;
                    return;
                }
            
                float value1 = texelFetch(u_tex0, t1, 0).r;
                float value2 = texelFetch(u_tex0, t2, 0).r;
            
                if (u_op == 0) result = value1 + value2;
                else if (u_op == 1) result = value1 * value2;
            }`, [input], outputTexture, uniforms);

            return outputTexture;
        }
        const inputTexture = this.createUnpackedTexture();
        let outputTexture: Texture = inputTexture;
        let stride = 1; // Starting with adjacent elements.

        const totalNumberOfElements = prod(this.shape);
        while (stride < totalNumberOfElements) {
            // console.log(`${outputTexture.read()}`)
            outputTexture = sumDim(outputTexture, this.shape, stride);
            stride *= 2;
        }


        const outputTexturePacked = Texture.createUnpackedFromShape(null, this.shape);
        // Pack data
        const uniforms: WEBGLKernelUniform[] = [
            { name: "width", value: outputTexturePacked.width, type: UniformType.INT },
            { name: "u_axisLength", value: axisLength, type: UniformType.INT },
        ];

        WEBGLContext.runKernel(`#version 300 es
        precision highp int;
        precision highp float;
        precision highp sampler2D;

        uniform sampler2D u_tex0;

        uniform int width;
        uniform int u_axisLength;

        out float result;

        ivec2 getIndexCoords(int index) {
            return ivec2(index % width, index / width);
        }

        void main() {
            int index = int(gl_FragCoord.x) + int(gl_FragCoord.y) * width;
            ivec2 coords = getIndexCoords(index * u_axisLength);
            vec4 t1 = texelFetch(u_tex0, coords, 0);
        
            result = t1.r;
        }`, [outputTexture], outputTexturePacked, uniforms);

        // console.timeEnd("reduce_op");


        function calculateReducedShape(originalShape, axes, keepdim = false) {
            if (!keepdim) {
                return originalShape.filter((_, index) => !axes.includes(index));
            } else {
                return originalShape.map((dim, index) => axes.includes(index) ? 1 : dim);
            }
        }
        let resultShape = calculateReducedShape(this.shape, axes, false);
        resultShape = resultShape.length === 0 ? resultShape = [1] : resultShape;

        // const r = new WEBGLBuffer(outputTexturePacked, resultShape, TensorBuffer.computeStrides(resultShape), this.offset);
        // return r;
        const r = new WEBGLBuffer(outputTexturePacked, outputTexture.originalShape, TensorBuffer.computeStrides(outputTexture.originalShape), this.offset);
        return MovementOp.reshape(r, resultShape) as unknown as WEBGLBuffer;
    }

    public contiguous(): WEBGLBuffer {
        const inputTexture = this.createUnpackedTexture();
        const outputTexture = Texture.createUnpackedFromShape(null, this.shape);
        
        // console.log("CALLED CONTIGUOS", inputTexture.read());

        const MAX_DIMS = 10;
        if (this.shape.length !== this.strides.length) throw Error("Shape does not match strides");
        if (this.shape.length > MAX_DIMS) throw Error(`Maximum dimensions for contiguous call are 10, got ${this.shape.length}`);

        const uniforms: WEBGLKernelUniform[] = [
            // {name: "MAX_DIMS", value: MAX_DIMS, type: UniformType.INT},
            { name: "width", value: inputTexture.width, type: UniformType.INT },
            { name: "shape", value: this.shape, type: UniformType.INT_ARRAY },
            { name: "strides", value: this.strides, type: UniformType.INT_ARRAY },
            { name: "offset", value: this.offset, type: UniformType.INT },
            { name: "shapeLength", value: this.shape.length, type: UniformType.INT },
        ];

        WEBGLContext.runKernel(`#version 300 es
        precision highp int;
        precision highp float;
        precision highp sampler2D;

        uniform sampler2D u_tex0;

        in vec2 resultUV;
        out float result;

        const int MAX_DIMS = 10;

        uniform int width;
        uniform int[10] shape;
        uniform int[10] strides;
        uniform int offset;
        uniform int shapeLength;

        float getData(sampler2D tensor, int width, int i, int offset, int[MAX_DIMS] shape, int[MAX_DIMS] strides, int numDims) {
            int idx = 0;
            int totalSize = 1;
        
            for (int j = 0; j < numDims; ++j) {
                totalSize *= shape[j];
            }
        
            for (int dim = 0; dim < numDims; ++dim) {
                totalSize /= shape[dim];
                int coord = int(i / totalSize);
                i -= coord * totalSize;
                idx += strides[dim] * coord;
            }
            idx += offset;
        
            ivec2 coords = ivec2(idx % width, idx / width);
            return texelFetch(tensor, coords, 0).r;
        }

        void main() {
            int index = int(gl_FragCoord.x) + int(gl_FragCoord.y) * width;
            float d = getData(u_tex0, width, index, offset, shape, strides, shapeLength);
            result = d;
        }`, [inputTexture], outputTexture, uniforms);

        const r = new WEBGLBuffer(outputTexture, this.shape, TensorBuffer.computeStrides(this.shape), this.offset);
        return r;
    }

    private copyToShape(shape: number[]): WEBGLBuffer {
        const inputTexture = this.texture;
        const outputTexture = Texture.createUnpackedFromShape(null, shape.slice());

        const uniforms: WEBGLKernelUniform[] = [
            { name: "widthIn", value: inputTexture.width, type: UniformType.INT },
            { name: "widthOut", value: outputTexture.width, type: UniformType.INT },
        ];

        WEBGLContext.runKernel(`#version 300 es
        precision highp int;
        precision highp float;
        precision highp sampler2D;
        
        uniform sampler2D u_tex0;
        uniform int widthIn;
        uniform int widthOut;

        out float result;

        ivec2 getIndexCoords(int index) {
            return ivec2(index % widthIn, index / widthIn);
        }

        void main() {
            int index = int(gl_FragCoord.x) + int(gl_FragCoord.y) * widthOut;
            ivec2 coords = getIndexCoords(index);

            vec4 t1 = texelFetch(u_tex0, coords, 0);
        
            result = t1.r;
        }`, [inputTexture], outputTexture, uniforms);

        return new WEBGLBuffer(outputTexture, shape, TensorBuffer.computeStrides(shape), this.offset);
    }




    public toString() {
        function fixed(key, val) {
            return val.toFixed ? Number(val.toFixed(4)) : val;
        }
        return JSON.stringify(this.getData(), fixed)
    }

    public copy(): WEBGLBuffer {
        throw Error("Not implemented");
    }
}