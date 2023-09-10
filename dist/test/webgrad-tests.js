var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// src/backend/TensorBuffer.ts
var TensorBuffer = class {
  constructor(shape, strides, offset, device) {
    this.shape = shape.slice();
    this.strides = strides.slice();
    this.offset = offset;
    this.device = device;
  }
  copy() {
    throw Error("Not implemented");
  }
  static CreateFromArray(array) {
    throw Error("Not implemented");
  }
  static CreateFromFloat32Array(array, shape = [], strides = [], offset = 0) {
    throw Error("Not implemented");
  }
  static CreateFromNumber(num) {
    throw Error("Not implemented");
  }
  unary_op(op) {
    throw Error("UnaryOp not implemented");
  }
  binary_op(other, op) {
    throw Error("BinaryOp not implemented");
  }
  reduce_op(op, axis) {
    throw Error("ReduceOp not implemented");
  }
  contiguous() {
    throw Error("ReduceOp not implemented");
  }
  toString() {
    throw Error("toString not implemented.");
  }
  // TODO: Check for compatibility between data and shape
  static computeShape(data, shape = []) {
    if (!data.length || data.length == 0) {
      return shape;
    }
    shape.push(data.length);
    return TensorBuffer.computeShape(data[0], shape);
  }
  static computeStrides(shape) {
    let strides = new Array(shape.length);
    strides[strides.length - 1] = 1;
    for (let i = strides.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }
  getInternalData() {
    throw Error("Not implemented");
  }
  get(i) {
    const data = this.getInternalData();
    let idx = 0;
    let totalSize = this.shape.reduce((a, b) => a * b, 1);
    for (let dim = 0; dim < this.shape.length; ++dim) {
      totalSize /= this.shape[dim];
      const coord = Math.floor(i / totalSize);
      i -= coord * totalSize;
      idx += this.strides[dim] * coord;
    }
    return data[(idx + this.offset) % data.length];
  }
  getValue(data, indices) {
    let index = 0;
    for (let i = 0; i < indices.length; i++) {
      index += this.strides[i] * indices[i];
    }
    return data[(index + this.offset) % data.length];
  }
  getNestedData(data, indices = []) {
    if (indices.length === this.shape.length) {
      return this.getValue(data, indices);
    }
    let dimSize = this.shape[indices.length];
    let nestedData = new Array(dimSize);
    for (let i = 0; i < dimSize; i++) {
      nestedData[i] = this.getNestedData(data, [...indices, i]);
    }
    return nestedData;
  }
  getData() {
    const data = this.getInternalData();
    const finalData = this.getNestedData(data);
    return finalData instanceof Array ? finalData : [finalData];
  }
  is_contiguous() {
    let expected_stride = 1;
    for (let i = this.strides.length - 1; i >= 0; i--) {
      if (this.strides[i] !== expected_stride) {
        return false;
      }
      expected_stride *= this.shape[i];
    }
    return true;
  }
};

// src/backend/UnaryOps.ts
var UnaryOp = class {
  static unary_op(x, op) {
    return x.unary_op(op);
  }
  static abs(x) {
    return UnaryOp.unary_op(x, 0 /* ABS */);
  }
  static exp(x) {
    return UnaryOp.unary_op(x, 1 /* EXP */);
  }
  static tanh(x) {
    return UnaryOp.unary_op(x, 2 /* TANH */);
  }
  static log(x) {
    return UnaryOp.unary_op(x, 3 /* LOG */);
  }
};

// src/backend/MovementOps.ts
var MovementOp = class {
  // TODO: Fix this, should be an op along from FROM, CONTIGUOUS etc
  // This is slow, there are more efficient ways to make it contiguous like updating the data as needed instead of the whole thing
  static contiguous(x) {
    return x.contiguous();
  }
  // TODO: Error handling
  static reshape(x, shape) {
    if (!(shape instanceof Array))
      shape = [shape];
    const totalSize = x.shape.reduce((acc, val) => acc * val, 1);
    const missingSize = shape.reduce((acc, val) => val >= 0 ? acc * val : acc, 1);
    const inferredShape = shape.map((val) => val === -1 ? totalSize / missingSize : val);
    if (!x.is_contiguous()) {
      return MovementOp.reshape(MovementOp.contiguous(x), inferredShape);
    }
    return Backend.CreateFromDataShapeAndStrides(x, inferredShape, TensorBuffer.computeStrides(inferredShape));
  }
  static expand(x, shape) {
    const lenDiff = shape.length - x.shape.length;
    let oldShape = x.shape;
    let oldStrides = x.strides;
    if (lenDiff > 0) {
      oldShape = Array(lenDiff).fill(1).concat(x.shape);
      oldStrides = Array(lenDiff).fill(0).concat(x.strides);
    }
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] == -1) {
        if (i >= oldShape.length) {
          throw new Error("Cannot infer dimension for expansion");
        }
        shape[i] = oldShape[i];
      }
    }
    let newStrides = new Array(shape.length).fill(0);
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] == oldShape[i]) {
        newStrides[i] = oldStrides[i];
      } else if (oldShape[i] == 1) {
        newStrides[i] = 0;
      }
    }
    return Backend.CreateFromDataShapeAndStrides(x, shape, newStrides);
  }
  static permute(m, axes = null) {
    if (axes === null) {
      return Backend.CreateFromDataShapeAndStrides(m, [...m.shape].reverse(), [...m.strides].reverse());
    }
    let newShape = [];
    let newStrides = [];
    for (let i = 0; i < axes.length; i++) {
      let axis = axes[i] < 0 ? m.shape.length + axes[i] : axes[i];
      newShape[i] = m.shape[axis];
      newStrides[i] = m.strides[axis];
    }
    return Backend.CreateFromDataShapeAndStrides(m, newShape, newStrides);
  }
  static transpose(m, dim0, dim1) {
    dim0 = dim0 < 0 ? m.shape.length + dim0 : dim0;
    dim1 = dim1 < 0 ? m.shape.length + dim1 : dim1;
    if (dim0 >= m.shape.length || dim1 >= m.shape.length) {
      throw new Error("Transpose dimensions out of range");
    }
    let axes = [];
    for (let i = 0; i < m.shape.length; i++) {
      axes[i] = i;
    }
    let tmp = axes[dim0];
    axes[dim0] = axes[dim1];
    axes[dim1] = tmp;
    return MovementOp.permute(m, axes);
  }
};

// src/backend/BinaryOps.ts
var BinaryOp = class {
  static binary_op(x, y, op) {
    if (x.device !== y.device) {
      throw Error(`Cannot perform binary op since TensorBuffer 1 is on ${Device[x.device]} and TensorBuffer 2 is on ${Device[y.device]}`);
    }
    if (!x.is_contiguous())
      x = MovementOp.contiguous(x);
    if (!y.is_contiguous())
      y = MovementOp.contiguous(y);
    return x.binary_op(y, op);
  }
  static add(x, y) {
    return BinaryOp.binary_op(x, y, 0 /* ADD */);
  }
  static sub(x, y) {
    return BinaryOp.binary_op(x, y, 1 /* SUB */);
  }
  static mul(x, y) {
    return BinaryOp.binary_op(x, y, 2 /* MUL */);
  }
  static div(x, y) {
    return BinaryOp.binary_op(x, y, 3 /* DIV */);
  }
  static pow(x, y) {
    return BinaryOp.binary_op(x, y, 4 /* POW */);
  }
  static equal(x, y) {
    return BinaryOp.binary_op(x, y, 5 /* CMPEQ */);
  }
  static maximum(x, y) {
    return BinaryOp.binary_op(x, y, 6 /* MAX */);
  }
};

// src/backend/ReduceOps.ts
var ReduceOp = class {
  static parseAxisParam(axis, shape) {
    const rank = shape.length;
    axis = axis == null ? shape.map((s, i) => i) : [].concat(axis);
    return axis.map((a) => a < 0 ? rank + a : a);
  }
  static getAxesPermutation(axes, rank) {
    const result = [];
    for (let i = 0; i < rank; ++i) {
      if (axes.indexOf(i) === -1) {
        result.push(i);
      }
    }
    axes.forEach((axis) => result.push(axis));
    return result;
  }
  static getInnerMostAxes(numAxes, rank) {
    const res = [];
    for (let i = rank - numAxes; i < rank; ++i) {
      res.push(i);
    }
    return res;
  }
  static combineLocations(outputLoc, reduceLoc, axes) {
    const rank = outputLoc.length + reduceLoc.length;
    const loc = [];
    let outIdx = 0;
    let reduceIdx = 0;
    for (let dim = 0; dim < rank; dim++) {
      if (axes.indexOf(dim) === -1) {
        loc.push(outputLoc[outIdx++]);
      } else {
        loc.push(reduceLoc[reduceIdx++]);
      }
    }
    return loc;
  }
  static expandShapeToKeepDim(shape, axes) {
    const reduceSubShape = axes.map((x) => 1);
    return ReduceOp.combineLocations(shape, reduceSubShape, axes);
  }
  static reduce_op(x, op, axis, keepdim) {
    const origAxes = ReduceOp.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = ReduceOp.getAxesPermutation(axes, x.shape.length);
    const inputIsTransposed = permutedAxes !== null;
    let input = x;
    if (inputIsTransposed) {
      input = MovementOp.permute(input, permutedAxes);
      axes = ReduceOp.getInnerMostAxes(axes.length, x.shape.length);
      input = MovementOp.contiguous(input);
    }
    const r = input.reduce_op(op, axes, x.shape, []);
    if (keepdim) {
      const s = r.shape;
      let shape = s.length === 1 && s[0] === 1 ? [] : s;
      let newShape = ReduceOp.expandShapeToKeepDim(shape, origAxes);
      if (newShape.length === 1 && newShape[0] === void 0)
        newShape = [s.reduce((p, c) => p * c)];
      return MovementOp.reshape(r, newShape);
    }
    return r;
  }
  static sum(x, axis, keepdim) {
    return ReduceOp.reduce_op(x, 0 /* SUM */, axis, keepdim);
  }
  static prod(x, axis, keepdim) {
    return ReduceOp.reduce_op(x, 1 /* PROD */, axis, keepdim);
  }
};

// src/backend/CPU.ts
var _CPUBuffer = class extends TensorBuffer {
  constructor(data, shape, strides, offset) {
    super(shape, strides, offset, 0 /* CPU */);
    if (data instanceof Float32Array)
      this.data = data.slice();
    else if (data instanceof _CPUBuffer)
      this.data = data.data.slice();
  }
  static CreateFromArray(array) {
    const data = Float32Array.from(array.flat(Infinity));
    const shape = TensorBuffer.computeShape(array);
    const strides = TensorBuffer.computeStrides(shape);
    return new _CPUBuffer(data, shape, strides, 0);
  }
  static CreateFromFloat32Array(array, shape = [], strides = [], offset = 0) {
    const _shape = shape.length === 0 ? [array.length] : shape;
    const _strides = strides.length === 0 ? this.computeStrides(_shape) : strides;
    return new _CPUBuffer(array, _shape, _strides, offset);
  }
  static CreateFromNumber(num) {
    return new _CPUBuffer(new Float32Array([num]), [1], [1], 0);
  }
  getInternalData() {
    return this.data;
  }
  copy() {
    return new _CPUBuffer(this.data.slice(), this.shape, this.strides, this.offset);
  }
  unary_op(op) {
    if (op === 0 /* ABS */)
      return new _CPUBuffer(new Float32Array(this.data.map((v) => Math.abs(v))), this.shape, this.strides, this.offset);
    else if (op === 1 /* EXP */)
      return new _CPUBuffer(new Float32Array(this.data.map((v) => Math.exp(v))), this.shape, this.strides, this.offset);
    else if (op === 2 /* TANH */)
      return new _CPUBuffer(new Float32Array(this.data.map((v) => Math.tanh(v))), this.shape, this.strides, this.offset);
    else if (op === 3 /* LOG */)
      return new _CPUBuffer(new Float32Array(this.data.map((v) => Math.log(v))), this.shape, this.strides, this.offset);
  }
  binary_op(other, op) {
    let [_m1b, _m2b] = [this, other];
    let newData = new Float32Array(_m1b.shape.reduce((p, c) => p * c));
    for (let i = 0; i < newData.length; i++) {
      const v1 = _m1b.get(i);
      const v2 = _m2b.get(i);
      let value = 0;
      if (op == 0 /* ADD */)
        value = v1 + v2;
      else if (op == 1 /* SUB */)
        value = v1 - v2;
      else if (op == 2 /* MUL */)
        value = v1 * v2;
      else if (op == 3 /* DIV */)
        value = v1 / v2;
      else if (op == 4 /* POW */)
        value = v1 ** v2;
      else if (op == 5 /* CMPEQ */)
        value = Math.abs(v1 - v2) < _CPUBuffer.EPS === true ? 1 : 0;
      else if (op == 6 /* MAX */)
        value = v1 > v2 ? v1 : v2;
      newData[i] = isNaN(value) ? 0 : value;
    }
    const shape = _m1b.shape.slice();
    return new _CPUBuffer(newData, shape, TensorBuffer.computeStrides(shape), _m1b.offset);
  }
  reduce_op(op, axes) {
    function computeOutAndReduceShapes(aShape, axes2) {
      const outShape2 = [];
      const rank = aShape.length;
      for (let dim = 0; dim < rank; dim++) {
        if (axes2.indexOf(dim) === -1) {
          outShape2.push(aShape[dim]);
        }
      }
      const reduceShape2 = axes2.map((dim) => aShape[dim]);
      return [outShape2, reduceShape2];
    }
    let [outShape, reduceShape] = computeOutAndReduceShapes(this.shape, axes);
    outShape = outShape.length === 0 ? [1] : outShape;
    const sp = outShape.reduce((p, c) => p * c);
    let output = new Float32Array(sp);
    if (op === 1 /* PROD */)
      output.fill(1);
    const vals = reduceShape.reduce((p, c) => p * c);
    let additionCounter = 0;
    const length = this.shape.reduce((p, c) => p * c);
    for (let i = 0; i < length; i++) {
      for (let index = 0; index < vals; index++) {
        if (op === 0 /* SUM */) {
          output[additionCounter] += this.get(i);
        } else if (op === 1 /* PROD */) {
          output[additionCounter] *= this.get(i);
        }
        i++;
      }
      additionCounter++;
      i--;
    }
    return new _CPUBuffer(output, outShape, TensorBuffer.computeStrides(outShape), 0);
  }
  contiguous() {
    const r = _CPUBuffer.CreateFromArray(this.getData());
    return r;
  }
  toString() {
    function fixed(key, val) {
      return val.toFixed ? Number(val.toFixed(4)) : val;
    }
    return `${JSON.stringify(this.getData(), fixed)}`;
  }
};
var CPUBuffer = _CPUBuffer;
CPUBuffer.EPS = 1e-5;

// src/backend/WebGL.ts
function equalArrays(a, b) {
  if (a.length !== b.length)
    return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i])
      return false;
  }
  return true;
}
var Texture = class {
  constructor(data, info) {
    if (data === void 0)
      throw Error("Got undefined data");
    this.data = data;
    this.width = info.width;
    this.height = info.height;
    this.internalFormat = info.internalFormat;
    this.format = info.format;
    this.type = info.type;
    this.originalShape = info.originalShape;
    const gl = WEBGLContext.gl;
    if (data instanceof Texture) {
      if (!data.texture)
        throw Error("Passed texture but no data.texture found");
      gl.bindTexture(gl.TEXTURE_2D, data.texture);
      this.texture = data.texture;
    } else {
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
  toString() {
    return `Texture(
            data=[${this.data}],
            width=${this.width},
            height=${this.height},
            internalFormat=${WEBGLContext.glCodeToStr(this.internalFormat)}
            format=${WEBGLContext.glCodeToStr(this.format)},
            type=${WEBGLContext.glCodeToStr(this.type)}
        )`;
  }
  decodeMatrixFromUnpackedArray(unpackedArray, matrix, channelsPerTexture) {
    function getMatrixSizeFromUnpackedArraySize(unpackedSize, channelsPerTexture2) {
      if (unpackedSize % channelsPerTexture2 !== 0) {
        throw new Error(`unpackedSize (${unpackedSize}) must be a multiple of ${channelsPerTexture2}`);
      }
      return unpackedSize / channelsPerTexture2;
    }
    const requiredSize = getMatrixSizeFromUnpackedArraySize(unpackedArray.length, channelsPerTexture);
    if (matrix.length < requiredSize)
      throw new Error(`matrix length (${matrix.length}) must be >= ${requiredSize}`);
    let dst = 0;
    for (let src = 0; src < unpackedArray.length; src += channelsPerTexture) {
      matrix[dst++] = unpackedArray[src];
    }
  }
  read() {
    if (this.data)
      return this.data;
    this.data = WEBGLContext.readTextureData(this);
    return this.data;
  }
  static shapeTo2d(shape) {
    let shape2d = [shape[0], 1];
    for (let i = 1; i < shape.length; i++) {
      shape2d[1] *= shape[i];
    }
    return shape2d;
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
  static calculateWidthAndHeightToFitShape(shape, channels) {
    const shape2D = Texture.shapeTo2d(shape);
    const prodShape = shape2D.reduce((p, c) => p * c);
    let width = Math.ceil(Math.sqrt(prodShape / channels));
    let height = Math.floor(Math.sqrt(prodShape / channels));
    while (width * height * channels < prodShape) {
      height++;
      if (width * height * channels < prodShape)
        width++;
    }
    return [width, height];
  }
  static createUnpackedFromShape(data, shape) {
    const gl = WEBGLContext.gl;
    const [width, height] = Texture.calculateWidthAndHeightToFitShape(shape, 1);
    if (data && data.length < width * height) {
      const _data = new Float32Array(width * height);
      _data.set(data);
      data = _data;
    }
    return new Texture(data, { width, height, internalFormat: gl.R32F, format: gl.RED, type: gl.FLOAT, originalShape: shape });
  }
  static createUnpackedFromDimensions(data, width, height) {
    const gl = WEBGLContext.gl;
    return new Texture(data, { width, height, internalFormat: gl.R32F, format: gl.RED, type: gl.FLOAT, originalShape: [width, height] });
  }
};
var _WEBGLCache = class {
  static hashCode(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) {
      h = Math.imul(31, h) + s.charCodeAt(i) | 0;
    }
    return h;
  }
  static getShader(code) {
    const key = _WEBGLCache.hashCode(code);
    return _WEBGLCache.shaderCache.has(key) ? _WEBGLCache.shaderCache.get(key).shader : null;
  }
  static setShader(code, shader) {
    const key = _WEBGLCache.hashCode(code);
    _WEBGLCache.shaderCache.set(key, {
      key,
      code,
      shader
    });
  }
  static hasShader(code) {
    const key = _WEBGLCache.hashCode(code);
    return _WEBGLCache.shaderCache.has(key);
  }
  static getProgram(fragmentCode) {
    const key = _WEBGLCache.hashCode(fragmentCode);
    return _WEBGLCache.programCache.has(key) ? _WEBGLCache.programCache.get(key) : null;
  }
  static setProgram(fragmentCode, program) {
    const key = _WEBGLCache.hashCode(fragmentCode);
    _WEBGLCache.programCache.set(key, program);
  }
  static hasProgram(code) {
    const key = _WEBGLCache.hashCode(code);
    return _WEBGLCache.programCache.has(key);
  }
};
var WEBGLCache = _WEBGLCache;
WEBGLCache.shaderCache = /* @__PURE__ */ new Map();
WEBGLCache.programCache = /* @__PURE__ */ new Map();
var _WEBGLContext = class {
  static get gl() {
    if (!_WEBGLContext._gl)
      _WEBGLContext._gl = _WEBGLContext.setup();
    return _WEBGLContext._gl;
  }
  constructor() {
    throw Error("Cannot call WEBGLContext with new.");
  }
  static hashCode(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++)
      h = Math.imul(31, h) + s.charCodeAt(i) | 0;
    return h;
  }
  static setup() {
    if (typeof window === "undefined")
      throw Error("Window not found, WebGL2 is only supported in browsers.");
    if (typeof document === "undefined")
      throw Error("Document not found, WebGL2 is only supported in browsers.");
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2");
    if (!gl)
      throw Error("Could not setup WebGL2");
    if (!gl.getExtension("EXT_color_buffer_float"))
      throw Error("EXT_color_buffer_float not supported");
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    document.body.appendChild(canvas);
    return gl;
  }
  static compileShader(gl, shaderType, shaderCode) {
    const shader = gl.createShader(shaderType);
    gl.shaderSource(shader, shaderCode);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.log(`${shaderCode}`);
      throw new Error("Could not compile shader: " + gl.getShaderInfoLog(shader));
    }
    return shader;
  }
  static createShaderProgram(gl, vertexShader, fragmentShader) {
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
  static createQuad(gl, program) {
    const vertex_data = new Float32Array([-1, 1, 0, 0, 1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, 0]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertex_data, gl.STATIC_DRAW);
    const posOffset = 0;
    const uvOffset = 3 * 4;
    const stride = 3 * 4 + 2 * 4;
    const clipSpacePosLoc = gl.getAttribLocation(program, "clipSpacePos");
    gl.vertexAttribPointer(clipSpacePosLoc, 3, gl.FLOAT, false, stride, posOffset);
    gl.enableVertexAttribArray(clipSpacePosLoc);
    const uvLoc = gl.getAttribLocation(program, "uv");
    gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, stride, uvOffset);
    gl.enableVertexAttribArray(uvLoc);
    return vertexBuffer;
  }
  static bindTexture(gl, texture, program, location) {
    gl.activeTexture(gl.TEXTURE0 + location);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const textureLocation = gl.getUniformLocation(program, `u_tex${location}`);
    gl.uniform1i(textureLocation, location);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.uniform1i(textureLocation, location);
    return textureLocation;
  }
  static processUniforms(gl, program, uniforms) {
    if (!uniforms)
      return;
    for (let i = 0; i < uniforms.length; i++) {
      const name = uniforms[i].name;
      const value = uniforms[i].value;
      const type = uniforms[i].type;
      const location = gl.getUniformLocation(program, name);
      if (location === null)
        throw Error(`Got null uniform location ${name} ${value}`);
      if (value instanceof Array) {
        if (type === 8 /* FLOAT_ARRAY */)
          gl.uniform1fv(location, value);
        else if (type === 9 /* INT_ARRAY */)
          gl.uniform1iv(location, value);
        else if (type === 1 /* FLOAT_VEC2 */)
          gl.uniform2fv(location, value);
        else if (type === 2 /* FLOAT_VEC3 */)
          gl.uniform3fv(location, value);
        else if (type === 3 /* FLOAT_VEC4 */)
          gl.uniform4fv(location, value);
        else if (type === 5 /* INT_VEC2 */)
          gl.uniform2iv(location, value);
        else if (type === 6 /* INT_VEC3 */)
          gl.uniform3iv(location, value);
        else if (type === 7 /* INT_VEC4 */)
          gl.uniform4iv(location, value);
      } else {
        if (type === 0 /* FLOAT */)
          gl.uniform1f(location, value);
        if (type === 4 /* INT */)
          gl.uniform1i(location, value);
      }
    }
  }
  static runKernel(shader, inputs, output, uniforms) {
    if (inputs.length === 0)
      throw Error("Cannot run kernel without any buffers.");
    const gl = _WEBGLContext.gl;
    let vertexShader = WEBGLCache.getShader(_WEBGLContext.defaultVertexShader);
    let fragmentShader = WEBGLCache.getShader(shader);
    if (vertexShader === null) {
      vertexShader = _WEBGLContext.compileShader(gl, gl.VERTEX_SHADER, _WEBGLContext.defaultVertexShader);
      WEBGLCache.setShader(_WEBGLContext.defaultVertexShader, vertexShader);
    }
    if (fragmentShader === null) {
      fragmentShader = _WEBGLContext.compileShader(gl, gl.FRAGMENT_SHADER, shader);
      WEBGLCache.setShader(shader, fragmentShader);
    }
    let shaderProgram = WEBGLCache.getProgram(shader);
    if (shaderProgram === null) {
      shaderProgram = _WEBGLContext.createShaderProgram(gl, vertexShader, fragmentShader);
      WEBGLCache.setProgram(shader, shaderProgram);
    }
    const quadVertexBuffer = _WEBGLContext.createQuad(gl, shaderProgram);
    gl.useProgram(shaderProgram);
    _WEBGLContext.processUniforms(gl, shaderProgram, uniforms);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.viewport(0, 0, output.width, output.height);
    _WEBGLContext.bindTexture(gl, output.texture, shaderProgram, 0);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, output.texture, 0);
    for (let i = 0; i < inputs.length; i++) {
      const bufferLocation = _WEBGLContext.bindTexture(gl, inputs[i].texture, shaderProgram, i);
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  static readTextureData(texture) {
    const gl = _WEBGLContext.gl;
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture.texture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
      throw Error("Framebuffer incomplete");
    const data = new Float32Array(texture.width * texture.height * 1);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, texture.width, texture.height, gl.RED, gl.FLOAT, data);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(framebuffer);
    return data;
  }
  static glCodeToStr(code) {
    if (!_WEBGLContext.glCodesTable) {
      _WEBGLContext.glCodesTable = {};
      for (var key in _WEBGLContext.gl) {
        _WEBGLContext.glCodesTable[_WEBGLContext.gl[key]] = key;
      }
    }
    return "gl." + _WEBGLContext.glCodesTable[code];
  }
};
var WEBGLContext = _WEBGLContext;
WEBGLContext.defaultVertexShader = `#version 300 es
    precision highp float;
    in vec3 clipSpacePos;
    in vec2 uv;
    out vec2 resultUV;

    void main() {
        gl_Position = vec4(clipSpacePos, 1);
        resultUV = uv;
    }`;
var WEBGLBuffer = class extends TensorBuffer {
  constructor(data, shape, strides, offset) {
    if (!data || data === null)
      throw Error("Cannot create buffer with no data");
    super(shape, strides, offset, 1 /* WEBGL */);
    if (data instanceof Float32Array)
      this.data = data;
    if (data instanceof Texture)
      this.texture = data;
    if (data instanceof WEBGLBuffer) {
      if (!data.data && !data.texture)
        throw Error("Tried to create WEBGLBuffer with no data or texture.");
      if (data.data)
        this.data = data.data;
      else if (data.texture)
        this.texture = data.texture;
    }
    try {
      throw Error("creator");
    } catch (error) {
      this.creator = error;
    }
  }
  static CreateFromArray(array) {
    const data = Float32Array.from(array.flat(Infinity));
    const shape = TensorBuffer.computeShape(array);
    const strides = TensorBuffer.computeStrides(shape);
    return new WEBGLBuffer(data, shape, strides, 0);
  }
  static CreateFromFloat32Array(data, shape = [], strides = [], offset = 0) {
    const _shape = shape.length === 0 ? [data.length] : shape;
    const _strides = strides.length === 0 ? this.computeStrides(_shape) : strides;
    return new WEBGLBuffer(data, _shape, _strides, offset);
  }
  static CreateFromNumber(num) {
    return new WEBGLBuffer(new Float32Array([num]), [1], [1], 0);
  }
  getInternalData() {
    return this.data ? this.data : this.texture.read();
  }
  createUnpackedTexture() {
    if (!this.data && !this.texture)
      throw Error("Tried to create unpacked texture without a data or texture field");
    if (this.texture) {
      if (!equalArrays(this.shape, this.texture.originalShape)) {
        this.texture = this.copyToShape(this.shape).texture;
      }
      return this.texture;
    } else if (this.data) {
      this.texture = Texture.createUnpackedFromShape(this.data, this.shape);
    }
    return this.texture;
  }
  createUnpackedTextureFromDimensions(width, height) {
    if (!this.data && !this.texture)
      throw Error("Tried to create unpacked texture without a data or texture field");
    if (this.texture) {
      if (this.texture.data) {
        this.texture = Texture.createUnpackedFromDimensions(this.texture.data, width, height);
      }
      return this.texture;
    } else if (this.data) {
      this.texture = Texture.createUnpackedFromDimensions(this.data, width, height);
    }
    return this.texture;
  }
  unary_op(op) {
    function processOp(op2) {
      if (op2 === 0 /* ABS */)
        return "abs(t1)";
      else if (op2 === 1 /* EXP */)
        return "exp(t1)";
      else if (op2 === 2 /* TANH */)
        return "tanh(t1)";
      else if (op2 === 3 /* LOG */)
        return "log(t1)";
    }
    const inputTexture = this.createUnpackedTexture();
    const outputTexture = Texture.createUnpackedFromShape(null, this.shape);
    WEBGLContext.runKernel(`#version 300 es
        precision mediump float;
        
        uniform sampler2D u_tex0;

        out vec4 result;
        
        void main() {
            ivec2 coords = ivec2(gl_FragCoord.xy);
            vec4 t1 = texelFetch(u_tex0, coords, 0);

            result = ${processOp(op)};
        }`, [inputTexture], outputTexture);
    return new WEBGLBuffer(outputTexture, this.shape, this.strides, this.offset);
  }
  binary_op(other, op) {
    function processOp(op2) {
      if (op2 === 0 /* ADD */)
        return "t1 + t2";
      else if (op2 === 1 /* SUB */)
        return "t1 - t2";
      else if (op2 === 2 /* MUL */)
        return "t1 * t2";
      else if (op2 === 3 /* DIV */)
        return "t1 / t2";
      else if (op2 === 4 /* POW */)
        return "myPow(t1, t2)";
      else if (op2 === 5 /* CMPEQ */)
        return "vec4(t1.r == t2.r, t1.g == t2.g, t1.b == t2.b, t1.a == t2.a)";
      else if (op2 === 6 /* MAX */)
        return "max(t1, t2)";
    }
    const inputTextureX = this.createUnpackedTexture();
    const inputTextureY = other.createUnpackedTexture();
    const outputTexture = Texture.createUnpackedFromShape(null, this.shape);
    WEBGLContext.runKernel(`#version 300 es
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
  reduce_op(op, axes) {
    function prod(array) {
      return array.reduce((p, c) => p * c);
    }
    const axisLength = axes.length === this.shape.length ? prod(this.shape) : this.shape[this.shape.length - 1];
    function sumDim(input, shape, stride2) {
      const outputTexture2 = Texture.createUnpackedFromShape(null, shape);
      const uniforms2 = [
        { name: "width", value: outputTexture2.width, type: 4 /* INT */ },
        { name: "u_stride", value: stride2, type: 4 /* INT */ },
        { name: "u_axisLength", value: axisLength, type: 4 /* INT */ },
        { name: "u_op", value: op, type: 4 /* INT */ }
      ];
      WEBGLContext.runKernel(`#version 300 es
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
            }`, [input], outputTexture2, uniforms2);
      return outputTexture2;
    }
    const inputTexture = this.createUnpackedTexture();
    let outputTexture = inputTexture;
    let stride = 1;
    const totalNumberOfElements = prod(this.shape);
    while (stride < totalNumberOfElements) {
      outputTexture = sumDim(outputTexture, this.shape, stride);
      stride *= 2;
    }
    const outputTexturePacked = Texture.createUnpackedFromShape(null, this.shape);
    const uniforms = [
      { name: "width", value: outputTexturePacked.width, type: 4 /* INT */ },
      { name: "u_axisLength", value: axisLength, type: 4 /* INT */ }
    ];
    WEBGLContext.runKernel(`#version 300 es
        precision mediump float;

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
    function calculateReducedShape(originalShape, axes2, keepdim = false) {
      if (!keepdim) {
        return originalShape.filter((_, index) => !axes2.includes(index));
      } else {
        return originalShape.map((dim, index) => axes2.includes(index) ? 1 : dim);
      }
    }
    let resultShape = calculateReducedShape(this.shape, axes, false);
    resultShape = resultShape.length === 0 ? resultShape = [1] : resultShape;
    const r = new WEBGLBuffer(outputTexturePacked, outputTexture.originalShape, TensorBuffer.computeStrides(outputTexture.originalShape), this.offset);
    return MovementOp.reshape(r, resultShape);
  }
  contiguous() {
    const inputTexture = this.createUnpackedTexture();
    const outputTexture = Texture.createUnpackedFromShape(null, this.shape);
    const MAX_DIMS = 10;
    if (this.shape.length !== this.strides.length)
      throw Error("Shape does not match strides");
    if (this.shape.length > MAX_DIMS)
      throw Error(`Maximum dimensions for contiguous call are 10, got ${this.shape.length}`);
    const uniforms = [
      // {name: "MAX_DIMS", value: MAX_DIMS, type: UniformType.INT},
      { name: "width", value: inputTexture.width, type: 4 /* INT */ },
      { name: "shape", value: this.shape, type: 9 /* INT_ARRAY */ },
      { name: "strides", value: this.strides, type: 9 /* INT_ARRAY */ },
      { name: "offset", value: this.offset, type: 4 /* INT */ },
      { name: "shapeLength", value: this.shape.length, type: 4 /* INT */ }
    ];
    WEBGLContext.runKernel(`#version 300 es
        precision mediump float;

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
  copyToShape(shape) {
    const inputTexture = this.texture;
    const outputTexture = Texture.createUnpackedFromShape(null, shape.slice());
    const uniforms = [
      { name: "widthIn", value: inputTexture.width, type: 4 /* INT */ },
      { name: "widthOut", value: outputTexture.width, type: 4 /* INT */ }
    ];
    WEBGLContext.runKernel(`#version 300 es
        precision mediump float;
        
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
  toString() {
    function fixed(key, val) {
      return val.toFixed ? Number(val.toFixed(4)) : val;
    }
    return JSON.stringify(this.getData(), fixed);
  }
  copy() {
    throw Error("Not implemented");
  }
};

// src/backend/Backend.ts
var Device = /* @__PURE__ */ ((Device5) => {
  Device5[Device5["CPU"] = 0] = "CPU";
  Device5[Device5["WEBGL"] = 1] = "WEBGL";
  return Device5;
})(Device || {});
var Backend = class {
  static CreateFromArray(device, array) {
    if (device === 0 /* CPU */)
      return CPUBuffer.CreateFromArray(array);
    else if (device === 1 /* WEBGL */)
      return WEBGLBuffer.CreateFromArray(array);
    throw Error(`Unable to call CreateFromArray for device ${Device[device]}`);
  }
  static CreateFromFloat32Array(device, array, shape = [], strides = [], offset = 0) {
    if (device === 0 /* CPU */)
      return CPUBuffer.CreateFromFloat32Array(array, shape, strides, offset);
    else if (device === 1 /* WEBGL */)
      return WEBGLBuffer.CreateFromFloat32Array(array, shape, strides, offset);
    throw Error(`Unable to call CreateFromFloat32Array for device ${Device[device]}`);
  }
  static CreateFromNumber(device, num) {
    if (device === 0 /* CPU */)
      return CPUBuffer.CreateFromNumber(num);
    else if (device === 1 /* WEBGL */)
      return WEBGLBuffer.CreateFromNumber(num);
    throw Error(`Unable to call CreateFromNumber for device ${Device[device]}`);
  }
  static CreateFromDataShapeAndStrides(data, shape, strides, offset = 0) {
    if (data instanceof CPUBuffer)
      return new CPUBuffer(data, shape, strides, offset);
    else if (data instanceof WEBGLBuffer)
      return new WEBGLBuffer(data, shape, strides, offset);
    throw Error(`Unable to call CreateFromDataShapeAndStrides`);
  }
};

// src/Tensor.ts
var DefaultTensorOptions = {
  _children: [],
  _op: null,
  device: 0 /* CPU */,
  requires_grad: false
};
var Tensor = class {
  get shape() {
    return this.data.shape;
  }
  get strides() {
    return this.data.strides;
  }
  get offset() {
    return this.data.offset;
  }
  constructor(data, options) {
    this.id = "P" + Math.floor(Math.random() * 1e6).toString().padStart(6, "0");
    const _options = Object.assign({}, DefaultTensorOptions, options);
    if (_options._children.length !== 0) {
      _options.requires_grad = _options._children[0].requires_grad;
    }
    if (data instanceof Tensor) {
      this.data = data.data;
      _options.device = options && options.device ? options.device : data.device;
    } else if (data instanceof TensorBuffer) {
      this.data = data;
      _options.device = options && options.device ? options.device : data.device;
    } else if (data instanceof Array)
      this.data = Backend.CreateFromArray(_options.device, data);
    else if (data instanceof Float32Array)
      this.data = Backend.CreateFromFloat32Array(_options.device, data);
    else if (!isNaN(data))
      this.data = Backend.CreateFromNumber(_options.device, data);
    this.grad = Backend.CreateFromFloat32Array(_options.device, new Float32Array([0]), this.shape, TensorBuffer.computeStrides(this.shape));
    this.device = _options.device;
    this.requires_grad = _options.requires_grad;
    this._op = _options._op;
    this._prev = new Set(_options._children);
    this._children = _options._children;
    this.options = _options;
  }
  backward() {
    let topo = [];
    let visited = /* @__PURE__ */ new Set();
    function build_topo(v) {
      if (!visited.has(v)) {
        visited.add(v);
        for (let child of v._prev) {
          build_topo(child);
        }
        topo.push(v);
      }
    }
    build_topo(this);
    this.grad = Tensor.ones(this.data.shape, { device: this.device }).data;
    for (let v of topo.reverse()) {
      if (v._op === null)
        continue;
      const grads = v._op.backward(v.grad);
      if (grads) {
        for (let i = 0; i < grads.length; i++) {
          if (grads[i] !== null) {
            if (v._children[i].grad) {
              v._children[i].grad = BinaryOp.add(v._children[i].grad, grads[i]);
            } else {
              v._children[i].grad = grads[i];
            }
          }
        }
      }
    }
  }
  // Helpers
  // TODO: Should use strides and shape to do this
  static TensorFromNumber(value, shape, options) {
    const desiredElements = shape.reduce((acc, val) => acc * val, 1);
    const device = options && options.device ? options.device : 0 /* CPU */;
    if (value === 0)
      return new Tensor(Backend.CreateFromFloat32Array(device, new Float32Array(desiredElements), shape), options);
    return new Tensor(Backend.CreateFromFloat32Array(device, new Float32Array(desiredElements).fill(value), shape), options);
  }
  static zeros(size, options) {
    return Tensor.TensorFromNumber(0, size, options);
  }
  static ones(size, options) {
    return Tensor.TensorFromNumber(1, size, options);
  }
  static full(size, value, options) {
    return Tensor.TensorFromNumber(value, size, options);
  }
  static arange(start, stop, step = 1, options) {
    let data = new Float32Array(Math.floor((stop - start) / step));
    let s = 0;
    for (let i = start; i < stop; i += step) {
      data[s] = i;
      s++;
    }
    return new Tensor(data, options);
  }
  static rand(shape, options) {
    let data = new Float32Array(shape.reduce((prev, curr) => prev * curr));
    for (let i = 0; i < data.length; i++) {
      data[i] = Random.Random();
    }
    return new Tensor(data, options).reshape(shape);
  }
  static uniform(low, high, shape, options) {
    let data = new Float32Array(shape.reduce((prev, curr) => prev * curr));
    for (let i = 0; i < data.length; i++) {
      data[i] = Random.RandomRange(low, high);
    }
    const device = options.device ? options.device : 0 /* CPU */;
    const tb = Backend.CreateFromFloat32Array(device, data, shape);
    return new Tensor(tb, options);
  }
  is_contiguous() {
    return this.data.is_contiguous();
  }
  contiguous() {
    return new Tensor(MovementOp.contiguous(this.data), { device: this.device, requires_grad: this.requires_grad });
  }
  // Movement ops
  get T() {
    return this.permute();
  }
  expand(shape) {
    return new Operations_exports.Expand().forward(this, shape);
  }
  static equalArrays(first, second) {
    if (first.length != second.length)
      return false;
    for (let i = 0; i < first.length; i++) {
      if (first[i] !== second[i])
        return false;
    }
    return true;
  }
  broadcast(other) {
    const [x, y] = [this, other];
    if (Tensor.equalArrays(x.shape, y.shape))
      return [x, y];
    let finalShape = [];
    let maxLength = Math.max(x.shape.length, y.shape.length);
    for (let i = 0; i < maxLength; i++) {
      finalShape.push(Math.max(x.shape[x.shape.length - i - 1] || 1, y.shape[y.shape.length - i - 1] || 1));
    }
    finalShape = finalShape.reverse();
    return [
      x.expand(finalShape),
      y.expand(finalShape)
    ];
  }
  slice(indices) {
    if (indices.length != this.shape.length)
      throw Error(`Indices [${indices}] must match tensor shape ${this.shape}`);
    let offset = this.offset;
    let newShape = [];
    let newStrides = [];
    for (let i = 0; i < indices.length; i++) {
      const index = indices[i];
      if (Array.isArray(index)) {
        const [start, stop] = index;
        if (start < 0 || start >= this.shape[i] || stop < 0 || stop > this.shape[i]) {
          throw Error(`Slice ${start}:${stop} out of bounds for axis ${i} with size ${this.shape[i]}`);
        }
        offset += start * this.strides[i];
        newShape.push(stop - start);
        newStrides.push(this.strides[i]);
      } else if (index !== null) {
        if (index < 0 || index >= this.shape[i]) {
          throw Error(`Index ${index} out of bounds for axis ${i} with size ${this.shape[i]}`);
        }
        offset += index * this.strides[i];
      } else {
        newShape.push(this.shape[i]);
        newStrides.push(this.strides[i]);
      }
    }
    const tb = Backend.CreateFromDataShapeAndStrides(this.data, newShape, newStrides, offset);
    return new Tensor(tb, { device: this.device, requires_grad: this.requires_grad });
  }
  split(split_sizes, dim = null) {
    if (Array.isArray(split_sizes))
      throw Error("Split split_sizes as array not supported");
    if (dim !== null)
      throw Error("Split dim not supported");
    const chunkSize = split_sizes;
    const lastDim = this.shape[this.shape.length - 1];
    if (lastDim % chunkSize !== 0) {
      throw new Error("Invalid chunk size, not evenly divisible into last tensor dimension");
    }
    const numChunks = lastDim / chunkSize;
    const out = [];
    let start = 0;
    for (let i = 0; i < numChunks; i++) {
      let end = start + chunkSize;
      const sliceIndices = this.shape.map((dimSize, idx) => idx === this.shape.length - 1 ? [start, end] : [0, dimSize]);
      const chunk = this.slice(sliceIndices);
      out.push(chunk);
      start = end;
    }
    return out;
  }
  // UnaryOps
  abs() {
    return this.relu().add(this.mul(-1).relu());
  }
  log() {
    return new Operations_exports.Log().forward(this);
  }
  masked_fill(mask, value) {
    const [mb, maskb] = this.broadcast(mask);
    const fillTensor = Tensor.full(mb.shape, value, { device: this.device, requires_grad: this.requires_grad });
    const filled = Tensor.where(maskb, fillTensor, mb);
    return filled;
  }
  softmax(dim) {
    return this.exp().div(this.exp().sum(dim, true));
  }
  static _tri(r, c, k = 0, options) {
    let a = Tensor.arange(0, r, 1, options).unsqueeze(1).expand([r, c]);
    let b = Tensor.arange(-k, c - k, 1, options).unsqueeze(0).expand([r, c]);
    return a.lte(b);
  }
  triu(k = 0) {
    const a = Tensor._tri(this.shape[this.shape.length - 2], this.shape[this.shape.length - 1], k, { device: this.device });
    return Tensor.where(a, this, Tensor.zeros(this.shape, { device: this.device }));
  }
  tril(k = 0) {
    const a = Tensor._tri(this.shape[this.shape.length - 2], this.shape[this.shape.length - 1], k + 1, { device: this.device });
    return Tensor.where(a, Tensor.zeros(this.shape, { device: this.device }), this);
  }
  // BinaryOps
  add(other) {
    const otherTensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, { device: this.device, requires_grad: this.requires_grad });
    return new Operations_exports.Add().forward(...this.broadcast(otherTensor));
  }
  sub(other) {
    const otherTensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, { device: this.device, requires_grad: this.requires_grad });
    return this.add(otherTensor.mul(-1));
  }
  mul(other) {
    const otherTensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, { device: this.device, requires_grad: this.requires_grad });
    return new Operations_exports.Mul().forward(...this.broadcast(otherTensor));
  }
  div(other) {
    if (!(other instanceof Tensor))
      other = new Tensor(other, { device: this.device, requires_grad: this.requires_grad });
    return this.mul(other.pow(-1));
  }
  pow(other) {
    const otherTensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, { device: this.device, requires_grad: this.requires_grad });
    return new Operations_exports.Pow().forward(...this.broadcast(otherTensor));
  }
  sqrt() {
    return this.pow(0.5);
  }
  rsqrt() {
    return this.pow(-0.5);
  }
  _matmul(other) {
    const [m1, m2] = [this, other instanceof Tensor ? other : Tensor.full(this.shape, other)];
    const x = m1.reshape([...m1.shape.slice(0, m1.shape.length - 1), 1, ...m1.shape.slice(m1.shape.length - 1, m1.shape.length)]);
    let w = m2.reshape([...m2.shape.slice(0, m2.shape.length - 2), 1, ...m2.shape.slice(m2.shape.length - 2, m2.shape.length - 1), ...m2.shape.slice(m2.shape.length - 1, m2.shape.length)]);
    w = w.transpose(-1, -2);
    let r = x.mul(w);
    r = r.sum(-1);
    if (m1.shape.length == 1) {
      r = r.reshape([...r.shape.slice(0, r.shape.length - 3), ...r.shape.slice(r.shape.length - 2, r.shape.length - 1)]);
    }
    return r;
  }
  matmul(other) {
    const otherTensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, { device: this.device, requires_grad: this.requires_grad });
    return new Operations_exports.Matmul().forward(this, otherTensor);
  }
  sum(axis = null, keepdims = false) {
    return new Operations_exports.Sum().forward(this, axis, keepdims);
  }
  mean(axis = null, keepdims = false) {
    const out = this.sum(axis, keepdims);
    const outShapeLen = out.shape.reduce((p, c) => p * c);
    const thisShapeLen = this.shape.reduce((p, c) => p * c);
    return out.mul(outShapeLen / thisShapeLen);
  }
  var(axis = null, keepdims = false) {
    const x = this.sub(this.mean(axis, true)).abs().pow(2);
    return x.mean(axis, keepdims);
  }
  reshape(shape) {
    return new Operations_exports.Reshape().forward(this, shape);
  }
  exp() {
    return new Operations_exports.Exp().forward(this);
  }
  relu() {
    return new Operations_exports.Relu().forward(this);
  }
  reciprocal() {
    return Tensor.ones(this.data.shape, { device: this.device, requires_grad: this.requires_grad }).div(this);
  }
  sigmoid() {
    return Tensor.ones(this.data.shape, { device: this.device, requires_grad: this.requires_grad }).add(this.mul(-1).exp()).reciprocal();
  }
  tanh() {
    const one = new Tensor(1, { device: this.device, requires_grad: this.requires_grad });
    const two1 = new Tensor(2, { device: this.device, requires_grad: this.requires_grad });
    const two2 = new Tensor(2, { device: this.device, requires_grad: this.requires_grad });
    return two1.mul(two2.mul(this).sigmoid()).sub(one);
  }
  permute(axes = null) {
    return new Operations_exports.Permute().forward(this, axes);
  }
  transpose(dim0, dim1) {
    return new Operations_exports.Transpose().forward(this, dim0, dim1);
  }
  zero_grad() {
    this.grad = Tensor.zeros(this.data.shape, { device: this.device, requires_grad: this.requires_grad }).data;
  }
  toString() {
    return `Tensor(data=${this.data}, grad=${this.grad})`;
  }
  // New ops
  maximum(other) {
    const otherTensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, { device: this.device });
    return new Operations_exports.Maximum().forward(...this.broadcast(otherTensor));
  }
  eq(other) {
    const otherTensor = other instanceof Tensor ? other : Tensor.full(this.data.shape, other, { device: this.device });
    return new Operations_exports.Equal().forward(...this.broadcast(otherTensor));
  }
  ne(other) {
    const one = new Tensor(1, { device: this.device });
    return one.sub(this.eq(other));
  }
  gte(other) {
    return this.maximum(other).eq(this);
  }
  lte(other) {
    return this.maximum(other).eq(other);
  }
  gt(other) {
    const one = new Tensor([1], { device: this.device });
    return one.sub(this.lte(other));
  }
  lt(other) {
    const one = new Tensor([1], { device: this.device });
    return one.sub(this.gte(other));
  }
  static where(condition, x, y) {
    const one = new Tensor([1], { device: x.device, requires_grad: x.requires_grad });
    return condition.mul(x).add(one.sub(condition).mul(y));
  }
  unsqueeze(dim) {
    if (dim < 0)
      dim = this.shape.length + dim + 1;
    return this.reshape([...this.shape.slice(0, dim), 1, ...this.shape.slice(dim)]);
  }
  assign(tensor) {
    this.data = new Tensor(tensor.data.getData(), tensor.options).data;
    this.grad = new Tensor(tensor.grad.getData(), tensor.options).data;
    this.options = Object.assign({}, tensor.options);
    return this;
  }
  to(device) {
    this.device = device;
    this.options.device = device;
    return this.assign(this);
  }
};

// src/Module.ts
var Module = class {
  constructor() {
    this._total_params = 0;
  }
  forward(x) {
    throw Error("Not implemented");
  }
  zero_grad() {
    for (let p of this.parameters()) {
      p.zero_grad();
    }
  }
  get total_params() {
    return this._total_params;
  }
  named_parameters(prefix = "") {
    let result = {};
    for (let key of Object.keys(this)) {
      let fullPath = prefix + key;
      if (this[key] instanceof Module) {
        let childParameters = this[key].named_parameters(fullPath + ".");
        for (let childKey in childParameters) {
          result[childKey] = childParameters[childKey];
        }
      } else if (Array.isArray(this[key])) {
        this[key].forEach((item, index) => {
          if (item instanceof Module) {
            let childParameters = item.named_parameters(fullPath + "." + index + ".");
            for (let childKey in childParameters) {
              result[childKey] = childParameters[childKey];
            }
          } else if (item instanceof Tensor) {
            result[`${fullPath}[${index}]`] = item;
          }
        });
      } else if (this[key] instanceof Tensor) {
        result[fullPath] = this[key];
      }
    }
    return result;
  }
  parameters() {
    let params = [];
    const keys = Object.keys(this);
    for (let key of keys) {
      const property = this[key];
      if (property instanceof Module) {
        const module = property;
        params.push(...module.parameters());
      } else if (property instanceof Array) {
        for (let ak of property) {
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
  get_name() {
    return this.constructor.name;
  }
  toString() {
    function _addindent(s_, numSpaces) {
      let s = s_.split("\n");
      if (s.length == 1) {
        return s_;
      }
      const first = s.shift();
      s = s.map((line) => new Array(numSpaces).fill(" ").join("") + line);
      let str = "";
      for (let line of s) {
        str += "\n" + line;
      }
      str = first + str;
      return str;
    }
    let child_lines = [];
    const keys = Object.keys(this);
    for (let key of keys) {
      if (this[key] instanceof Module || this[key]["parameters"]) {
        const module = this[key];
        let mod_str = `${module}`;
        mod_str = _addindent(mod_str, 2);
        child_lines.push("(" + key + "): " + mod_str);
      } else if (this[key] instanceof Array && key !== "layers") {
        for (let ak of this[key]) {
          const module = ak;
          let mod_str = `${module}`;
          mod_str = _addindent(mod_str, 2);
          child_lines.push("(" + key + "): " + mod_str);
        }
      }
    }
    const lines = child_lines;
    let main_str = this.get_name() + "(";
    if (lines) {
      main_str += "\n  ";
      for (let line of lines) {
        main_str += line + "\n  ";
      }
    }
    main_str += ")";
    return main_str;
  }
  load_state_dict(stateDict) {
    const namedParameters = this.named_parameters();
    const entries = Object.entries(stateDict);
    for (let state of entries) {
      const path = state[0];
      const tensor = state[1];
      if (!namedParameters[path])
        throw Error(`Layer ${path} not found`);
      const t = new Tensor(tensor);
      const stateTensorShape = t.shape.reduce((p, c) => p * c);
      const modelParameterShape = namedParameters[path].shape.reduce((p, c) => p * c);
      if (stateTensorShape != modelParameterShape)
        throw Error(`State tensor shape (${stateTensorShape}) doesn't match model tensor shape (${modelParameterShape})`);
      namedParameters[path].assign(t);
    }
  }
  to(device) {
    for (let parameter of this.parameters()) {
      parameter.assign(parameter.to(device));
    }
    return this;
  }
};

// src/Random.ts
var _Random = class {
  static SetRandomSeed(seed) {
    _Random._seed = seed;
  }
  static Random() {
    var t = _Random._seed += 1831565813;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    t = ((t ^ t >>> 14) >>> 0) / 4294967296;
    _Random._seed++;
    return t;
  }
  static RandomRange(min, max) {
    return _Random.Random() * (max - min) + min;
  }
};
var Random = _Random;
Random._seed = Math.floor(Math.random() * 1e8);

// src/nn/index.ts
var nn_exports = {};
__export(nn_exports, {
  Dropout: () => Dropout,
  Embedding: () => Embedding,
  LayerNorm: () => LayerNorm,
  Linear: () => Linear,
  Sequential: () => Sequential,
  Softmax: () => Softmax
});

// src/nn/Sequential.ts
var Sequential = class extends Module {
  constructor(...modules) {
    super();
    this.modules = modules;
  }
  forward(x) {
    for (let module of this.modules) {
      x = module.forward(x);
    }
    return x;
  }
};

// src/nn/Linear.ts
var Linear = class extends Module {
  constructor(in_features, out_features) {
    super();
    this.weight = Tensor.uniform(-1, 1, [out_features, in_features], { requires_grad: true });
    this.bias = Tensor.zeros([out_features], { requires_grad: true });
    this.in_features = in_features;
    this.out_features = out_features;
  }
  forward(x) {
    const wt = this.weight.permute();
    x = wt.shape.length === 1 ? x.mul(wt) : x.matmul(wt);
    return this.bias ? x.add(this.bias) : x;
  }
  parameters() {
    return [this.weight, this.bias];
  }
  toString() {
    return `Linear(in_features=${this.in_features}, out_features=${this.out_features})`;
  }
};

// src/nn/Dropout.ts
var Dropout = class extends Module {
  // probability of an element to be zeroed.
  constructor(p = 0.5) {
    super();
    this.p = p;
  }
  forward(x) {
    if (this.p === 0)
      return x;
    const mask = Tensor.rand(x.shape).gte(this.p).reshape(x.shape);
    return x.mul(mask).mul(new Tensor(1).div(new Tensor(1).sub(this.p)));
  }
  parameters() {
    return [];
  }
  toString() {
    return `Dropout(p=${this.p.toFixed(2)})`;
  }
};

// src/nn/LayerNorm.ts
var LayerNorm = class extends Module {
  constructor(normalized_shape, eps = 1e-5, elementwise_affine = true) {
    super();
    this.eps = eps;
    this.elementwise_affine = elementwise_affine;
    this.weight = Tensor.ones(normalized_shape, { requires_grad: true });
    this.bias = elementwise_affine ? Tensor.zeros(normalized_shape, { requires_grad: true }) : null;
  }
  layernorm(self, axis = -1, eps = 1e-5) {
    const a = self.mean(axis, true);
    const y = self.sub(a);
    const b = y.mul(y);
    const c = b.mean(axis, true);
    const d = c.add(eps);
    const e = d.rsqrt();
    const f = y.mul(e);
    return f;
  }
  forward(x) {
    const axis = -1;
    x = this.layernorm(x, axis, this.eps);
    if (!this.elementwise_affine)
      return x;
    return x.mul(this.weight).add(this.bias);
  }
  parameters() {
    return [this.weight, this.bias];
  }
  toString() {
    return `LayerNorm(eps=${this.eps})`;
  }
};

// src/nn/Softmax.ts
var Softmax = class extends Module {
  constructor(dim) {
    super();
    this.dim = dim;
  }
  forward(x) {
    return x.exp().div(x.exp().sum(this.dim, true));
  }
  parameters() {
    return [];
  }
  toString() {
    return `SoftMax(dim=${this.dim})`;
  }
};

// src/nn/Embedding.ts
var Embedding = class extends Module {
  constructor(num_embeddings, embedding_dim) {
    super();
    this.weight = Tensor.uniform(-1, 1, [num_embeddings, embedding_dim], { requires_grad: true });
    this.num_embeddings = num_embeddings;
    this.embedding_dim = embedding_dim;
  }
  getFirsts(v) {
    const data = v.data.getData();
    return [data[0][0][0], data[0][0][1], data[0][0][2]];
  }
  forward(x) {
    const va = Tensor.arange(0, this.num_embeddings, 1, { device: x.device });
    const vb = va.reshape([1, 1, this.num_embeddings]);
    const vc = vb.expand([...x.shape, this.num_embeddings]);
    const vocab_counter = vc;
    const a = x.unsqueeze(2);
    const b = vocab_counter.eq(a);
    const c = b.expand([...x.shape, this.num_embeddings]);
    const d = c.matmul(this.weight);
    return new Tensor(d);
  }
  parameters() {
    return [this.weight];
  }
  toString() {
    return `Embedding(num_embeddings=${this.num_embeddings}, embedding_dim=${this.embedding_dim})`;
  }
};

// src/Operations.ts
var Operations_exports = {};
__export(Operations_exports, {
  Add: () => Add,
  Equal: () => Equal,
  Exp: () => Exp,
  Expand: () => Expand,
  Log: () => Log,
  Matmul: () => Matmul,
  Maximum: () => Maximum,
  Mul: () => Mul,
  Operation: () => Operation,
  Permute: () => Permute,
  Pow: () => Pow,
  Relu: () => Relu,
  Reshape: () => Reshape,
  Sum: () => Sum,
  Transpose: () => Transpose
});
var Operation = class {
  forward(...args) {
    throw Error("Not implemented");
  }
  backward(grad) {
    throw Error("Not implemented");
  }
};
var Add = class extends Operation {
  forward(x, y) {
    this.x = x;
    this.y = y;
    return new Tensor(BinaryOp.add(x.data, y.data), { _children: [x, y], _op: this });
  }
  backward(grad) {
    return [grad, grad];
  }
};
var Mul = class extends Operation {
  forward(x, y) {
    this.x = x;
    this.y = y;
    return new Tensor(BinaryOp.mul(x.data, y.data), { _children: [x, y], _op: this });
  }
  backward(grad) {
    return [
      BinaryOp.mul(this.y.data, grad),
      BinaryOp.mul(this.x.data, grad)
    ];
  }
};
var Pow = class extends Operation {
  forward(x, y) {
    this.x = x;
    this.y = y;
    return new Tensor(BinaryOp.pow(x.data, y.data), { _children: [x], _op: this });
  }
  backward(grad) {
    const a = BinaryOp.sub(this.y.data, Tensor.ones(this.y.shape, { device: this.y.device }).data);
    const b = BinaryOp.pow(this.x.data, a);
    const c = BinaryOp.mul(this.y.data, b);
    const d = BinaryOp.mul(c, grad);
    return [d, null];
  }
};
var Matmul = class extends Operation {
  forward(x, y) {
    this.x = x;
    this.y = y;
    return new Tensor(x._matmul(y), { _children: [x, y], _op: this });
  }
  backward(grad) {
    return [
      new Tensor(grad)._matmul(this.y.T).data,
      this.x.T._matmul(new Tensor(grad)).data
    ];
  }
};
var Sum = class extends Operation {
  forward(x, axis = null, keepdims = false) {
    this.shape = x.shape.slice();
    return new Tensor(ReduceOp.sum(x.data, axis, keepdims), { _children: [x], _op: this });
  }
  backward(grad) {
    return [
      MovementOp.expand(grad, this.shape),
      null
    ];
  }
};
var Log = class extends Operation {
  forward(x) {
    this.x = x;
    return new Tensor(UnaryOp.log(x.data), { _children: [x], _op: this });
  }
  backward(grad) {
    return [
      BinaryOp.div(grad, this.x.data),
      null
    ];
  }
};
var Exp = class extends Operation {
  forward(x) {
    this.out = new Tensor(UnaryOp.exp(x.data), { _children: [x], _op: this });
    return this.out;
  }
  backward(grad) {
    return [
      BinaryOp.mul(grad, this.out.data),
      null
    ];
  }
};
var Relu = class extends Operation {
  forward(x) {
    this.out = new Tensor(Tensor.zeros(x.shape, { device: x.device }).maximum(x), { _children: [x], _op: this });
    return this.out;
  }
  backward(grad) {
    return [
      Tensor.where(this.out.gt(0), new Tensor(grad), Tensor.zeros(this.out.shape, { device: this.out.device })).data,
      null
    ];
  }
};
var Expand = class extends Operation {
  forward(x, shape) {
    this.shape = x.shape.slice();
    return new Tensor(MovementOp.expand(x.data, shape), { _children: [x], _op: this });
  }
  backward(grad1) {
    let gradTensor = new Tensor(grad1);
    const gradShape = gradTensor.shape.slice();
    const originalShape = this.shape.slice();
    if (originalShape.length > gradShape.length) {
      throw Error(`Gradient shape ${gradShape} needs to be bigger or equal to the input shape ${originalShape}`);
    }
    if (originalShape.length === gradShape.length) {
      for (let i = 0; i < gradShape.length; i++) {
        const g = gradShape[i];
        const o = originalShape[i];
        if (g !== o) {
          if (o !== 1)
            throw Error(`Broadcasted went wrong? Shape lengths match but non matching dimension is expected to have dim 1 and has dim ${o}`);
          gradTensor = gradTensor.sum(i);
        }
      }
    } else {
      const sumsRequired = gradShape.length - originalShape.length;
      for (let i = 0; i < sumsRequired; i++) {
        gradTensor = gradTensor.sum(0);
      }
    }
    return [
      gradTensor.data,
      null
    ];
  }
};
var Reshape = class extends Operation {
  forward(x, shape) {
    this.shape = x.shape;
    return new Tensor(MovementOp.reshape(x.data, shape), { _children: [x], _op: this });
  }
  backward(grad) {
    return [
      MovementOp.reshape(grad, this.shape),
      null
    ];
  }
};
var Permute = class extends Operation {
  forward(x, axes) {
    this.axes = axes;
    return new Tensor(MovementOp.permute(x.data, axes), { _children: [x], _op: this });
  }
  backward(grad) {
    return [
      MovementOp.permute(grad, this.axes),
      null
    ];
  }
};
var Transpose = class extends Operation {
  forward(x, dim0, dim1) {
    this.dim0 = dim0;
    this.dim1 = dim1;
    return new Tensor(MovementOp.transpose(x.data, dim0, dim1), { _children: [x], _op: this });
  }
  backward(grad) {
    return [
      MovementOp.transpose(grad, this.dim0, this.dim1),
      null
    ];
  }
};
var Maximum = class extends Operation {
  forward(x, y) {
    this.x = x;
    this.y = y;
    return new Tensor(BinaryOp.maximum(x.data, y.data), { _children: [x, y], _op: this });
  }
  backward(grad) {
    return [
      this.x.gte(this.y).data,
      this.y.gt(this.x).data
    ];
  }
};
var Equal = class extends Operation {
  forward(x, y) {
    return new Tensor(BinaryOp.equal(x.data, y.data), { _children: [x, y], _op: this });
  }
  backward(grad) {
    return [null, null];
  }
};

// src/Matrix.ts
var Matrix = class {
  constructor(data, shape, strides, offset) {
    this.data = data instanceof Float32Array ? data : new Float32Array(data.flat(Infinity));
    this.shape = shape === void 0 || shape.length === 0 ? this.computeShape(data, []) : [...shape];
    this.strides = strides ? [...strides] : Matrix.computeStrides(this.shape);
    this.offset = offset ? offset : 0;
  }
  // Getter function for data
  get(i) {
    let idx = 0;
    let totalSize = this.shape.reduce((a, b) => a * b, 1);
    for (let dim = 0; dim < this.shape.length; ++dim) {
      totalSize /= this.shape[dim];
      const coord = Math.floor(i / totalSize);
      i -= coord * totalSize;
      idx += this.strides[dim] * coord;
    }
    return this.data[(idx + this.offset) % this.data.length];
  }
  get1D(index) {
    return this.data[this.offset + this.strides[0] * index];
  }
  get2D(index) {
    let row = Math.floor(index / this.shape[1]);
    let col = index % this.shape[1];
    let dataIndex = this.offset + this.strides[0] * row + this.strides[1] * col;
    return this.data[dataIndex];
  }
  getValue(indices) {
    let index = 0;
    for (let i = 0; i < indices.length; i++) {
      index += this.strides[i] * indices[i];
    }
    return this.data[(index + this.offset) % this.data.length];
  }
  getNestedData(indices = []) {
    if (indices.length === this.shape.length) {
      return this.getValue(indices);
    }
    let dimSize = this.shape[indices.length];
    let nestedData = new Array(dimSize);
    for (let i = 0; i < dimSize; i++) {
      nestedData[i] = this.getNestedData([...indices, i]);
    }
    return nestedData;
  }
  getData() {
    return this.getNestedData();
  }
  // TODO: Check for compatibility between data and shape
  computeShape(data, shape) {
    if (!data.length || data.length == 0) {
      return shape;
    }
    shape.push(data.length);
    return this.computeShape(data[0], shape);
  }
  static computeStrides(shape) {
    let strides = new Array(shape.length);
    strides[strides.length - 1] = 1;
    for (let i = strides.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }
  static slice(m, indices) {
    if (indices.length != m.shape.length)
      throw Error(`Indices [${indices}] must match matrix shape ${m.shape}`);
    let offset = m.offset;
    let newShape = [];
    let newStrides = [];
    for (let i = 0; i < indices.length; i++) {
      const index = indices[i];
      if (Array.isArray(index)) {
        const [start, stop] = index;
        if (start < 0 || start >= m.shape[i] || stop < 0 || stop > m.shape[i]) {
          throw Error(`Slice ${start}:${stop} out of bounds for axis ${i} with size ${m.shape[i]}`);
        }
        offset += start * m.strides[i];
        newShape.push(stop - start);
        newStrides.push(m.strides[i]);
      } else if (index !== null) {
        if (index < 0 || index >= m.shape[i]) {
          throw Error(`Index ${index} out of bounds for axis ${i} with size ${m.shape[i]}`);
        }
        offset += index * m.strides[i];
      } else {
        newShape.push(m.shape[i]);
        newStrides.push(m.strides[i]);
      }
    }
    return new Matrix(m.data, newShape, newStrides, offset);
  }
  static multinomial(matrix, num_samples, normalized = false) {
    const origRank = matrix.shape.length;
    const logits2D = origRank === 1 ? matrix.reshape([1, -1]) : matrix;
    const probabilities = normalized ? logits2D : Matrix.softmax(logits2D, -1);
    const batchSize = probabilities.shape[0];
    const numEvents = probabilities.shape[1];
    const probVals = probabilities.data;
    const resShape = [batchSize, num_samples];
    const resVals = new Float32Array(resShape.reduce((p, c) => p * c));
    for (let b = 0; b < batchSize; ++b) {
      const offset = b * numEvents;
      const cdf = new Float32Array(numEvents - 1);
      cdf[0] = probVals[offset];
      for (let event = 1; event < cdf.length; ++event) {
        cdf[event] = cdf[event - 1] + probVals[offset + event];
      }
      const outOffset = b * num_samples;
      for (let sampleId = 0; sampleId < num_samples; ++sampleId) {
        const r = Random.Random();
        resVals[outOffset + sampleId] = cdf.length;
        for (let event = 0; event < cdf.length; event++) {
          if (r < cdf[event]) {
            resVals[outOffset + sampleId] = event;
            break;
          }
        }
      }
    }
    return new Matrix(resVals, resShape);
  }
  // public static split(matrix: Matrix, split_sizes: number | number[], dim: null | number = null): Matrix[] {
  //     if (Array.isArray(split_sizes)) throw Error("Split split_sizes as array not supported");
  //     if (dim !== null) throw Error("Split dim not supported");
  //     const chunkSize = split_sizes;
  //     const stride = matrix.shape[matrix.shape.length - 1];
  //     if (stride % chunkSize !== 0) {
  //         throw new Error('Invalid chunk size, not evently divisible into last tensor dimension')
  //     }
  //     // Setup the output chunks
  //     const out: Matrix[] = [];
  //     const chunks = stride / chunkSize;
  //     for (let c = 0; c < chunks; c++) {
  //         out.push(Matrix.zeros([...matrix.shape.slice(0, matrix.shape.length - 1), chunkSize]));
  //     }
  //     const outOffsets = out.map(_ => 0);
  //     let sourceOffset = 0;
  //     // Split up the actual data
  //     const macroChunks = matrix.data.length / stride;
  //     for (let i = 0; i < macroChunks; i++) {
  //         for (let j = 0; j < chunks; j++) {
  //             out[j].data.set(matrix.data.slice(sourceOffset, sourceOffset + chunkSize), outOffsets[j])
  //             outOffsets[j] += chunkSize;
  //             sourceOffset += chunkSize
  //         }
  //     }
  //     return out;
  // }
  static split(matrix, split_sizes, dim = null) {
    if (Array.isArray(split_sizes))
      throw Error("Split split_sizes as array not supported");
    if (dim !== null)
      throw Error("Split dim not supported");
    const chunkSize = split_sizes;
    const lastDim = matrix.shape[matrix.shape.length - 1];
    if (lastDim % chunkSize !== 0) {
      throw new Error("Invalid chunk size, not evenly divisible into last tensor dimension");
    }
    const numChunks = lastDim / chunkSize;
    const out = [];
    let start = 0;
    for (let i = 0; i < numChunks; i++) {
      let end = start + chunkSize;
      const sliceIndices = matrix.shape.map(
        (dimSize, idx) => idx === matrix.shape.length - 1 ? [start, end] : [0, dimSize]
      );
      const chunk = Matrix.slice(matrix, sliceIndices);
      out.push(chunk);
      start = end;
    }
    return out;
  }
  // TODO: Handle negative axis and error checking
  static concatenateArrays(arrays, axis = 0) {
    if (axis < 0) {
      throw new Error(`Invalid axis value ${axis} ${arrays[0].length}`);
    }
    if (axis === 0) {
      return [].concat(...arrays);
    } else {
      const result = [];
      for (let i = 0; i < arrays[0].length; i++) {
        const newArrays = arrays.map((arr) => arr[i]);
        result.push(Matrix.concatenateArrays(newArrays, axis - 1));
      }
      return result;
    }
  }
  static cat(matrices, dim = 0) {
    if (dim === null) {
      const flattenData = matrices.map((v) => Array.from(v.data)).flat(Infinity);
      return new Matrix(flattenData);
    }
    const data = matrices.map((m) => m.getData());
    return new Matrix(Matrix.concatenateArrays(data, dim));
  }
  static permute(m, axes = null) {
    if (axes === null) {
      return new Matrix(m.data, [...m.shape].reverse(), [...m.strides].reverse());
    }
    let newShape = [];
    let newStrides = [];
    for (let i = 0; i < axes.length; i++) {
      let axis = axes[i] < 0 ? m.shape.length + axes[i] : axes[i];
      newShape[i] = m.shape[axis];
      newStrides[i] = m.strides[axis];
    }
    const ret = new Matrix(m.data, newShape, newStrides);
    return ret;
  }
  static transpose(m, dim0, dim1) {
    dim0 = dim0 < 0 ? m.shape.length + dim0 : dim0;
    dim1 = dim1 < 0 ? m.shape.length + dim1 : dim1;
    if (dim0 >= m.shape.length || dim1 >= m.shape.length) {
      throw new Error("Transpose dimensions out of range");
    }
    let axes = [];
    for (let i = 0; i < m.shape.length; i++) {
      axes[i] = i;
    }
    let tmp = axes[dim0];
    axes[dim0] = axes[dim1];
    axes[dim1] = tmp;
    return Matrix.permute(m, axes);
  }
  static isContiguous(m) {
    let stride = 1;
    for (let i = m.shape.length - 1; i >= 0; --i) {
      if (m.strides[i] !== stride) {
        return false;
      }
      stride *= m.shape[i];
    }
    return true;
  }
  static reshape(m, newShape) {
    if (newShape === -1)
      newShape = [m.data.length];
    if (!(newShape instanceof Array))
      newShape = [newShape];
    const totalSize = m.shape.reduce((acc, val) => acc * val, 1);
    const missingSize = newShape.reduce((acc, val) => val >= 0 ? acc * val : acc, 1);
    const inferredShape = newShape.map((val) => val === -1 ? totalSize / missingSize : val);
    const newStrides = new Array(inferredShape.length).fill(0);
    let stride = 1;
    for (let i = inferredShape.length - 1; i >= 0; --i) {
      newStrides[i] = stride;
      stride *= inferredShape[i];
    }
    let newData = m.data;
    const notContiguous = m.strides[m.strides.length - 1] !== 1 || !this.isContiguous(m);
    if (m.strides[m.strides.length - 1] !== 1 || notContiguous) {
      newData = new Float32Array(totalSize);
      const coords = new Array(m.shape.length).fill(0);
      for (let i = 0; i < totalSize; ++i) {
        let index = 0;
        for (let j = 0; j < coords.length; ++j) {
          index += m.strides[j] * coords[j];
        }
        newData[i] = m.data[index];
        ++coords[coords.length - 1];
        for (let j = coords.length - 1; j >= 0; --j) {
          if (coords[j] >= m.shape[j] && j > 0) {
            coords[j] = 0;
            ++coords[j - 1];
          }
        }
      }
    }
    return new Matrix(newData, inferredShape);
  }
  static reshape2d(shape) {
    let shape2d = [shape[0], 1];
    for (let i = 1; i < shape.length; i++) {
      shape2d[1] *= shape[i];
    }
    return shape2d;
  }
  static reshape1d(shape) {
    return shape.reduce((p, c) => p * c);
  }
  static equalArrays(first, second) {
    if (first.length != second.length)
      return false;
    for (let i = 0; i < first.length; i++) {
      if (first[i] !== second[i])
        return false;
    }
    return true;
  }
  static broadcast(x, y) {
    if (!(x instanceof Matrix))
      x = new Matrix([x]);
    if (!(y instanceof Matrix))
      y = new Matrix([y]);
    if (Matrix.equalArrays(x.shape, y.shape))
      return [x, y];
    let finalShape = [];
    let maxLength = Math.max(x.shape.length, y.shape.length);
    for (let i = 0; i < maxLength; i++) {
      finalShape.push(Math.max(x.shape[x.shape.length - i - 1] || 1, y.shape[y.shape.length - i - 1] || 1));
    }
    finalShape = finalShape.reverse();
    return [x.expand(finalShape), y.expand(finalShape)];
  }
  broadcast(other) {
    return Matrix.broadcast(this, other);
  }
  broadcast_to(shape) {
    return Matrix.broadcast(this, Matrix.zeros(shape))[0];
  }
  // Helpers
  static MatrixFromNumber(value, shape) {
    const desiredElements = shape.reduce((acc, val) => acc * val, 1);
    if (value === 0)
      return new Matrix(new Float32Array(desiredElements), shape);
    return new Matrix(new Float32Array(desiredElements).fill(value), shape);
  }
  static arange(start, stop, step = 1) {
    let data = new Float32Array(Math.floor((stop - start) / step));
    let s = 0;
    for (let i = start; i < stop; i += step) {
      data[s] = i;
      s++;
    }
    return new Matrix(data);
  }
  static rand(shape) {
    let data = new Float32Array(shape.reduce((prev, curr) => prev * curr));
    for (let i = 0; i < data.length; i++) {
      data[i] = Random.Random();
    }
    return new Matrix(data, shape);
  }
  static uniform(low, high, shape) {
    const m = Matrix.zeros(shape);
    for (let i = 0; i < m.data.length; i++) {
      m.data[i] = Random.RandomRange(low, high);
    }
    return m;
  }
  static expand_dims(m, axis) {
    if (!Array.isArray(axis))
      axis = [axis];
    let newShape = [...m.shape];
    for (let i = 0; i < axis.length; i++) {
      newShape.splice(axis[i] < 0 ? newShape.length + axis[i] + 1 : axis[i], 0, 1);
    }
    return this.reshape(m, newShape);
  }
  static unsqueeze(m, dim) {
    if (dim < 0)
      dim = m.shape.length + dim + 1;
    return m.reshape([...m.shape.slice(0, dim), 1, ...m.shape.slice(dim)]);
  }
  expand(shape) {
    const lenDiff = shape.length - this.shape.length;
    let oldShape = this.shape;
    let oldStrides = this.strides;
    if (lenDiff > 0) {
      oldShape = Array(lenDiff).fill(1).concat(this.shape);
      oldStrides = Array(lenDiff).fill(0).concat(this.strides);
    }
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] == -1) {
        if (i >= oldShape.length) {
          throw new Error("Cannot infer dimension for expansion");
        }
        shape[i] = oldShape[i];
      }
    }
    let newStrides = new Array(shape.length).fill(0);
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] == oldShape[i]) {
        newStrides[i] = oldStrides[i];
      } else if (oldShape[i] == 1) {
        newStrides[i] = 0;
      }
    }
    const newShape = shape;
    return new Matrix(this.data, newShape, newStrides);
  }
  static _tri(r, c, k = 0) {
    let a = Matrix.arange(0, r).unsqueeze(1).expand([r, c]);
    let b = Matrix.arange(-k, c - k).unsqueeze(0).expand([r, c]);
    return a.lte(b);
  }
  static triu(m, k = 0) {
    const a = Matrix._tri(m.shape[m.shape.length - 2], m.shape[m.shape.length - 1], k);
    return Matrix.where(a, m, Matrix.zeros(m.shape));
  }
  static tril(m, k = 0) {
    const a = Matrix._tri(m.shape[m.shape.length - 2], m.shape[m.shape.length - 1], k + 1);
    return Matrix.where(a, Matrix.zeros(m.shape), m);
  }
  static masked_fill(m, mask, value) {
    const [mb, maskb] = Matrix.broadcast(m, mask);
    const fillMatrix = Matrix.full(mb.shape, value);
    const filled = Matrix.where(maskb, fillMatrix, mb);
    return filled;
  }
  copy() {
    return new Matrix(this.data.slice(), this.shape.slice());
  }
  static zeros(size) {
    return Matrix.MatrixFromNumber(0, size);
  }
  static ones(size) {
    return Matrix.MatrixFromNumber(1, size);
  }
  static full(size, value) {
    return Matrix.MatrixFromNumber(value, size);
  }
  static binary_op(m1, m2, op) {
    let newData = new Float32Array(m1.shape.reduce((p, c) => p * c));
    for (let i = 0; i < newData.length; i++) {
      let value = 0;
      if (op == 0 /* ADD */)
        value = m1.get(i) + m2.get(i);
      else if (op == 1 /* SUB */)
        value = m1.get(i) - m2.get(i);
      else if (op == 2 /* MUL */)
        value = m1.get(i) * m2.get(i);
      else if (op == 3 /* DIV */)
        value = m1.get(i) / m2.get(i);
      else if (op == 4 /* POW */)
        value = m1.get(i) ** m2.get(i);
      newData[i] = isNaN(value) ? 0 : value;
    }
    return new Matrix(newData, m1.shape);
  }
  static add(m1, m2) {
    return Matrix.binary_op(...Matrix.broadcast(m1, m2), 0 /* ADD */);
  }
  static sub(m1, m2) {
    return Matrix.binary_op(...Matrix.broadcast(m1, m2), 1 /* SUB */);
  }
  static mul(m1, m2) {
    return Matrix.binary_op(...Matrix.broadcast(m1, m2), 2 /* MUL */);
  }
  static div(m1, m2) {
    return Matrix.binary_op(...Matrix.broadcast(m1, m2), 3 /* DIV */);
  }
  static pow(m1, m2) {
    return Matrix.binary_op(...Matrix.broadcast(m1, m2), 4 /* POW */);
  }
  static dot(m1, m2) {
    const x = m1.reshape([...m1.shape.slice(0, m1.shape.length - 1), 1, ...m1.shape.slice(m1.shape.length - 1, m1.shape.length)]);
    let w = m2.reshape([...m2.shape.slice(0, m2.shape.length - 2), 1, ...m2.shape.slice(m2.shape.length - 2, m2.shape.length - 1), ...m2.shape.slice(m2.shape.length - 1, m2.shape.length)]);
    w = w.transpose(-1, -2);
    let r = x.mul(w);
    r = r.sum(-1);
    if (m1.shape.length == 1) {
      r = r.reshape([...r.shape.slice(0, r.shape.length - 3), ...r.shape.slice(r.shape.length - 2, r.shape.length - 1)]);
    }
    return r;
  }
  static reduce(m, func, axis = null, initialValue = 0, keepdims = false) {
    if (axis === null) {
      const result = m.data.reduce((a, b) => func(new Matrix(new Float32Array([a])), new Matrix(new Float32Array([b]))).data[0], initialValue);
      return new Matrix([result], keepdims ? m.shape.map(() => 1) : [1]);
    }
    if (axis < 0)
      axis += m.shape.length;
    const resultShape = [...m.shape];
    resultShape.splice(axis, 1);
    const numSlices = m.shape[axis];
    const sliceSize = m.data.length / numSlices;
    const stride = m.strides[axis];
    let resultData = new Float32Array();
    for (let slice = 0; slice < numSlices; slice++) {
      let sliceData = new Float32Array(sliceSize);
      for (let i = 0; i < sliceSize; i++) {
        const idx = Math.floor(i / stride) * stride * numSlices + i % stride + stride * slice;
        sliceData[i] = m.data[idx];
      }
      let sliceMatrix = new Matrix(sliceData, [sliceSize]);
      if (slice === 0) {
        resultData = sliceMatrix.data;
      } else {
        resultData = func(new Matrix(resultData, [resultData.length]), sliceMatrix).data;
      }
    }
    if (keepdims) {
      resultShape.splice(axis, 0, 1);
    }
    return new Matrix(resultData, resultShape);
  }
  static reduce2(m, func, axis = null, keepdims = false) {
    if (axis === null) {
      const data = [m.data.reduce((p2, c) => func(p2, c))];
      if (keepdims) {
        return new Matrix(data, m.shape.map((v) => 1));
      }
      return new Matrix(data);
    }
    if (axis < 0)
      axis = m.shape.length + axis;
    function parseAxisParam(axis2, shape) {
      const rank = shape.length;
      axis2 = axis2 == null ? shape.map((s, i) => i) : [].concat(axis2);
      return axis2.map((a) => a < 0 ? rank + a : a);
    }
    function getAxesPermutation(axes2, rank) {
      const result = [];
      for (let i = 0; i < rank; ++i) {
        if (axes2.indexOf(i) === -1) {
          result.push(i);
        }
      }
      axes2.forEach((axis2) => result.push(axis2));
      return result;
    }
    const axes = parseAxisParam(axis, m.shape);
    const pa = getAxesPermutation(axes, m.shape.length);
    let p = m.permute(pa);
    const resultShape = [...m.shape];
    resultShape.splice(axis, 1);
    if (keepdims === true) {
      resultShape.splice(axis, 0, 1);
    }
    const sp = resultShape.reduce((p2, c) => p2 * c);
    let output = new Float32Array(sp);
    const vals = m.shape[axis];
    let additionCounter = 0;
    const l = p.shape.reduce((p2, c) => p2 * c);
    p = p.reshape(Matrix.reshape2d(p.shape));
    for (let i = 0; i < l; i++) {
      for (let index = 0; index < vals; index++) {
        output[additionCounter] = func(output[additionCounter], p.get2D(i));
        i++;
      }
      additionCounter++;
      i--;
    }
    return new Matrix(output, resultShape);
  }
  static sum(m, axis = null, keepdims = false) {
    const sumOp = (a, b) => a + b;
    return Matrix.reduce2(m, sumOp, axis, keepdims);
  }
  static prod(m, axis = null) {
    return Matrix.reduce(m, Matrix.mul, axis, 1);
  }
  static mean(m, axis = null, keepdims = false) {
    if (axis < 0)
      axis = m.shape.length + axis;
    const sum = Matrix.sum(m, axis, keepdims);
    if (axis === null)
      return sum.div(m.data.length);
    return sum.div(m.shape[axis]);
  }
  static var(m, axis = null, keepdims = false) {
    const x = Matrix.abs(m.sub(Matrix.mean(m, axis, true))).pow(2);
    return Matrix.mean(x, axis, keepdims);
  }
  static softmax(m, dim) {
    return m.exp().div(m.exp().sum(dim, true));
  }
  // TODO: No data manipulation
  static abs(m) {
    return new Matrix(m.data.map((v) => v < 0 ? v * -1 : v), m.shape);
  }
  // TODO: No data manipulation
  static exp(x) {
    return new Matrix(x.data.map((v) => Math.exp(v)), x.shape);
  }
  // TODO: No data manipulation
  static tanh(x) {
    return new Matrix(x.data.map((v) => Math.tanh(v)), x.shape);
  }
  compare(m, sign) {
    m = m instanceof Matrix ? m : new Matrix([m]);
    const [m1, m2] = Matrix.broadcast(this, m);
    const data = new Float32Array(m1.shape.reduce((p, c) => p * c));
    for (let i = 0; i < m1.shape.reduce((p, c) => p * c); i++) {
      if (sign == "<")
        data[i] = m1.get(i) < m2.get(i) ? 1 : 0;
      else if (sign == "<=")
        data[i] = m1.get(i) <= m2.get(i) ? 1 : 0;
      else if (sign == ">")
        data[i] = m1.get(i) > m2.get(i) ? 1 : 0;
      else if (sign == ">=")
        data[i] = m1.get(i) >= m2.get(i) ? 1 : 0;
      else if (sign == "==")
        data[i] = m1.get(i) == m2.get(i) ? 1 : 0;
    }
    return new Matrix(data, m1.shape);
  }
  gt(value) {
    return this.compare(value, ">");
  }
  gte(value) {
    return this.compare(value, ">=");
  }
  lt(value) {
    return this.compare(value, "<");
  }
  lte(value) {
    return this.compare(value, "<=");
  }
  equal(value) {
    return this.compare(value, "==");
  }
  static where(condition, x, y) {
    return new Matrix(x.data.map((v, i) => {
      return condition.get(i) ? x.get(i) : y.get(i);
    }), x.shape);
  }
  static maximum(x1, x2) {
    return Matrix.where(x1.gte(x2), x1, x2);
  }
  // Fillers
  add(m) {
    return Matrix.add(this, m);
  }
  sub(m) {
    return Matrix.sub(this, m);
  }
  mul(m) {
    return Matrix.mul(this, m);
  }
  div(m) {
    return Matrix.div(this, m);
  }
  pow(m) {
    return Matrix.pow(this, m);
  }
  exp() {
    return Matrix.exp(this);
  }
  dot(m) {
    return Matrix.dot(this, m);
  }
  split(split_sizes, dim = null) {
    return Matrix.split(this, split_sizes, dim);
  }
  masked_fill(mask, value) {
    return Matrix.masked_fill(this, mask, value);
  }
  unsqueeze(dim) {
    return Matrix.unsqueeze(this, dim);
  }
  sum(axis = null, keepdims = false) {
    return Matrix.sum(this, axis, keepdims);
  }
  mean(axis = null, keepdims = false) {
    return Matrix.mean(this, axis, keepdims);
  }
  var(axis = null, keepdims = false) {
    return Matrix.var(this, axis, keepdims);
  }
  reshape(shape) {
    return Matrix.reshape(this, shape);
  }
  permute(axes = null) {
    return Matrix.permute(this, axes);
  }
  transpose(dim0, dim1) {
    return Matrix.transpose(this, dim0, dim1);
  }
  prod(axis) {
    return Matrix.prod(this, axis);
  }
  tril(k) {
    return Matrix.tril(this, k);
  }
  get T() {
    return Matrix.permute(this);
  }
  toString() {
    function fixed(key, val) {
      return val.toFixed ? Number(val.toFixed(4)) : val;
    }
    return `Matrix(${JSON.stringify(this.getData(), fixed)}, shape=[${this.shape}])`;
  }
};

// test/TestUtils.ts
function assert(condition) {
  if (!condition)
    throw Error("Ups, condition is false");
}
function equalTensorBuffer(a, b, EPS = 1e-5) {
  if (a.shape.length !== b.shape.length) {
    console.log(`Shape of a (${a.shape}) doesn't match shape of b (${b.shape})`);
    return false;
  }
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) {
      console.log(`Shape of a (${a.shape}) doesn't match shape of b (${b.shape})`);
      return false;
    }
  }
  for (let i = 0; i < a.shape.reduce((p, c) => p * c); i++) {
    const aV = a.get(i);
    const bV = b.get(i);
    if (aV === void 0 || bV === void 0) {
      console.log(`Expected values but got a[${i}]=${aV} and b[${i}]=${bV}`);
      return false;
    } else if (aV === null || bV === null) {
      console.log(`Expected values but got a[${i}]=${aV} and b[${i}]=${bV}`);
      return false;
    }
    if (Math.abs(aV - bV) > EPS) {
      console.log(`Data of a (${aV}) doesn't match data of b (${bV})`);
      return false;
    }
  }
  return true;
}
function equalMatrix(a, b, EPS = 1e-5) {
  if (a.shape.length !== b.shape.length) {
    console.log(`Shape of a (${a.shape}) doesn't match shape of b (${b.shape})`);
    return false;
  }
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) {
      console.log(`Shape of a (${a.shape}) doesn't match shape of b (${b.shape})`);
      return false;
    }
  }
  for (let i = 0; i < a.shape.reduce((p, c) => p * c); i++) {
    const aV = a.get(i);
    const bV = b.get(i);
    if (aV === void 0 || bV === void 0) {
      console.log(`Expected values but got a[${i}]=${aV} and b[${i}]=${bV}`);
      return false;
    } else if (aV === null || bV === null) {
      console.log(`Expected values but got a[${i}]=${aV} and b[${i}]=${bV}`);
      return false;
    }
    if (Math.abs(aV - bV) > EPS) {
      console.log(`Data of a (${aV}) doesn't match data of b (${bV})`);
      return false;
    }
  }
  return true;
}
function equalTensor(a, b, EPS = 1e-5) {
  const equalData = equalTensorBuffer(a.data, b.data, EPS);
  const equalGrads = a.grad && b.grad ? equalTensorBuffer(a.grad, b.grad, EPS) : true;
  return equalData && equalGrads;
}
function equalArray(a, b, EPS = 1e-5) {
  if (a.length !== b.length) {
    console.log(`Length of a(${a.length}) not equal to length of b(${b.length})`);
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > EPS) {
      console.log(`Value of a[${i}]=${a[i]} not equal to value of b[${i}]=${b[i]}`);
      return false;
    }
  }
  return true;
}
function equal(a, b, EPS = 1e-5) {
  if (a instanceof Tensor && b instanceof Tensor)
    return equalTensor(a, b, EPS);
  else if (a instanceof Matrix && b instanceof Matrix)
    return equalMatrix(a, b, EPS);
  else if (a instanceof Array && b instanceof Array)
    return equalArray(a, b, EPS);
  throw Error("Tried to compare unknown type");
  return false;
}
function TensorFactory(tensorData) {
  const tensor = new Tensor(tensorData.data);
  tensor.grad = new Tensor(tensorData.grad).data;
  return tensor;
}

// test/web/NN.test.ts
function NNTest(device) {
  TestRunner.describe("Linear", () => {
    let linear = new nn_exports.Linear(2, 4);
    linear.weight = new Tensor([[-0.5963, -62e-4], [0.1741, -0.1097], [-0.4237, -0.6666], [0.1204, 0.2781]], { requires_grad: true });
    linear.bias = new Tensor([-0.458, -0.3401, 0.295, 0.1145], { requires_grad: true });
    linear.to(device);
    const input = new Tensor([[1, 2], [3, 4]], { device, requires_grad: true });
    const out = linear.forward(input);
    assert(`${linear}` === `Linear(in_features=2, out_features=4)`);
    assert(equal(out, TensorFactory({ data: [[-1.0668, -0.3854, -1.4619, 0.7911], [-2.2719, -0.2566, -3.6425, 1.5882]], grad: [[0, 0, 0, 0], [0, 0, 0, 0]] }), 1e-3));
  });
  TestRunner.describe("Sequential", () => {
    const model = new nn_exports.Sequential(
      new nn_exports.Linear(2, 4),
      new nn_exports.Linear(4, 4),
      new nn_exports.Linear(4, 1)
    ).to(device);
    assert(model.modules.length === 3);
    assert(`${model.modules[0]}` === `Linear(in_features=2, out_features=4)`);
    assert(`${model.modules[1]}` === `Linear(in_features=4, out_features=4)`);
    assert(`${model.modules[2]}` === `Linear(in_features=4, out_features=1)`);
  });
  TestRunner.describe("Dropout", () => {
    Random.SetRandomSeed(1337);
    let x = new Tensor([[
      9e-3,
      0,
      0.1623,
      0,
      0,
      0.4064,
      0,
      0,
      0.1924,
      0,
      0,
      0.0542,
      0,
      0.4154,
      0,
      0.2993,
      0,
      0.3429,
      0.3209,
      82e-4
    ]]);
    const dropout = new nn_exports.Dropout().to(device);
    x = dropout.forward(x);
    assert(equal(x, TensorFactory({ data: [[0, 0, 0.3246, 0, 0, 0.8128, 0, 0, 0, 0, 0, 0.1084, 0, 0.8308, 0, 0.5986, 0, 0, 0.6418, 0]], grad: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] })));
  });
  TestRunner.describe("LayerNorm", () => {
    const x = Tensor.arange(0, 10, 1, { device });
    const layerNorm = new nn_exports.LayerNorm([10]).to(device);
    const r = layerNorm.forward(x);
    assert(equal(r, TensorFactory({ data: [-1.5666979540876567, -1.2185428531792886, -0.8703877522709204, -0.5222326513625523, -0.17407755045418408, 0.17407755045418408, 0.5222326513625523, 0.8703877522709204, 1.2185428531792886, 1.5666979540876567], grad: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] })));
  });
  TestRunner.describe("Softmax", () => {
    const x = Tensor.arange(0, 10, 1, { device });
    const softmax = new nn_exports.Softmax(0).to(device);
    const r = softmax.forward(x);
    assert(equal(r, TensorFactory({ data: [7801341612780744e-20, 21206245143623275e-20, 5764455082375903e-19, 0.0015669413501390806, 0.004259388198344144, 0.011578217539911801, 0.031472858344688034, 0.08555209892803112, 0.23255471590259755, 0.6321492583604867], grad: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] })));
    const x1 = new Tensor([[5, 6, 3]], { device });
    const softmax1 = new nn_exports.Softmax(1).to(device);
    const r1 = softmax1.forward(x1);
    assert(equal(r1, TensorFactory({ data: [[0.25949646034241913, 0.7053845126982412, 0.03511902695933972]], grad: [[0, 0, 0]] })));
    const softmax2 = new nn_exports.Softmax(-1).to(device);
    const r2 = softmax2.forward(x1);
    assert(equal(r2, TensorFactory({ data: [[0.25949646034241913, 0.7053845126982412, 0.03511902695933972]], grad: [[0, 0, 0]] })));
  });
  TestRunner.describe("Embedding", () => {
    Random.SetRandomSeed(1337);
    let embedding = new nn_exports.Embedding(10, 3);
    embedding.weight = new Tensor([
      [0.1808, -0.07, -0.3596],
      [-0.9152, 0.6258, 0.0255],
      [0.9545, 0.0643, 0.3612],
      [1.1679, -1.3499, -0.5102],
      [0.236, -0.2398, -0.4713],
      [84e-4, -0.6631, -0.2513],
      [1.0101, 0.1215, 0.1584],
      [1.134, -0.2221, 0.6924],
      [-0.5075, -0.9239, 0.5467],
      [-1.4948, -1.2057, 0.5718]
    ]);
    embedding = embedding.to(device);
    const input = new Tensor([[1, 2, 4, 5], [4, 3, 2, 9]], { device });
    const out = embedding.forward(input);
    assert(equal(out, TensorFactory({ data: [
      [
        [-0.9152, 0.6258, 0.0255],
        [0.9545, 0.0643, 0.3612],
        [0.236, -0.2398, -0.4713],
        [84e-4, -0.6631, -0.2513]
      ],
      [
        [0.236, -0.2398, -0.4713],
        [1.1679, -1.3499, -0.5102],
        [0.9545, 0.0643, 0.3612],
        [-1.4948, -1.2057, 0.5718]
      ]
    ], grad: [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]] })));
  });
}
var NNTests = { category: "Layers", func: NNTest };

// test/web/Tensor.Grad.test.ts
function TensorGradTest(device) {
  TestRunner.describe("Add", () => {
    const a = new Tensor([1, 2, 3], { device, requires_grad: true });
    const b = new Tensor([4, 5, 6], { device, requires_grad: true });
    const c = a.add(b);
    c.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3], grad: [1, 1, 1] })));
    assert(equal(b, TensorFactory({ data: [4, 5, 6], grad: [1, 1, 1] })));
    assert(equal(c, TensorFactory({ data: [5, 7, 9], grad: [1, 1, 1] })));
  });
  TestRunner.describe("Sub", () => {
    const a = new Tensor([1, 2, 3], { device, requires_grad: true });
    const b = new Tensor([4, 5, 6], { device, requires_grad: true });
    const c = a.sub(b);
    c.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3], grad: [1, 1, 1] })));
    assert(equal(b, TensorFactory({ data: [4, 5, 6], grad: [-1, -1, -1] })));
    assert(equal(c, TensorFactory({ data: [-3, -3, -3], grad: [1, 1, 1] })));
  });
  TestRunner.describe("Mul", () => {
    const a = new Tensor([1, 2, 3], { device, requires_grad: true });
    const b = new Tensor([4, 5, 6], { device, requires_grad: true });
    const c = a.mul(b);
    c.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3], grad: [4, 5, 6] })));
    assert(equal(b, TensorFactory({ data: [4, 5, 6], grad: [1, 2, 3] })));
    assert(equal(c, TensorFactory({ data: [4, 10, 18], grad: [1, 1, 1] })));
  });
  TestRunner.describe("Div", () => {
    const a = new Tensor([1, 2, 3], { device, requires_grad: true });
    const b = new Tensor([4, 5, 6], { device, requires_grad: true });
    const c = a.div(b);
    c.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3], grad: [0.25, 0.2, 0.16666666666666666] })));
    assert(equal(b, TensorFactory({ data: [4, 5, 6], grad: [-0.0625, -0.08, -0.08333333333333333] })));
    assert(equal(c, TensorFactory({ data: [0.25, 0.4, 0.5], grad: [1, 1, 1] })));
  });
  TestRunner.describe("Pow", () => {
    const a = new Tensor([4, 5, 6], { device, requires_grad: true });
    const b = a.pow(2);
    b.backward();
    assert(equal(a, TensorFactory({ data: [4, 5, 6], grad: [8, 10, 12] })));
    assert(equal(b, TensorFactory({ data: [16, 25, 36], grad: [1, 1, 1] })));
  });
  TestRunner.describe("Matmul", () => {
    const a = new Tensor([[1, 2], [3, 4]], { device, requires_grad: true });
    const b = new Tensor([[5, 6], [7, 8]], { device, requires_grad: true });
    const c = a.matmul(b);
    c.backward();
    assert(equal(a, TensorFactory({ data: [[1, 2], [3, 4]], grad: [[11, 15], [11, 15]] })));
    assert(equal(b, TensorFactory({ data: [[5, 6], [7, 8]], grad: [[4, 4], [6, 6]] })));
    assert(equal(c, TensorFactory({ data: [[19, 22], [43, 50]], grad: [[1, 1], [1, 1]] })));
  });
  TestRunner.describe("Sum", () => {
    const a = new Tensor([1, 2, 3], { device, requires_grad: true });
    const c = a.sum();
    c.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3], grad: [1, 1, 1] })));
    assert(equal(c, TensorFactory({ data: [6], grad: [1] })));
  });
  TestRunner.describe("Mean", () => {
    const a = new Tensor([1, 2, 3], { device, requires_grad: true });
    const c = a.mean();
    c.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3], grad: [0.3333333333333333, 0.3333333333333333, 0.3333333333333333] })));
    assert(equal(c, TensorFactory({ data: [2], grad: [1] })));
  });
  TestRunner.describe("Reshape", () => {
    const a = new Tensor([1, 2, 3, 4], { device, requires_grad: true });
    const c = a.reshape([2, 2]);
    c.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3, 4], grad: [1, 1, 1, 1] })));
    assert(equal(c, TensorFactory({ data: [[1, 2], [3, 4]], grad: [[1, 1], [1, 1]] })));
  });
  TestRunner.describe("Exp", () => {
    const a = new Tensor([1, 2, 3, 4], { device, requires_grad: true });
    const c = a.exp();
    c.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3, 4], grad: [2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236] })));
    assert(equal(c, TensorFactory({ data: [2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236], grad: [1, 1, 1, 1] })));
  });
  TestRunner.describe("ReLu", () => {
    const a = new Tensor([-1, 2, 3, 4], { device, requires_grad: true });
    const c = a.relu();
    c.backward();
    assert(equal(a, TensorFactory({ data: [-1, 2, 3, 4], grad: [0, 1, 1, 1] })));
    assert(equal(c, TensorFactory({ data: [0, 2, 3, 4], grad: [1, 1, 1, 1] })));
  });
  TestRunner.describe("Reciprocal", () => {
    const a = new Tensor([1, 2, 3, 4], { device, requires_grad: true });
    const b = a.reciprocal();
    b.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3, 4], grad: [-1, -0.25, -0.1111111111111111, -0.0625] })));
    assert(equal(b, TensorFactory({ data: [1, 0.5, 0.3333333333333333, 0.25], grad: [1, 1, 1, 1] })));
  });
  TestRunner.describe("Sigmoid", () => {
    const a = new Tensor([1, 2, 3, 4], { device, requires_grad: true });
    const b = a.sigmoid();
    b.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3, 4], grad: [0.19661193324148188, 0.1049935854035065, 0.045176659730912144, 0.017662706213291114] })));
    assert(equal(b, TensorFactory({ data: [0.7310585786300049, 0.8807970779778823, 0.9525741268224334, 0.9820137900379085], grad: [1, 1, 1, 1] })));
  });
  TestRunner.describe("Tanh", () => {
    const a = new Tensor([1, 2, 3, 4], { device, requires_grad: true });
    const c = a.tanh();
    c.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3, 4], grad: [0.41997434161402614, 0.07065082485316443, 0.009866037165440211, 0.0013409506830258655] })));
    assert(equal(c, TensorFactory({ data: [0.7615941559557649, 0.9640275800758169, 0.9950547536867305, 0.999329299739067], grad: [1, 1, 1, 1] })));
  });
  TestRunner.describe("Permute", () => {
    const a = new Tensor([[1, 2], [3, 4], [5, 6]], { device, requires_grad: true });
    const c = a.permute([1, 0]);
    c.backward();
    assert(equal(a, TensorFactory({ data: [[1, 2], [3, 4], [5, 6]], grad: [[1, 1], [1, 1], [1, 1]] })));
    assert(equal(c, TensorFactory({ data: [[1, 3, 5], [2, 4, 6]], grad: [[1, 1, 1], [1, 1, 1]] })));
  });
  TestRunner.describe("Transpose", () => {
    const a = new Tensor([[1, 2], [3, 4], [5, 6]], { device, requires_grad: true });
    const c = a.transpose(1, 0);
    c.backward();
    assert(equal(a, TensorFactory({ data: [[1, 2], [3, 4], [5, 6]], grad: [[1, 1], [1, 1], [1, 1]] })));
    assert(equal(c, TensorFactory({ data: [[1, 3, 5], [2, 4, 6]], grad: [[1, 1, 1], [1, 1, 1]] })));
  });
  TestRunner.describe("Abs", () => {
    const a = new Tensor([[-3, 3, 3], [4, -4, 4]], { device, requires_grad: true });
    const b = a.abs();
    b.backward();
    assert(equal(a, TensorFactory({ data: [[-3, 3, 3], [4, -4, 4]], grad: [[-1, 1, 1], [1, -1, 1]] })));
    assert(equal(b, TensorFactory({ data: [[3, 3, 3], [4, 4, 4]], grad: [[1, 1, 1], [1, 1, 1]] })));
  });
  TestRunner.describe("Log", () => {
    const a = new Tensor([[1, 2, 3], [4, 5, 6]], { device, requires_grad: true });
    const b = a.log();
    b.backward();
    assert(equal(a, TensorFactory({ data: [[1, 2, 3], [4, 5, 6]], grad: [[1, 0.5, 0.3333], [0.25, 0.2, 0.1667]] }), 1e-4));
    assert(equal(b, TensorFactory({ data: [[0, 0.6931, 1.0986], [1.3863, 1.6094, 1.7918]], grad: [[1, 1, 1], [1, 1, 1]] }), 1e-4));
  });
  TestRunner.describe("Simple model", () => {
    const weight = new Tensor([[-0.4869, -0.0896], [-51e-4, -0.346], [0.1421, -0.5443]], { device, requires_grad: true });
    let x = new Tensor([[2, 3, -1]], { device, requires_grad: true });
    x = x.matmul(weight);
    const pred = x;
    const loss = pred.sum();
    loss.backward();
    assert(equal(weight, TensorFactory({ data: [[-0.4869, -0.0896], [-51e-4, -0.346], [0.1421, -0.5443]], grad: [[2, 2], [3, 3], [-1, -1]] })));
  });
  TestRunner.describe("Maximum", () => {
    const a = new Tensor([2, 3, 4], { device, requires_grad: true });
    const b = new Tensor([1, 5, 2], { device, requires_grad: true });
    const c = a.maximum(b);
    c.backward();
    assert(equal(a, TensorFactory({ data: [2, 3, 4], grad: [1, 0, 1] })));
    assert(equal(b, TensorFactory({ data: [1, 5, 2], grad: [0, 1, 0] })));
    assert(equal(c, TensorFactory({ data: [2, 5, 4], grad: [1, 1, 1] })));
  });
  TestRunner.describe("Equal", () => {
    const a = new Tensor([1, 2, 3], { device, requires_grad: true });
    const b = new Tensor([0, 2, 2], { device, requires_grad: true });
    const c = a.eq(b);
    c.backward();
    assert(equal(a, TensorFactory({ data: [1, 2, 3], grad: [0, 0, 0] })));
    assert(equal(b, TensorFactory({ data: [0, 2, 2], grad: [0, 0, 0] })));
    assert(equal(c, TensorFactory({ data: [0, 1, 0], grad: [1, 1, 1] })));
  });
}
var TensorGradTests = { category: "Tensor Grad", func: TensorGradTest };

// test/web/Tensor.test.ts
Random.SetRandomSeed(1337);
function TensorTest(device) {
  TestRunner.describe("Tensor creation", () => {
    const a = new Tensor([[1, 2], [3, 4]], { device });
    assert(equal(a, new Tensor([[1, 2], [3, 4]])));
    assert(equal(a.shape, [2, 2]));
    const b = new Tensor([1, 2, 3, 4, 5, 6], { device });
    assert(equal(b, new Tensor([1, 2, 3, 4, 5, 6])));
    assert(equal(b.shape, [6]));
    const c = new Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], { device });
    assert(equal(c, new Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])));
    assert(equal(c.shape, [2, 2, 2]));
  });
  TestRunner.describe("Tensor to", () => {
    const a = new Tensor([[1, 2], [3, 4]], { device });
    assert(a.device === device);
    const otherDevice = device === 0 /* CPU */ ? 1 /* WEBGL */ : 0 /* CPU */;
    const b = a.to(otherDevice);
    assert(b.device === otherDevice);
  });
  TestRunner.describe("Zeros", () => {
    const a = Tensor.zeros([2, 2, 3], { device });
    assert(equal(a, new Tensor([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])));
    assert(equal(a.shape, [2, 2, 3]));
  });
  TestRunner.describe("Ones", () => {
    const a = Tensor.ones([5, 1, 2], { device });
    assert(equal(a, new Tensor([[[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]]])));
    assert(equal(a.shape, [5, 1, 2]));
  });
  TestRunner.describe("arange", () => {
    const a = Tensor.arange(10, 20, 2, { device });
    assert(equal(a, new Tensor([10, 12, 14, 16, 18])));
    assert(equal(a.shape, [5]));
  });
  TestRunner.describe("rand", () => {
    Random.SetRandomSeed(1337);
    const a = Tensor.rand([2, 2], { device });
    assert(equal(a, new Tensor([[0.1844118325971067, 0.2681861550081521], [0.6026948785874993, 0.05738111538812518]])));
    assert(equal(a.shape, [2, 2]));
    const b = Tensor.rand([3, 1, 3], { device });
    assert(equal(b, new Tensor([[[0.4702075123786926, 0.6373465061187744, 0.3192155063152313]], [[0.7714118361473083, 0.441847562789917, 0.3168673813343048]], [[0.5497839450836182, 0.5445157885551453, 0.6433277726173401]]])));
  });
  TestRunner.describe("Reshape", () => {
    const a = new Tensor([0, 1, 2, 3, 4, 5], { device });
    const b = a.reshape([3, 2]);
    assert(equal(b, new Tensor([[0, 1], [2, 3], [4, 5]])));
    assert(equal(b.shape, [3, 2]));
    const c = a.reshape([2, 3]);
    assert(equal(c, new Tensor([[0, 1, 2], [3, 4, 5]])));
    assert(equal(c.shape, [2, 3]));
    const d = new Tensor([[1, 2, 3], [4, 5, 6]], { device });
    const e = d.reshape([6]);
    assert(equal(e, new Tensor([1, 2, 3, 4, 5, 6])));
    assert(equal(e.shape, [6]));
    const f = d.reshape([3, -1]);
    assert(equal(f, new Tensor([[1, 2], [3, 4], [5, 6]])));
    assert(equal(f.shape, [3, 2]));
    const g = d.reshape([-1, 3]);
    const h = d.reshape([3, -1]);
    assert(equal(g, new Tensor([[1, 2, 3], [4, 5, 6]])));
    assert(equal(g.shape, [2, 3]));
    assert(equal(h, new Tensor([[1, 2], [3, 4], [5, 6]])));
    assert(equal(h.shape, [3, 2]));
    const i = new Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], { device });
    const j = i.reshape([-1]);
    assert(equal(j, new Tensor([1, 2, 3, 4, 5, 6, 7, 8])));
    assert(equal(j.shape, [8]));
    const k = i.reshape(-1);
    assert(equal(k, new Tensor([1, 2, 3, 4, 5, 6, 7, 8])));
    assert(equal(k.shape, [8]));
  });
  TestRunner.describe("Broadcasting", () => {
    const a = new Tensor([[1, 1], [2, 2], [3, 3], [4, 4]], { device });
    const b = new Tensor([0.1], { device });
    const c = new Tensor([0.1, 0.2], { device });
    const tensorVector = a.broadcast(b);
    assert(equal(tensorVector[0], new Tensor([[1, 1], [2, 2], [3, 3], [4, 4]])));
    assert(equal(tensorVector[0].shape, [4, 2]));
    assert(equal(tensorVector[1], new Tensor([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])));
    assert(equal(tensorVector[1].shape, [4, 2]));
    const vectorTensor = b.broadcast(a);
    assert(equal(vectorTensor[0], new Tensor([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])));
    assert(equal(vectorTensor[0].shape, [4, 2]));
    assert(equal(vectorTensor[1], new Tensor([[1, 1], [2, 2], [3, 3], [4, 4]])));
    assert(equal(vectorTensor[1].shape, [4, 2]));
    const tensorTensor = a.broadcast(c);
    assert(equal(tensorTensor[0], new Tensor([[1, 1], [2, 2], [3, 3], [4, 4]])));
    assert(equal(tensorTensor[0].shape, [4, 2]));
    assert(equal(tensorTensor[1], new Tensor([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])));
    assert(equal(tensorTensor[1].shape, [4, 2]));
  });
  TestRunner.describe("Binary Ops", () => {
    const a = new Tensor([[1, 1, 1], [2, 2, 2]], { device });
    const b = new Tensor([[3, 3, 3], [4, 4, 4]], { device });
    const c = a.add(b);
    assert(equal(c, new Tensor([[4, 4, 4], [6, 6, 6]])));
    assert(equal(c.shape, [2, 3]));
    const d = a.sub(b);
    assert(equal(d, new Tensor([[-2, -2, -2], [-2, -2, -2]])));
    assert(equal(d.shape, [2, 3]));
    const e = a.mul(b);
    assert(equal(e, new Tensor([[3, 3, 3], [8, 8, 8]])));
    assert(equal(e.shape, [2, 3]));
    const f = new Tensor([[4, 4, 4], [2, 2, 2]]);
    const g = new Tensor([[2, 2, 2], [4, 4, 4]]);
    const h = f.div(g);
    assert(equal(h, new Tensor([[2, 2, 2], [0.5, 0.5, 0.5]])));
    assert(equal(h.shape, [2, 3]));
    const i = f.pow(g);
    assert(equal(i, new Tensor([[16, 16, 16], [16, 16, 16]])));
    assert(equal(i.shape, [2, 3]));
  });
  TestRunner.describe("Negative pow", () => {
    const a = new Tensor([-2, 2, -2], { device });
    const b = new Tensor([2, 2, 2], { device });
    const c = a.pow(b);
    assert(equal(c, new Tensor([4, 4, 4])));
    const d = new Tensor([3, 3, 3], { device });
    const e = a.pow(d);
    assert(equal(e, new Tensor([-8, 8, -8])));
  });
  TestRunner.describe("Binary Ops scalars", () => {
    const a = new Tensor([[1, 1, 1], [2, 2, 2]], { device });
    const b = a.add(10);
    assert(equal(b, new Tensor([[11, 11, 11], [12, 12, 12]])));
    assert(equal(b.shape, [2, 3]));
  });
  TestRunner.describe("Test add with broadcasting", () => {
    const a = new Tensor([[1], [2], [3], [4]], { device });
    const b = new Tensor([0.1], { device });
    const c = new Tensor([0.1, 0.2], { device });
    const d = a.add(b);
    assert(equal(d, new Tensor([[1.1], [2.1], [3.1], [4.1]])));
    assert(equal(d.shape, [4, 1]));
    const e = a.add(c);
    assert(equal(e, new Tensor([[1.1, 1.2], [2.1, 2.2], [3.1, 3.2], [4.1, 4.2]])));
    assert(equal(e.shape, [4, 2]));
  });
  TestRunner.describe("Matmul 1", () => {
    const a = new Tensor([[1, 2], [3, 4]], { device });
    const b = new Tensor([[5, 6], [7, 8]], { device });
    const c = a.matmul(b);
    assert(equal(c, new Tensor([[19, 22], [43, 50]])));
    assert(equal(c.shape, [2, 2]));
  });
  TestRunner.describe("Matmul 2", () => {
    const a = new Tensor([[1, 2], [3, 4], [5, 6]], { device });
    const b = new Tensor([[7], [8]], { device });
    const c = a.matmul(b);
    assert(equal(c, new Tensor([[23], [53], [83]])));
    assert(equal(c.shape, [3, 1]));
  });
  TestRunner.describe("Matmul 3", () => {
    const a = new Tensor([-0.63, -0.46, 0.2], { device }).reshape([3, 1]);
    const b = new Tensor([2], { device }).reshape([1, 1]);
    const c = a.matmul(b);
    assert(equal(c, new Tensor([[-1.26], [-0.92], [0.4]])));
    assert(equal(c.shape, [3, 1]));
  });
  TestRunner.describe("Matmul 4", () => {
    const x = new Tensor([2, 3, -1], { device }).reshape([1, 3]);
    const w = new Tensor([-0.63, -0.46, 0.2], { device }).reshape([3, 1]);
    const d = x.matmul(w);
    assert(equal(d, new Tensor([[-2.8400000000000003]])));
    assert(equal(d.shape, [1, 1]));
  });
  TestRunner.describe("Matmul 5", () => {
    const x = new Tensor([[0.2, 0.3], [-0.4, 0.8], [-0.3, 0.9], [0.5, 0.3]], { device });
    const w = new Tensor([[-0.47595065], [-0.68263206]], { device });
    const xw = x.matmul(w);
    assert(equal(xw, new Tensor([[-0.299979748], [-0.3557253880000001], [-0.471583659], [-0.44276494299999997]])));
    assert(equal(xw.shape, [4, 1]));
  });
  TestRunner.describe("Matmul 6", () => {
    const a = new Tensor([[-2.026, -2.0655, -1.2054], [-0.9122, -1.2502, 0.8032]], { device });
    const b = new Tensor([[-0.2071, 0.0544], [0.1378, -0.3889], [0.5133, 0.3319]], { device });
    const r = a.matmul(b);
    assert(equal(r, new Tensor([[-0.4837731200000001, 0.2929862900000002], [0.42892162, 0.7031611799999999]])));
    assert(equal(r.shape, [2, 2]));
  });
  TestRunner.describe("Matmul 7", () => {
    const a = new Tensor([[1, 1], [1, 1]], { device });
    const b = new Tensor([[-0.2071, 0.1378, 0.5133], [0.0544, -0.3889, 0.3319]], { device });
    const r = a.matmul(b);
    assert(equal(r, new Tensor([[-0.1527, -0.2511, 0.8452], [-0.1527, -0.2511, 0.8452]])));
    assert(equal(r.shape, [2, 3]));
  });
  TestRunner.describe("Matmul with permute", () => {
    const x = new Tensor([[1, 2], [3, 4]], { device });
    const w = new Tensor([[-0.5963, -62e-4], [0.1741, -0.1097], [-0.4237, -0.6666], [0.1204, 0.2781]], { device });
    const wP = w.permute([-1, -2]);
    const y = x.matmul(wP);
    assert(equal(y, new Tensor([[-0.6087, -0.0453, -1.7569, 0.6766], [-1.8137, 0.0835, -3.9375, 1.4736]])));
    assert(equal(y.shape, [2, 4]));
  });
  TestRunner.describe("Sum", () => {
    const a = new Tensor([0.5, 1.5], { device });
    const b = new Tensor([[1, 2], [3, 4]], { device });
    const c = new Tensor([[0, 1], [0, 5]], { device });
    const d = a.sum();
    assert(equal(d, new Tensor([2])));
    assert(equal(d.shape, [1]));
    const e = b.sum();
    assert(equal(e, new Tensor([10])));
    assert(equal(e.shape, [1]));
    const f = c.sum(0);
    assert(equal(f, new Tensor([0, 6])));
    assert(equal(f.shape, [2]));
    const g = c.sum(1);
    assert(equal(g, new Tensor([1, 5])));
    assert(equal(g.shape, [2]));
    const h = c.sum(-2);
    assert(equal(h, new Tensor([0, 6])));
    assert(equal(h.shape, [2]));
    const i = new Tensor([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0]], { device });
    assert(equal(i.sum(null, true), new Tensor([[6]])));
    assert(equal(i.sum(0, true), new Tensor([[2, 4, 0]])));
    assert(equal(i.sum(0, false), new Tensor([2, 4, 0])));
    assert(equal(i.sum(1, true), new Tensor([[0], [1], [2], [1], [2]])));
    assert(equal(i.sum(1, false), new Tensor([0, 1, 2, 1, 2])));
    const x = new Tensor([
      [
        [0, 1],
        [2, 3]
      ],
      [
        [4, 5],
        [6, 7]
      ]
    ], { device });
    assert(equal(x.sum(), new Tensor([28])));
    assert(equal(x.sum(0), new Tensor([[4, 6], [8, 10]])));
    assert(equal(x.sum(1), new Tensor([[2, 4], [10, 12]])));
    assert(equal(x.sum(2), new Tensor([[1, 5], [9, 13]])));
    assert(equal(x.sum(-1), new Tensor([[1, 5], [9, 13]])));
    assert(equal(x.sum(null, true), new Tensor([[[28]]])));
    assert(equal(x.sum(0, true), new Tensor([[[4, 6], [8, 10]]])));
    assert(equal(x.sum(1, true), new Tensor([[[2, 4]], [[10, 12]]])));
    assert(equal(x.sum(2, true), new Tensor([[[1], [5]], [[9], [13]]])));
    assert(equal(x.sum(-1, true), new Tensor([[[1], [5]], [[9], [13]]])));
    const y = new Tensor([
      [
        [
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]
        ],
        [
          [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]
          ]
        ],
        [
          [
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27]
          ]
        ]
      ]
    ], { device });
    assert(equal(y.sum(), new Tensor([378])));
    assert(equal(y.sum(0), new Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[10, 11, 12], [13, 14, 15], [16, 17, 18]]], [[[19, 20, 21], [22, 23, 24], [25, 26, 27]]]])));
    assert(equal(y.sum(1), new Tensor([[[[30, 33, 36], [39, 42, 45], [48, 51, 54]]]])));
    assert(equal(y.sum(2), new Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]])));
    assert(equal(y.sum(3), new Tensor([[[[12, 15, 18]], [[39, 42, 45]], [[66, 69, 72]]]])));
    assert(equal(y.sum(4), new Tensor([[[[6, 15, 24]], [[33, 42, 51]], [[60, 69, 78]]]])));
    assert(equal(y.sum(-1), new Tensor([[[[6, 15, 24]], [[33, 42, 51]], [[60, 69, 78]]]])));
    assert(equal(y.sum(-2), new Tensor([[[[12, 15, 18]], [[39, 42, 45]], [[66, 69, 72]]]])));
  });
  TestRunner.describe("Mean", () => {
    const a = new Tensor([[1, 2], [3, 4]], { device });
    assert(equal(a.mean(), new Tensor([2.5])));
    assert(equal(a.mean(null, true), new Tensor([[2.5]])));
    assert(equal(a.mean(0, true), new Tensor([[2, 3]])));
    assert(equal(a.mean(0, false), new Tensor([2, 3])));
    assert(equal(a.mean(1, true), new Tensor([[1.5], [3.5]])));
    assert(equal(a.mean(1, false), new Tensor([1.5, 3.5])));
    const b = new Tensor([[[0, 1, 2], [3, 4, 5]]], { device });
    assert(equal(b.mean(2, true), new Tensor([[[1], [4]]])));
    assert(equal(b.mean(-1, true), new Tensor([[[1], [4]]])));
    assert(equal(b.mean(-1, false), new Tensor([[1, 4]])));
  });
  TestRunner.describe("Var", () => {
    const a = new Tensor([[1, 2], [3, 4]], { device });
    assert(equal(a.var(), new Tensor([1.25])));
    assert(equal(a.var(null, true), new Tensor([[1.25]])));
    assert(equal(a.var(0, true), new Tensor([[1, 1]])));
    assert(equal(a.var(0, false), new Tensor([1, 1])));
    assert(equal(a.var(1, true), new Tensor([[0.25], [0.25]])));
    assert(equal(a.var(1, false), new Tensor([0.25, 0.25])));
  });
  TestRunner.describe("Abs", () => {
    const a = new Tensor([[-3, 3, 3], [4, -4, 4]], { device });
    assert(equal(a.abs(), new Tensor([[3, 3, 3], [4, 4, 4]])));
  });
  TestRunner.describe("Log", () => {
    const a = new Tensor([[1, 2, 3], [4, 5, 6]], { device });
    assert(equal(a.log(), new Tensor([[0, 0.6931, 1.0986], [1.3863, 1.6094, 1.7918]]), 1e-4));
  });
  TestRunner.describe("Equal", () => {
    const a = new Tensor([1, 2, 3], { device });
    const b = new Tensor([0, 2, 2], { device });
    const c = a.eq(b);
    assert(equal(c, new Tensor([0, 1, 0])));
    const d = new Tensor([3], { device });
    const e = a.eq(d);
    assert(equal(e, new Tensor([0, 0, 1])));
  });
  TestRunner.describe("Not equal", () => {
    const a = new Tensor([1, 2, 3], { device });
    const b = new Tensor([0, 2, 2], { device });
    const c = a.ne(b);
    assert(equal(c, new Tensor([1, 0, 1])));
    const d = new Tensor([3], { device });
    const e = a.ne(d);
    assert(equal(e, new Tensor([1, 1, 0])));
  });
  TestRunner.describe("Greater than", () => {
    const a = new Tensor([1, 2, 3], { device });
    const b = new Tensor([0, 2, 2], { device });
    const c = a.gt(2);
    assert(equal(c, new Tensor([0, 0, 1])));
    const d = a.gt(b);
    assert(equal(d, new Tensor([1, 0, 1])));
    const e = a.gte(b);
    assert(equal(e, new Tensor([1, 1, 1])));
  });
  TestRunner.describe("Less than", () => {
    const a = new Tensor([1, 2, 3], { device });
    const b = new Tensor([0, 2, 2], { device });
    const c = a.lt(2);
    assert(equal(c, new Tensor([1, 0, 0])));
    const d = a.lt(b);
    assert(equal(d, new Tensor([0, 0, 0])));
    const e = a.lte(b);
    assert(equal(e, new Tensor([0, 1, 0])));
    const f = new Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]);
    const g = new Tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]);
    const h = f.lte(g);
    assert(equal(h, new Tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]])));
  });
  TestRunner.describe("Less than equal", () => {
    const a = new Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], { device });
    const b = new Tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], { device });
    const c = a.lte(b);
    assert(equal(c, new Tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]])));
  });
  TestRunner.describe("Where using scalar", () => {
    const a = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], { device });
    const c = Tensor.where(a.lt(5), a, a.mul(10));
    assert(equal(c, new Tensor([1, 2, 3, 4, 50, 60, 70, 80, 90])));
  });
  TestRunner.describe("Where using matrix", () => {
    const x = new Tensor([[1, 2], [3, 4]], { device });
    const y = new Tensor([[9, 8], [7, 6]], { device });
    const condition = new Tensor([[1, 0], [1, 1]], { device });
    const c = Tensor.where(condition, x, y);
    assert(equal(c, new Tensor([[1, 8], [3, 4]])));
  });
  TestRunner.describe("Where using permute", () => {
    const a = new Tensor([[1, 2], [3, 4]], { device });
    const b = new Tensor([[0, 0], [1, 1]], { device });
    const c = Tensor.where(b.eq(0), a, a.mul(10));
    assert(equal(c, new Tensor([[1, 2], [30, 40]])));
    const d = Tensor.where(b.eq(0), a.T, a.T.mul(10));
    assert(equal(d, new Tensor([[1, 3], [20, 40]])));
  });
  TestRunner.describe("Maximum", () => {
    const a = new Tensor([2, 3, 4], { device });
    const b = new Tensor([1, 5, 2], { device });
    const c = a.maximum(b);
    assert(equal(c, new Tensor([2, 5, 4])));
    assert(equal(c.shape, [3]));
  });
  TestRunner.describe("Expand", () => {
    const a = new Tensor([[1], [2], [3]], { device });
    assert(equal(a.expand([3, 4]), new Tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])));
    assert(equal(a.expand([-1, 4]), new Tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])));
    const b = new Tensor([[1, 2, 3], [4, 5, 6]], { device });
    assert(equal(b.expand([4, 2, 3]), new Tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])));
    const c = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], { device });
    assert(equal(c.expand([3, 3, 3]), new Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])));
    const d = new Tensor([1, 2, 3], { device });
    assert(equal(d.expand([3, 3]), new Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])));
    const e = new Tensor([[[1], [2]], [[3], [4]]], { device });
    assert(equal(e.expand([2, 2, 3]), new Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])));
  });
  TestRunner.describe("Unsqueeze", () => {
    const a = new Tensor([1, 2, 3, 4], { device });
    const b = a.unsqueeze(0);
    assert(equal(b, new Tensor([[1, 2, 3, 4]])));
    assert(equal(b.shape, [1, 4]));
    const c = a.unsqueeze(1);
    assert(equal(c, new Tensor([[1], [2], [3], [4]])));
    assert(equal(c.shape, [4, 1]));
  });
  TestRunner.describe("Masked fill", () => {
    const a = new Tensor([1, 2, 3, 4, 5], { device });
    const mask = new Tensor([1, 0, 0, 0, 0], { device });
    const filled = a.masked_fill(mask, 10);
    assert(equal(filled, new Tensor([10, 2, 3, 4, 5])));
  });
  TestRunner.describe("Permute", () => {
    const a = new Tensor([[1, 2], [3, 4]], { device });
    assert(equal(a.permute(), new Tensor([[1, 3], [2, 4]])));
    const b = new Tensor([1, 2, 3, 4], { device });
    assert(equal(b.permute(), new Tensor([1, 2, 3, 4])));
    const c = new Tensor([[[1, 2, 3], [4, 5, 6]]], { device });
    assert(equal(c.permute([1, 0, 2]), new Tensor([[[1, 2, 3]], [[4, 5, 6]]])));
    const d = Tensor.ones([2, 3, 4, 5], { device });
    assert(equal(d.permute(), Tensor.ones([5, 4, 3, 2])));
    const e = new Tensor([[1, 2], [3, 4]], { device });
    assert(equal(e.permute(), new Tensor([[1, 3], [2, 4]])));
    assert(equal(e.permute().reshape([1, 4]), new Tensor([[1, 3, 2, 4]])));
  });
  TestRunner.describe("Transpose", () => {
    const a = new Tensor([[1.0028, -0.9893, 0.5809], [-0.1669, 0.7299, 0.4942]], { device });
    assert(equal(a.transpose(0, 1), new Tensor([[1.0028, -0.1669], [-0.9893, 0.7299], [0.5809, 0.4942]])));
  });
  TestRunner.describe("Softmax", () => {
    const a = Tensor.arange(0, 10, 1, { device });
    assert(equal(a.softmax(0), new Tensor([1e-4, 2e-4, 6e-4, 16e-4, 43e-4, 0.0116, 0.0315, 0.0856, 0.2326, 0.6321]), 1e-4));
    const b = new Tensor([[5, 6, 3]], { device });
    assert(equal(b.softmax(-1), new Tensor([[0.2595, 0.7054, 0.0351]]), 1e-4));
  });
  TestRunner.describe("Slice", () => {
    const a = new Tensor([
      [
        [0, 1, 2, 3],
        [4, 5, 6, 7]
      ],
      [
        [8, 9, 10, 11],
        [12, 13, 14, 15]
      ],
      [
        [16, 17, 18, 19],
        [20, 21, 22, 23]
      ]
    ], { device });
    assert(equal(a.slice([null, null, 0]), new Tensor([[0, 4], [8, 12], [16, 20]])));
    assert(equal(a.slice([null, null, 1]), new Tensor([[1, 5], [9, 13], [17, 21]])));
    assert(equal(a.slice([null, null, 2]), new Tensor([[2, 6], [10, 14], [18, 22]])));
    assert(equal(a.slice([null, null, [1, 2]]), new Tensor([[[1], [5]], [[9], [13]], [[17], [21]]])));
    assert(equal(a.slice([null, [1, 2], [1, 2]]), new Tensor([[[5]], [[13]], [[21]]])));
  });
  TestRunner.describe("Contiguous", () => {
    const a = new Tensor([
      [
        [0, 1],
        [2, 3]
      ],
      [
        [4, 5],
        [6, 7]
      ]
    ], { device });
    const b = a.T;
    assert(equal(b, new Tensor([[[0, 4], [2, 6]], [[1, 5], [3, 7]]])));
    assert(equal(b.shape, [2, 2, 2]));
    assert(equal(b.strides, [1, 2, 4]));
    const c = b.contiguous();
    assert(equal(c, new Tensor([[[0, 4], [2, 6]], [[1, 5], [3, 7]]])));
    assert(equal(c.shape, [2, 2, 2]));
    assert(equal(c.strides, [4, 2, 1]));
    const d = new Tensor([
      [0, 1],
      [2, 3],
      [4, 5],
      [6, 7]
    ], { device });
    const e = d.T;
    const f = e.reshape([2, 4]);
    assert(equal(f, new Tensor([[0, 2, 4, 6], [1, 3, 5, 7]])));
    assert(equal(f.shape, [2, 4]));
    assert(equal(f.strides, [4, 1]));
    const g = a.reshape([2, 4]);
    assert(equal(g, new Tensor([[0, 1, 2, 3], [4, 5, 6, 7]])));
    assert(equal(g.shape, [2, 4]));
    assert(equal(g.strides, [4, 1]));
  });
  TestRunner.describe("Tril", () => {
    const a = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], { device });
    const b = a.tril();
    assert(equal(b, new Tensor([[1, 0, 0], [4, 5, 0], [7, 8, 9], [10, 11, 12]])));
    assert(equal(b.shape, [4, 3]));
  });
  TestRunner.describe("Triu", () => {
    const a = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], { device });
    const b = a.triu();
    assert(equal(b, new Tensor([[1, 2, 3], [0, 5, 6], [0, 0, 9], [0, 0, 0]])));
    assert(equal(b.shape, [4, 3]));
  });
}
var TensorTests = { category: "Tensor", func: TensorTest };

// test/web/Test.test.ts
function TestTest(device) {
  TestRunner.describe("Matmul", () => {
    const a = new Tensor([[1, 2], [3, 4]], { device, requires_grad: true });
    const b = new Tensor([[5, 6], [7, 8]], { device, requires_grad: true });
    const c = a.matmul(b);
    c.backward();
    console.log(`a ${a}`);
    assert(equal(a, TensorFactory({ data: [[1, 2], [3, 4]], grad: [[11, 15], [11, 15]] })));
    assert(equal(b, TensorFactory({ data: [[5, 6], [7, 8]], grad: [[4, 4], [6, 6]] })));
    assert(equal(c, TensorFactory({ data: [[19, 22], [43, 50]], grad: [[1, 1], [1, 1]] })));
  });
}
var TestTests = { category: "Test", func: TestTest };

// test/run-web.ts
var TestRunner = class {
  static describe(name, func) {
    throw Error("Not implemented");
  }
};
TestRunner.UnitTests = [
  TestTests,
  // SumTests
  TensorTests,
  TensorGradTests,
  NNTests
  // NetworkTests,
  // NetworksTests
];

// test/web/Network.test.ts
function NetworkTest(device) {
  TestRunner.describe("1 dense layer", () => {
    Random.SetRandomSeed(1337);
    class SingleLayerModel extends Module {
      constructor(input_sample, output_sample) {
        super();
        this.dense1 = new nn_exports.Linear(input_sample, output_sample);
      }
      forward(x) {
        x = this.dense1.forward(x);
        return x;
      }
    }
    const Xb = new Tensor([
      [0.2, 0.3],
      [-0.4, 0.8],
      [-0.3, 0.9],
      [0.5, 0.3]
    ], { device, requires_grad: true });
    const yb = new Tensor([[1], [0.2], [0.3], [0.7]], { device, requires_grad: true });
    let model = new SingleLayerModel(2, 1);
    model.dense1.weight = new Tensor([[-0.5963, -62e-4]]);
    model.dense1.bias = new Tensor([0.1741]);
    model = model.to(device);
    const output = model.forward(Xb);
    const loss = output.sub(yb).pow(2).mean();
    model.zero_grad();
    loss.backward();
    const learning_rate = 0.01;
    for (let p of model.parameters()) {
      p.data = p.sub(new Tensor(p.grad, { device }).mul(learning_rate)).data;
    }
    assert(equal(loss, TensorFactory({ data: [0.40608614805], grad: [1] })));
    assert(equal(model.dense1.weight, TensorFactory({ data: [[-0.59280177, -458459e-8]], grad: [[-0.349823, -0.16154099999999993]] })));
    assert(equal(model.dense1.bias, TensorFactory({ data: [0.1816893], grad: [-0.75893] })));
  });
  TestRunner.describe("3 dense layers", () => {
    Random.SetRandomSeed(1337);
    class SimpleModel extends Module {
      constructor(input_sample, output_sample) {
        super();
        this.dense1 = new nn_exports.Linear(input_sample, 4);
        this.dense2 = new nn_exports.Linear(4, 4);
        this.dense3 = new nn_exports.Linear(4, output_sample);
      }
      forward(x) {
        x = this.dense1.forward(x);
        x = this.dense2.forward(x);
        x = this.dense3.forward(x);
        return x;
      }
    }
    function get_loss(model2, x_tensor, y_tensor) {
      const pred = model2.forward(x_tensor).tanh();
      const data_loss = pred.sub(y_tensor).pow(2).sum();
      let reg_loss = new Tensor([0], { device, requires_grad: true });
      for (let p of model2.parameters()) {
        const w_sum = p.mul(p).sum();
        const p_shape_prod = p.data.shape.reduce((p2, c) => p2 * c);
        const div = w_sum.div(new Tensor(p_shape_prod, { device, requires_grad: true }));
        reg_loss = reg_loss.add(div);
      }
      const alpha = new Tensor(1e-4, { device, requires_grad: true });
      const total_loss = data_loss.mean().add(reg_loss.mul(alpha));
      return total_loss;
    }
    const Xb = new Tensor([
      [2, 3, -1],
      [3, -1, 0.5],
      [0.5, 1, 1],
      [1, 1, -1]
    ], { device, requires_grad: true });
    const Y = new Tensor([1, 0.2, 0.3, 0.7], { device, requires_grad: true });
    const Yb = Y.reshape([Y.shape[0], 1]);
    let model = new SimpleModel(3, 2);
    model.dense1.weight = new Tensor([[-0.4869, -51e-4, 0.1421], [-0.0896, -0.346, -0.5443], [0.0983, 0.2271, -0.374], [-0.2777, 0.2408, 0.0935]], { requires_grad: true });
    model.dense1.bias = new Tensor([-0.5111, 0.3082, 0.4363, -0.2963], { requires_grad: true });
    model.dense2.weight = new Tensor([[0.1005, 0.2079, 0.0102, -0.0935], [0.3864, -0.1422, 0.3963, 0.4639], [-0.4852, 0.2358, 0.2884, 0.4469], [-0.0344, 0.3378, -0.3731, -0.2868]], { requires_grad: true });
    model.dense2.bias = new Tensor([-0.2056, -0.1323, -0.017, 0.1752], { requires_grad: true });
    model.dense3.weight = new Tensor([[0.0226, 0.0536, 0.0701, -0.2519], [0.104, 0.2077, -0.421, 0.2629]], { requires_grad: true });
    model.dense3.bias = new Tensor([0.2708, -0.4257], { requires_grad: true });
    model = model.to(device);
    let last_loss;
    const epochs = 100;
    for (let k = 0; k < epochs; k++) {
      const total_loss = get_loss(model, Xb, Yb);
      model.zero_grad();
      total_loss.backward();
      const learning_rate = 0.1;
      for (let p of model.parameters()) {
        p.data = p.sub(new Tensor(p.grad).mul(learning_rate)).data;
      }
      const total_loss2 = get_loss(model, Xb, Yb);
      last_loss = total_loss2;
    }
    assert(equal(last_loss, TensorFactory({ data: [0.017161300405859947], grad: [0] }), 1e-3));
  });
}
var NetworkTests = { category: "Network", func: NetworkTest };

// test/web/networks/MoonsData/MoonsData.test.ts
function NetworkMoonsData(device) {
  TestRunner.describe("MoonsData test", () => {
    Random.SetRandomSeed(1337);
    function generateData(n) {
      var data = [];
      var labels = [];
      for (var i = 0; i < Math.PI; i += Math.PI * 2 / n) {
        var point_1 = [
          Math.cos(i) + Random.RandomRange(-0.1, 0.1),
          Math.sin(i) + Random.RandomRange(-0.1, 0.1)
        ];
        data.push(point_1);
        labels.push(-1);
        var point_2 = [
          1 - Math.cos(i) + Random.RandomRange(-0.1, 0.1),
          1 - Math.sin(i) + Random.RandomRange(-0.1, 0.1) - 0.5
        ];
        data.push(point_2);
        labels.push(1);
      }
      return [data, labels];
    }
    class SimpleModel extends Module {
      constructor(input_sample, output_sample) {
        super();
        this.dense1 = new nn_exports.Linear(input_sample, 16);
        this.dense2 = new nn_exports.Linear(16, 16);
        this.dense3 = new nn_exports.Linear(16, output_sample);
      }
      forward(x) {
        x = this.dense1.forward(x).tanh();
        x = this.dense2.forward(x).tanh();
        x = this.dense3.forward(x).tanh();
        return x;
      }
    }
    const [Xb, yb] = generateData(100);
    const X = new Tensor(Xb, { device, requires_grad: true });
    const y = new Tensor(yb, { device, requires_grad: true }).reshape([100, 1]);
    const model = new SimpleModel(2, 1).to(device);
    let last_loss = new Tensor(0, { device });
    const epochs = 100;
    for (let k = 0; k < epochs; k++) {
      const pred = model.forward(X);
      const loss = pred.sub(y).pow(2).mean();
      model.zero_grad();
      loss.backward();
      for (let p of model.parameters()) {
        p.data = p.sub(new Tensor(p.grad).mul(0.5)).data;
      }
      last_loss = loss;
    }
    assert(equal(last_loss, TensorFactory({ data: [3e-3], grad: [1] }), 1e-4));
  });
}

// test/web/networks/Networks.test.ts
function NetworksTest(device) {
  NetworkMoonsData(device);
}
var NetworksTests = { category: "Networks", func: NetworksTest };

// test/src/TableBody.ts
var TableBody = class {
  constructor(table) {
    this.body = table.createTBody();
  }
  addEntry(entry) {
    const row = this.body.insertRow();
    if (entry instanceof HTMLElement) {
      row.appendChild(entry);
    } else {
      for (let cell of entry) {
        const cellElement = row.insertCell();
        cellElement.textContent = cell;
      }
    }
    return row;
  }
  updateEntry(entry, entryData) {
    if (entry.cells.length != entryData.length) {
      throw Error(`Number of entries in existing entry (${entry.cells.length}) don't match updated entry count (${updatedEntry.length})`);
    }
    for (let i = 0; i < entry.cells.length; i++) {
      if (entryData[i] === null)
        continue;
      const cell = entry.cells.item(i);
      cell.textContent = entryData[i];
    }
    return true;
  }
};

// test/src/TableHeader.ts
var TableHeader = class {
  constructor(table) {
    this.head = table.createTHead();
  }
  addEntry(entries) {
    const headerEntry = this.head.insertRow();
    for (let entry of entries) {
      const entryCell = headerEntry.insertCell();
      entryCell.textContent = entry;
    }
  }
};

// test/src/Table.ts
var Table = class {
  constructor(container) {
    this.container = container;
    this.table = document.createElement("table");
    this.tableHead = new TableHeader(this.table);
    this.tableBody = new TableBody(this.table);
    this.container.appendChild(this.table);
  }
};

// test/src/Test.ts
var TestResult = /* @__PURE__ */ ((TestResult2) => {
  TestResult2[TestResult2["PENDING"] = 0] = "PENDING";
  TestResult2[TestResult2["RUNNING"] = 1] = "RUNNING";
  TestResult2[TestResult2["PASSED"] = 2] = "PASSED";
  TestResult2[TestResult2["FAILED"] = 3] = "FAILED";
  return TestResult2;
})(TestResult || {});
var Test = class {
  constructor(category, name, device, func) {
    this.category = category;
    this.name = name;
    this.device = device;
    this.func = func;
    this.duration = 0;
    this.result = 0 /* PENDING */;
  }
  async run() {
    return new Promise((resolve) => {
      TestRunner.describe = async (name, func) => {
        try {
          const st = performance.now();
          func();
          this.duration = performance.now() - st;
          this.result = 2 /* PASSED */;
          resolve(true);
          return true;
        } catch (error) {
          console.log(error);
          this.resultDescription = error;
          this.result = 3 /* FAILED */;
          resolve(false);
        }
      };
      this.func();
    });
  }
};

// test/src/index.ts
var UnitTests = [
  // TestTests,
  // SumTests
  TensorTests,
  TensorGradTests,
  NNTests,
  NetworkTests,
  NetworksTests
];
var WebGradTests = class {
  constructor(container) {
    this.tests = [];
    this.table = new Table(container);
    this.table.tableHead.addEntry(["Category", "Duration", "Passed", "Failed"]);
    for (let unitTest of UnitTests) {
      const categoryEntry = this.table.tableBody.addEntry([unitTest.category, "-", "-", "-"]);
      categoryEntry.classList.add("category");
      const categortTable = this.createCategoryTable(unitTest.category);
      categortTable.container.style.display = "none";
      categoryEntry.addEventListener("click", (event) => {
        categortTable.container.style.display = categortTable.container.style.display === "none" ? "" : "none";
      });
      this.table.tableBody.addEntry(categortTable.container);
      for (let deviceKey of Object.keys(Device)) {
        if (!isNaN(parseInt(deviceKey)))
          continue;
        const device = Device[deviceKey];
        TestRunner.describe = (name, func) => {
          const unitTestFunction = () => {
            TestRunner.describe(name, func);
          };
          const test = new Test(unitTest.category, name, device, unitTestFunction);
          const testElement = categortTable.table.tableBody.addEntry([name, deviceKey, "-", TestResult[0 /* PENDING */]]);
          this.tests.push({
            categoryElement: categoryEntry,
            categoryTable: categortTable.table,
            testElement,
            test
          });
        };
        unitTest.func(device);
      }
    }
    setTimeout(() => {
      this.runTests();
    }, 1e3);
  }
  async runTests() {
    for (let unitTest of this.tests) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      unitTest.categoryTable.tableBody.updateEntry(unitTest.testElement, [
        null,
        null,
        null,
        TestResult[1 /* RUNNING */]
      ]);
      const result = await unitTest.test.run();
      const testResult = result ? TestResult[2 /* PASSED */] : TestResult[3 /* FAILED */];
      unitTest.categoryTable.tableBody.updateEntry(unitTest.testElement, [
        null,
        null,
        `${unitTest.test.duration.toFixed(4)}ms`,
        testResult
      ]);
      if (!result)
        unitTest.testElement.classList.add("failed");
      else
        unitTest.testElement.classList.add("passed");
      this.updateCategoryStats(unitTest.categoryElement);
    }
  }
  updateCategoryStats(category) {
    let duration = 0;
    let passed = 0;
    let failed = 0;
    let testsPerCategory = 0;
    for (let unitTest of this.tests) {
      if (unitTest.categoryElement !== category)
        continue;
      if (unitTest.test.result === 2 /* PASSED */)
        passed++;
      else if (unitTest.test.result === 3 /* FAILED */)
        failed++;
      duration += unitTest.test.duration;
      testsPerCategory++;
    }
    if (passed + failed === testsPerCategory) {
      if (failed > 0)
        category.classList.add("failed");
      else
        category.classList.add("passed");
    }
    this.table.tableBody.updateEntry(category, [null, `${duration.toFixed(4)}ms`, passed.toString(), failed.toString()]);
  }
  createCategoryTable(category) {
    const testEntrySpanned = document.createElement("td");
    testEntrySpanned.colSpan = 4;
    const testEntryTable = new Table(testEntrySpanned);
    testEntryTable.tableHead.addEntry(["Test", "Device", "Duration", "Result"]);
    return { container: testEntrySpanned, table: testEntryTable };
  }
};
export {
  WebGradTests
};
