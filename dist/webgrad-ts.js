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
  copyToShape(shape) {
    const inputTexture = this.texture;
    const outputTexture = Texture.createUnpackedFromShape(null, shape.slice());
    const uniforms = [
      { name: "widthIn", value: inputTexture.width, type: 4 /* INT */ },
      { name: "widthOut", value: outputTexture.width, type: 4 /* INT */ }
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
var Device = /* @__PURE__ */ ((Device2) => {
  Device2[Device2["CPU"] = 0] = "CPU";
  Device2[Device2["WEBGL"] = 1] = "WEBGL";
  return Device2;
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
  get device() {
    return this.data.device;
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
  split(split_sizes, dim = 0) {
    const matrix = this;
    const dimSize = matrix.shape[dim];
    let splitIntervals;
    if (typeof split_sizes === "number") {
      const numFullChunks = Math.floor(dimSize / split_sizes);
      const lastChunkSize = dimSize - numFullChunks * split_sizes;
      splitIntervals = Array(numFullChunks).fill(split_sizes);
      if (lastChunkSize > 0) {
        splitIntervals.push(lastChunkSize);
      }
    } else {
      splitIntervals = split_sizes;
      if (splitIntervals.reduce((acc, val) => acc + val, 0) !== dimSize) {
        throw new Error("Sum of split sizes must equal the size of the dimension being split.");
      }
    }
    const out = [];
    let start = 0;
    for (let size of splitIntervals) {
      let end = start + size;
      let sliceIndices = matrix.shape.map((size2, index) => {
        if (index === dim) {
          return [start, end];
        }
        return [0, size2];
      });
      const chunk = matrix.slice(sliceIndices);
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
      const modelTensor = namedParameters[path];
      const t = new Tensor(tensor, { device: modelTensor.device });
      const stateTensorShape = t.shape.reduce((p, c) => p * c);
      const modelParameterShape = modelTensor.shape.reduce((p, c) => p * c);
      if (stateTensorShape != modelParameterShape)
        throw Error(`State tensor shape (${stateTensorShape}) doesn't match model tensor shape (${modelParameterShape})`);
      modelTensor.assign(t);
    }
  }
  to(device) {
    for (let parameter of this.parameters()) {
      parameter.to(device);
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
export {
  Device,
  Module,
  Operations_exports as Operations,
  Random,
  Tensor,
  nn_exports as nn
};
