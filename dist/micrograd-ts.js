var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
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
      throw Error(`Indices ${indices} must match matrix shape ${m.shape}`);
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
  static split(matrix, split_sizes, dim = null) {
    if (Array.isArray(split_sizes))
      throw Error("Split split_sizes as array not supported");
    if (dim !== null)
      throw Error("Split dim not supported");
    const chunkSize = split_sizes;
    const stride = matrix.shape[matrix.shape.length - 1];
    if (stride % chunkSize !== 0) {
      throw new Error("Invalid chunk size, not evently divisible into last tensor dimension");
    }
    const out = [];
    const chunks = stride / chunkSize;
    for (let c = 0; c < chunks; c++) {
      out.push(Matrix.zeros([...matrix.shape.slice(0, matrix.shape.length - 1), chunkSize]));
    }
    const outOffsets = out.map((_) => 0);
    let sourceOffset = 0;
    const macroChunks = matrix.data.length / stride;
    for (let i = 0; i < macroChunks; i++) {
      for (let j = 0; j < chunks; j++) {
        out[j].data.set(matrix.data.slice(sourceOffset, sourceOffset + chunkSize), outOffsets[j]);
        outOffsets[j] += chunkSize;
        sourceOffset += chunkSize;
      }
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
    if (!(m1 instanceof Matrix) && !(m2 instanceof Matrix)) {
      m1 = new Matrix([m1]);
      m2 = new Matrix([m2]);
    }
    let _m1 = m1 instanceof Matrix ? m1 : new Matrix([m1]);
    let _m2 = m2 instanceof Matrix ? m2 : new Matrix([m2]);
    let [_m1b, _m2b] = Matrix.broadcast(_m1, _m2);
    const _m1bShape = _m1b.shape.slice();
    const s = Matrix.reshape1d(_m1b.shape);
    _m1b = _m1b.reshape(s);
    _m2b = _m2b.reshape(s);
    let newData = new Float32Array(_m1b.shape.reduce((p, c) => p * c));
    for (let i = 0; i < newData.length; i++) {
      let value = 0;
      if (op == 0 /* ADD */)
        value = _m1b.get1D(i) + _m2b.get1D(i);
      else if (op == 1 /* SUB */)
        value = _m1b.get1D(i) - _m2b.get1D(i);
      else if (op == 2 /* MUL */)
        value = _m1b.get1D(i) * _m2b.get1D(i);
      else if (op == 3 /* DIV */)
        value = _m1b.get1D(i) / _m2b.get1D(i);
      else if (op == 4 /* POW */)
        value = _m1b.get1D(i) ** _m2b.get1D(i);
      newData[i] = isNaN(value) ? 0 : value;
    }
    return new Matrix(newData, _m1bShape);
  }
  static add(m1, m2) {
    return Matrix.binary_op(m1, m2, 0 /* ADD */);
  }
  static sub(m1, m2) {
    return Matrix.binary_op(m1, m2, 1 /* SUB */);
  }
  static mul(m1, m2) {
    return Matrix.binary_op(m1, m2, 2 /* MUL */);
  }
  static div(m1, m2) {
    return Matrix.binary_op(m1, m2, 3 /* DIV */);
  }
  static pow(m1, m2) {
    return Matrix.binary_op(m1, m2, 4 /* POW */);
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

// src/Tensor.ts
var Tensor = class {
  get shape() {
    return this.data.shape;
  }
  constructor(data, _children = [], _op = null) {
    this.id = "P" + Math.floor(Math.random() * 1e6).toString().padStart(6, "0");
    if (data instanceof Matrix)
      this.data = data;
    else if (data instanceof Tensor)
      this.data = data.data.copy();
    else if (Array.isArray(data))
      this.data = new Matrix(data);
    else
      this.data = new Matrix([data]);
    this.grad = Matrix.zeros(this.data.shape);
    this._prev = new Set(_children);
    this._children = _children;
    this._op = _op;
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
    this.grad = Matrix.ones(this.data.shape);
    for (let v of topo.reverse()) {
      if (v._op === null)
        continue;
      const grads = v._op.backward(v.grad);
      if (grads) {
        for (let i = 0; i < grads.length; i++) {
          if (grads[i] !== null) {
            if (v._children[i].grad) {
              v._children[i].grad = v._children[i].grad.add(grads[i]);
            } else {
              v._children[i].grad = grads[i];
            }
          }
        }
      }
    }
  }
  add(other) {
    const otherTensor = other instanceof Tensor ? other : new Tensor(Matrix.full(this.data.shape, other));
    return new Operations_exports.Add().forward(this, otherTensor);
  }
  sub(other) {
    const otherTensor = other instanceof Tensor ? other : new Tensor(Matrix.full(this.data.shape, other));
    return this.add(otherTensor.mul(-1));
  }
  mul(other) {
    const otherTensor = other instanceof Tensor ? other : new Tensor(Matrix.full(this.data.shape, other));
    return new Operations_exports.Mul().forward(this, otherTensor);
  }
  div(other) {
    if (!(other instanceof Tensor))
      other = new Tensor(other);
    return this.mul(other.pow(-1));
  }
  pow(other) {
    const otherTensor = other instanceof Tensor ? other : new Tensor(Matrix.full(this.data.shape, other));
    return new Operations_exports.Pow().forward(this, otherTensor);
  }
  sqrt() {
    return this.pow(0.5);
  }
  rsqrt() {
    return this.pow(-0.5);
  }
  matmul(other) {
    const otherTensor = other instanceof Tensor ? other : new Tensor(Matrix.full(this.data.shape, other));
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
    return new Tensor(Matrix.ones(this.data.shape)).div(this);
  }
  sigmoid() {
    return new Tensor(Matrix.ones(this.data.shape)).add(this.mul(-1).exp()).reciprocal();
  }
  tanh() {
    const two1 = new Tensor(2);
    const two2 = new Tensor(2);
    return two1.mul(two2.mul(this).sigmoid()).sub(1);
  }
  permute(axes = null) {
    return new Operations_exports.Permute().forward(this, axes);
  }
  transpose(dim0, dim1) {
    return new Operations_exports.Tranpose().forward(this, dim0, dim1);
  }
  zero_grad() {
    this.grad = Matrix.zeros(this.data.shape);
  }
  __neg__() {
    return this.mul(new Tensor(this.data.mul(-1)));
  }
  __radd__(other) {
    return this.add(other);
  }
  __rsub__(other) {
    const o = other instanceof Tensor ? other : new Tensor(other);
    return o.add(this.mul(-1));
  }
  __rmul__(other) {
    return this.mul(other);
  }
  __rtruediv__(other) {
    return other.mul(this.pow(-1));
  }
  toString() {
    return `Tensor(data=${this.data}, grad=${this.grad})`;
  }
  assign(other) {
    this.data = other.data.copy();
    return this;
  }
  copy() {
    return new Tensor(this.data.copy());
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
};

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
    this.weight = new Tensor(Matrix.uniform(-1, 1, [out_features, in_features]));
    this.bias = new Tensor(Matrix.zeros([out_features]));
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
  // p = probability of an element to be zeroed.
  constructor(p = 0.5) {
    super();
    this.pScalar = p;
    this.p = new Tensor(p);
  }
  // def dropout(self, p=0.5) -> Tensor:
  //     if not Tensor.training: return self
  //     mask = (Tensor.rand(*self.shape, requires_grad=False) >= p).cast(dtypes.bool)
  //     return self * mask * (1/(1.0 - p))
  forward(x) {
    if (this.pScalar === 0)
      return x;
    const mask = new Tensor(Matrix.rand(x.shape).gte(this.p.data.reshape(x.shape)));
    return x.mul(mask).mul(new Tensor(1).div(new Tensor(1).sub(this.p)));
  }
  parameters() {
    return [];
  }
  toString() {
    return `Dropout(p=${this.p.data.data[0].toFixed(2)})`;
  }
};

// src/nn/LayerNorm.ts
var LayerNorm = class extends Module {
  constructor(normalized_shape, eps = 1e-5, elementwise_affine = true) {
    super();
    this.eps = eps;
    this.elementwise_affine = elementwise_affine;
    this.weight = new Tensor(Matrix.ones(normalized_shape));
    this.bias = elementwise_affine ? new Tensor(Matrix.zeros(normalized_shape)) : null;
  }
  numel(tensor) {
    return tensor.shape.reduce((acc, val) => acc * val, 1);
  }
  // def layernorm(self, axis=-1, eps:float=1e-5) -> Tensor:
  // y = (self - self.mean(axis, keepdim=True))
  // return y.mul((y*y).mean(axis, keepdim=True).add(eps).rsqrt())
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
    this.weight = new Tensor(Matrix.uniform(-1, 1, [num_embeddings, embedding_dim]));
    this.num_embeddings = num_embeddings;
    this.embedding_dim = embedding_dim;
  }
  getFirsts(v) {
    const data = v instanceof Tensor ? v.data.getData() : v.getData();
    return [data[0][0][0], data[0][0][1], data[0][0][2]];
  }
  forward(x) {
    const va = Matrix.arange(0, this.num_embeddings);
    const vb = va.reshape([1, 1, this.num_embeddings]);
    const vc = vb.expand([...x.shape, this.num_embeddings]);
    const vocab_counter = vc;
    const a = x.data.unsqueeze(2);
    const b = vocab_counter.equal(a);
    const c = b.expand([...x.shape, this.num_embeddings]);
    const d = c.dot(this.weight.data);
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
  Exp: () => Exp,
  Matmul: () => Matmul,
  Mul: () => Mul,
  Operation: () => Operation,
  Permute: () => Permute,
  Pow: () => Pow,
  Relu: () => Relu,
  Reshape: () => Reshape,
  Sum: () => Sum,
  Tranpose: () => Tranpose
});
var Operation = class {
  forward(...args) {
    throw Error("Not implemented");
  }
  backward(grad) {
    throw Error("Not implemented");
  }
};
function get_repeat_axis(left_shape, right_shape) {
  const len1 = left_shape.length;
  const len2 = right_shape.length;
  const left_not_repeat = len1 - len2;
  const repeat_axis = Matrix.arange(0, Math.abs(len1 - len2), 1);
  if (repeat_axis.data.length > 1)
    throw Error("Multiple axis repeated, sum wont work");
  return [left_not_repeat, repeat_axis.data[0]];
}
var Add = class extends Operation {
  forward(x, y) {
    this.x = x;
    this.y = y;
    return new Tensor(x.data.add(y.data), [x, y], this);
  }
  backward(grad) {
    const [self_not_repeat, self_repeat_axis] = get_repeat_axis(this.x.grad.shape, grad.shape);
    const [other_not_repeat, other_repeat_axis] = get_repeat_axis(this.y.grad.shape, grad.shape);
    return [
      self_not_repeat < 0 ? grad.sum(self_repeat_axis) : grad,
      other_not_repeat < 0 ? grad.sum(other_repeat_axis) : grad
    ];
  }
};
var Mul = class extends Operation {
  forward(x, y) {
    this.x = x;
    this.y = y;
    return new Tensor(x.data.mul(y.data), [x, y], this);
  }
  backward(grad) {
    return [this.y.data.mul(grad), this.x.data.mul(grad)];
  }
};
var Pow = class extends Operation {
  forward(x, y) {
    this.x = x;
    this.y = y;
    return new Tensor(x.data.pow(y.data), [x], this);
  }
  backward(grad) {
    const a = this.y.data.sub(1);
    const b = this.x.data.pow(a);
    const c = this.y.data.mul(b);
    const d = c.mul(grad);
    return [d, null];
  }
};
var Matmul = class extends Operation {
  forward(x, y) {
    this.x = x;
    this.y = y;
    return new Tensor(Matrix.dot(x.data, y.data), [x, y], this);
  }
  backward(grad) {
    return [Matrix.dot(grad, this.y.data.T), Matrix.dot(this.x.data.T, grad)];
  }
};
var Sum = class extends Operation {
  forward(x, axis = null, keepdims = false) {
    this.shape = x.shape.slice();
    return new Tensor(Matrix.sum(x.data, axis, keepdims), [x], this);
  }
  backward(grad) {
    return [grad.expand(this.shape), null];
  }
};
var Reshape = class extends Operation {
  forward(x, shape) {
    this.shape = x.shape;
    return new Tensor(Matrix.reshape(x.data, shape), [x], this);
  }
  backward(grad) {
    return [grad.reshape(this.shape), null];
  }
};
var Exp = class extends Operation {
  forward(x) {
    this.out = new Tensor(Matrix.exp(x.data), [x], this);
    return this.out;
  }
  backward(grad) {
    return [grad.mul(this.out.data), null];
  }
};
var Relu = class extends Operation {
  forward(x) {
    this.out = new Tensor(Matrix.maximum(Matrix.zeros(x.shape), x.data), [x], this);
    return this.out;
  }
  backward(grad) {
    return [Matrix.where(this.out.data.gt(0), grad, Matrix.zeros(this.out.data.shape)), null];
  }
};
var Permute = class extends Operation {
  forward(x, axes) {
    this.axes = axes;
    return new Tensor(Matrix.permute(x.data, axes), [x], this);
  }
  backward(grad) {
    return [Matrix.permute(grad, this.axes), null];
  }
};
var Tranpose = class extends Operation {
  forward(x, dim0, dim1) {
    this.dim0 = dim0;
    this.dim1 = dim1;
    return new Tensor(Matrix.transpose(x.data, dim0, dim1), [x], this);
  }
  backward(grad) {
    return [Matrix.transpose(grad, this.dim0, this.dim1), null];
  }
};
export {
  Matrix,
  Module,
  Operations_exports as Operations,
  Random,
  Tensor,
  nn_exports as nn
};
