import { describe, assert, equal } from "./TestUtils";
import { Matrix } from "../src/Matrix";

import { Random } from "../src/Random";

Random.SetRandomSeed(1337);

describe("Matrix creation", () => {
    const a = new Matrix([[1, 2], [3, 4]]); // Compute shape
    assert(equal(a, new Matrix([[1, 2], [3, 4]], [2, 2])));

    const b = new Matrix([1, 2, 3, 4], [2, 2]); // Pass shape
    assert(equal(b, new Matrix([[1, 2], [3, 4]], [2, 2])));

    const c = new Matrix([1, 2, 3, 4, 5, 6]); // 1D
    assert(equal(c, new Matrix([1, 2, 3, 4, 5, 6], [6])));

    const d = new Matrix([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]); // 3D
    assert(equal(d, new Matrix([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [2, 2, 2])));
})

describe("Reshape", () => {
    const a = new Matrix([0, 1, 2, 3, 4, 5]);

    const b = Matrix.reshape(a, [3, 2]);
    assert(equal(b, new Matrix([[0, 1], [2, 3], [4, 5]], [3, 2])));

    const c = Matrix.reshape(a, [2, 3]);
    assert(equal(c, new Matrix([[0, 1, 2], [3, 4, 5]], [2, 3])));

    const d = new Matrix([[1, 2, 3], [4, 5, 6]]);
    const e = Matrix.reshape(d, [6]);
    assert(equal(e, new Matrix([1, 2, 3, 4, 5, 6], [6])));

    const f = Matrix.reshape(d, [3, -1]);
    assert(equal(f, new Matrix([[1, 2], [3, 4], [5, 6]], [3, 2])));

    const g = Matrix.reshape(d, [-1, 3]);
    const h = Matrix.reshape(d, [3, -1]);
    assert(equal(g, new Matrix([[1, 2, 3], [4, 5, 6]], [2, 3])));
    assert(equal(h, new Matrix([[1, 2], [3, 4], [5, 6]], [3, 2])));

    const i = new Matrix([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const j = Matrix.reshape(i, [-1]);
    assert(equal(j, new Matrix([1, 2, 3, 4, 5, 6, 7, 8], [8])));

    const k = Matrix.reshape(i, -1);
    assert(equal(k, new Matrix([1, 2, 3, 4, 5, 6, 7, 8], [8])));
})

describe("Broadcasting", () => {
    const a = new Matrix([[1, 1], [2, 2], [3, 3], [4, 4]]);
    const b = new Matrix([0.1]);
    const c = new Matrix([0.1, 0.2]);

    const matrixVector = Matrix.broadcast(a, b);
    assert(equal(matrixVector[0], new Matrix([[1, 1], [2, 2], [3, 3], [4, 4]], [4, 2])));
    assert(equal(matrixVector[1], new Matrix([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]], [4, 2])));

    const vectorMatrix = Matrix.broadcast(b, a);
    assert(equal(vectorMatrix[0], new Matrix([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]], [4, 2])));
    assert(equal(vectorMatrix[1], new Matrix([[1, 1], [2, 2], [3, 3], [4, 4]], [4, 2])));

    const matrixMatrix = Matrix.broadcast(a, c);
    assert(equal(matrixMatrix[0], new Matrix([[1, 1], [2, 2], [3, 3], [4, 4]], [4, 2])));
    assert(equal(matrixMatrix[1], new Matrix([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]], [4, 2])));
})

describe("Binary Ops", () => {
    const a = new Matrix([[1, 1, 1], [2, 2, 2]]);
    const b = new Matrix([[3, 3, 3], [4, 4, 4]]);

    const c = Matrix.add(a, b);
    assert(equal(c, new Matrix([[4, 4, 4], [6, 6, 6]], [2, 3])));

    const d = Matrix.sub(a, b);
    assert(equal(d, new Matrix([[-2, -2, -2], [-2, -2, -2]], [2, 3])));

    const e = Matrix.mul(a, b);
    assert(equal(e, new Matrix([[3, 3, 3], [8, 8, 8]], [2, 3])));


    const f = new Matrix([[4, 4, 4], [2, 2, 2]]);
    const g = new Matrix([[2, 2, 2], [4, 4, 4]]);

    const h = Matrix.div(f, g);
    assert(equal(h, new Matrix([[2, 2, 2], [0.5, 0.5, 0.5]], [2, 3])));

    const i = Matrix.pow(f, g);
    assert(equal(i, new Matrix([[16, 16, 16], [16, 16, 16]], [2, 3])));
})

describe("Binary Ops scalars", () => {
    const a = new Matrix([[1, 1, 1], [2, 2, 2]]);
    const b = Matrix.add(a, 10);
    assert(equal(b, new Matrix([[11, 11, 11], [12, 12, 12]], [2, 3])));
})

describe("Test add with broadcasting", () => {
    const a = new Matrix([[1], [2], [3], [4]]);
    const b = new Matrix([0.1]);
    const c = Matrix.add(a, b);

    assert(equal(c, new Matrix([[1.1], [2.1], [3.1], [4.1]], [4, 1])));
})

describe("Matmul 1", () => {
    const a = new Matrix([[1, 2], [3, 4]]);
    const b = new Matrix([[5, 6], [7, 8]]);
    const c = Matrix.dot(a, b);

    assert(equal(c, new Matrix([[19, 22], [43, 50]], [2, 2])));
})

describe("Matmul 2", () => {
    const a = new Matrix([[1, 2], [3, 4], [5, 6]])
    const b = new Matrix([[7], [8]]);
    const c = Matrix.dot(a, b);

    assert(equal(c, new Matrix([[23], [53], [83]], [3, 1])));
})

describe("Matmul 3", () => {
    const a = new Matrix([-0.63, -0.46, 0.20], [3, 1]);
    const b = new Matrix([2], [1, 1]);
    const c = Matrix.dot(a, b);

    assert(equal(c, new Matrix([[-1.26], [-0.92], [0.4]], [3, 1])));
})

describe("Matmul 4", () => {
    const x = new Matrix([2, 3, -1], [1, 3]);
    const w = new Matrix([-0.63, -0.46, 0.20], [3, 1]);
    const d = Matrix.dot(x, w);

    assert(equal(d, new Matrix([[-2.8400000000000003]], [1, 1])));
})

describe("Matmul 5", () => {
    const x = new Matrix([[0.2, 0.3], [-0.4, 0.8], [-0.3, 0.9], [0.5, 0.3]]);
    const w = new Matrix([[-0.47595065], [-0.68263206]]);
    const xw = Matrix.dot(x, w);

    assert(equal(xw, new Matrix([[-0.299979748], [-0.3557253880000001], [-0.471583659], [-0.44276494299999997]], [4, 1])));
});

describe("Matmul 6", () => {
    const a = new Matrix([[-2.0260, -2.0655, -1.2054], [-0.9122, -1.2502, 0.8032]]);
    const b = new Matrix([[-0.2071, 0.0544], [0.1378, -0.3889], [0.5133, 0.3319]]);
    const r = Matrix.dot(a, b);

    assert(equal(r, new Matrix([[-0.4837731200000001, 0.2929862900000002], [0.42892162, 0.7031611799999999]], [2, 2])));
});

describe("Matmul 7", () => {
    const a = new Matrix([[1, 1], [1, 1]]);
    const b = new Matrix([[-0.2071, 0.1378, 0.5133], [0.0544, -0.3889, 0.3319]]);
    const r = Matrix.dot(a, b);

    assert(equal(r, new Matrix([[-0.1527, -0.2511, 0.8452], [-0.1527, -0.2511, 0.8452]], [2, 3])));
});


describe("Matmul 8", () => {
    const a = new Matrix([[1, 2], [3, 4]]);
    const b = new Matrix([[5, 6], [7, 8]]);

    assert(equal(Matrix.dot(a, b), new Matrix([[19, 22], [43, 50]], [2, 2])));
})

describe("Sum", () => {
    const a = new Matrix([0.5, 1.5]);
    const b = new Matrix([[1, 2], [3, 4]]);
    const c = new Matrix([[0, 1], [0, 5]]);

    const d = Matrix.sum(a);
    assert(equal(d, new Matrix([2], [1])));

    const e = Matrix.sum(b);
    assert(equal(e, new Matrix([10], [1])));

    const f = Matrix.sum(c, 0);
    assert(equal(f, new Matrix([0, 6], [2])));

    const g = Matrix.sum(c, 1);
    assert(equal(g, new Matrix([1, 5], [2])));

    const h = Matrix.sum(c, -2);
    assert(equal(h, new Matrix([0, 6], [2])));


    // Keepdims
    const i = new Matrix([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0]]);

    assert(equal(Matrix.sum(i, null, true), new Matrix([[6]])));

    assert(equal(Matrix.sum(i, 0, true), new Matrix([[2, 4, 0]])));
    assert(equal(Matrix.sum(i, 0, false), new Matrix([2, 4, 0])));

    assert(equal(Matrix.sum(i, 1, true), new Matrix([[0], [1], [2], [1], [2]])));
    assert(equal(Matrix.sum(i, 1, false), new Matrix([0, 1, 2, 1, 2])));


    const x = new Matrix([
        [
            [0, 1],
            [2, 3]
        ],
        [
            [4, 5],
            [6, 7]
        ]
    ])

    assert(equal(x.sum(), new Matrix([28])));
    assert(equal(x.sum(0), new Matrix([[4, 6], [8, 10]])));
    assert(equal(x.sum(1), new Matrix([[2, 4], [10, 12]])));
    assert(equal(x.sum(2), new Matrix([[1, 5], [9, 13]])));
    assert(equal(x.sum(-1), new Matrix([[1, 5], [9, 13]])));

    assert(equal(x.sum(null, true), new Matrix([[[28]]])));
    assert(equal(x.sum(0, true), new Matrix([[[4, 6], [8, 10]]])));
    assert(equal(x.sum(1, true), new Matrix([[[2, 4]], [[10, 12]]])));
    assert(equal(x.sum(2, true), new Matrix([[[1], [5]], [[9], [13]]])));
    assert(equal(x.sum(-1, true), new Matrix([[[1], [5]], [[9], [13]]])));

    const y = new Matrix([
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
    ])

    assert(equal(y.sum(), new Matrix([378])));
    assert(equal(y.sum(0), new Matrix([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[10, 11, 12], [13, 14, 15], [16, 17, 18]]], [[[19, 20, 21], [22, 23, 24], [25, 26, 27]]]])));
    assert(equal(y.sum(1), new Matrix([[[[30, 33, 36], [39, 42, 45], [48, 51, 54]]]])));
    assert(equal(y.sum(2), new Matrix([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]])));
    assert(equal(y.sum(3), new Matrix([[[[12, 15, 18]], [[39, 42, 45]], [[66, 69, 72]]]])));
    assert(equal(y.sum(4), new Matrix([[[[6, 15, 24]], [[33, 42, 51]], [[60, 69, 78]]]])));
    assert(equal(y.sum(-1), new Matrix([[[[6, 15, 24]], [[33, 42, 51]], [[60, 69, 78]]]])));
    assert(equal(y.sum(-2), new Matrix([[[[12, 15, 18]], [[39, 42, 45]], [[66, 69, 72]]]])));
})

describe("Mean", () => {
    const a = new Matrix([[1, 2], [3, 4]]);

    assert(equal(Matrix.mean(a), new Matrix([2.5])));
    assert(equal(Matrix.mean(a, null, true), new Matrix([[2.5]])));

    assert(equal(Matrix.mean(a, 0, true), new Matrix([[2, 3]])));
    assert(equal(Matrix.mean(a, 0, false), new Matrix([2, 3])));

    assert(equal(Matrix.mean(a, 1, true), new Matrix([[1.5], [3.5]])));
    assert(equal(Matrix.mean(a, 1, false), new Matrix([1.5, 3.5])));

    const b = new Matrix([[[0, 1, 2], [3, 4, 5]]]);
    assert(equal(Matrix.mean(b, 2, true), new Matrix([[[1], [4]]])));
    assert(equal(Matrix.mean(b, -1, true), new Matrix([[[1], [4]]])));
    assert(equal(Matrix.mean(b, -1, false), new Matrix([[1, 4]])));
});

describe("Var", () => {
    const a = new Matrix([[1, 2], [3, 4]]);

    assert(equal(Matrix.var(a), new Matrix([1.25])));
    assert(equal(Matrix.var(a, null, true), new Matrix([[1.25]])));

    assert(equal(Matrix.var(a, 0, true), new Matrix([[1, 1]])));
    assert(equal(Matrix.var(a, 0, false), new Matrix([1, 1])));

    assert(equal(Matrix.var(a, 1, true), new Matrix([[0.25], [0.25]])));
    assert(equal(Matrix.var(a, 1, false), new Matrix([0.25, 0.25])));
});

describe("Prod", () => {
    const a = new Matrix([0.5, 1.5]);
    const b = new Matrix([[1, 2], [3, 4]]);
    const c = new Matrix([[0, 1], [0, 5]]);

    const d = Matrix.prod(a);
    assert(equal(d, new Matrix([0.75], [1])));

    const e = Matrix.prod(b);
    assert(equal(e, new Matrix([24], [1])));

    const f = Matrix.prod(c);
    assert(equal(f, new Matrix([0], [1])));

    const g = Matrix.prod(c, 0);
    assert(equal(g, new Matrix([0, 5], [2])));

    const h = Matrix.prod(c, 1);
    assert(equal(h, new Matrix([0, 0], [2])));
})

describe("Zeros", () => {
    const a = Matrix.zeros([2, 2, 3]);
    assert(equal(a, new Matrix([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]], [2, 2, 3])));
})

describe("Ones", () => {
    const a = Matrix.ones([5, 1, 2]);
    assert(equal(a, new Matrix([[[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]]], [5, 1, 2])));
})

describe("arange", () => {
    const a = Matrix.arange(10, 20, 2);
    assert(equal(a, new Matrix([10, 12, 14, 16, 18], [5])));
})

describe("rand", () => {
    const a = Matrix.rand([2, 2]);
    assert(equal(a, new Matrix([[0.1844118325971067, 0.2681861550081521], [0.6026948785874993, 0.05738111538812518]], [2, 2])));
    
    const b = Matrix.rand([3,1,3]);
    assert(equal(b, new Matrix([[[0.4702075123786926,0.6373465061187744,0.3192155063152313]],[[0.7714118361473083,0.441847562789917,0.3168673813343048]],[[0.5497839450836182,0.5445157885551453,0.6433277726173401]]])));
})

describe("Greater than", () => {
    const a = new Matrix([1, 2, 3]);
    const b = new Matrix([0, 2, 2]);
    const c = a.gt(2);

    assert(equal(c, new Matrix([false, false, true], [3])));

    const d = a.gt(b);
    assert(equal(d, new Matrix([true, false, true], [3])));

    const e = a.gte(b);
    assert(equal(e, new Matrix([true, true, true], [3])));
})

describe("Less than", () => {
    const a = new Matrix([1, 2, 3]);
    const b = new Matrix([0, 2, 2]);
    const c = a.lt(2);

    assert(equal(c, new Matrix([true, false, false], [3])));

    const d = a.lt(b);
    assert(equal(d, new Matrix([false, false, false], [3])));

    const e = a.lte(b);
    assert(equal(e, new Matrix([false, true, false], [3])));

    const f = new Matrix([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]);
    const g = new Matrix([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]);

    const h = f.lte(g);
    assert(equal(h, new Matrix([[false, true, true], [false, false, true], [false, false, false], [false, false, false]], [4, 3])));
})

describe("Where using scalar", () => {
    const a = new Matrix([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const c = Matrix.where(a.lt(5), a, a.mul(10));

    assert(equal(c, new Matrix([1, 2, 3, 4, 50, 60, 70, 80, 90], [9])));
})

describe("Where using matrix", () => {
    const x = new Matrix([[1, 2], [3, 4]]);
    const y = new Matrix([[9, 8], [7, 6]]);

    const condition = new Matrix([[true, false], [true, true]]);

    const c = Matrix.where(condition, x, y);

    assert(equal(c, new Matrix([[1, 8], [3, 4]], [2, 2])));
})

describe("Where using permute", () => {
    const a = new Matrix([[1, 2], [3, 4]]);
    const b = new Matrix([[0, 0], [1, 1]]);
    
    const c = Matrix.where(b.equal(0), a, a.mul(10));
    assert(equal(c, new Matrix([[1,2],[30,40]])));

    const d = Matrix.where(b.equal(0), a.T, a.T.mul(10));
    assert(equal(d, new Matrix([[1,3],[20,40]])));
})

describe("Maximum", () => {
    const a = new Matrix([2, 3, 4]);
    const b = new Matrix([1, 5, 2]);
    const c = Matrix.maximum(a, b);

    assert(equal(c, new Matrix([2, 5, 4], [3])));
})

describe("Expand dims", () => {
    const a = new Matrix([1, 2]);
    const b = Matrix.expand_dims(a, 0);

    assert(equal(b, new Matrix([[1, 2]], [1, 2])));

    const c = Matrix.expand_dims(a, 1);
    assert(equal(c, new Matrix([[1], [2]], [2, 1])));

    const d = Matrix.expand_dims(a, [0, 1]);
    assert(equal(d, new Matrix([[[1, 2]]], [1, 1, 2])));

    const e = Matrix.expand_dims(a, [2, 0]);
    assert(equal(e, new Matrix([[[1], [2]]], [1, 2, 1])));

    const f = Matrix.expand_dims(a, []);
    assert(equal(f, new Matrix([1, 2], [2])));
})

describe("Expand", () => {
    const a = new Matrix([[1], [2], [3]]);

    assert(equal(a.expand([3,4]), new Matrix([[1,1,1,1],[2,2,2,2],[3,3,3,3]])));
    assert(equal(a.expand([-1,4]), new Matrix([[1,1,1,1],[2,2,2,2],[3,3,3,3]])));

    const b = new Matrix([[1, 2, 3], [4, 5, 6]]);
    assert(equal(b.expand([4,2,3]), new Matrix([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])));

    const c = new Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    assert(equal(c.expand([3,3,3]), new Matrix([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])));

    const d = new Matrix([1, 2, 3]);
    assert(equal(d.expand([3,3]), new Matrix([[1, 2, 3], [1, 2, 3], [1, 2, 3]])));

    const e = new Matrix([[[1], [2]], [[3], [4]]]);
    assert(equal(e.expand([2,2,3]), new Matrix([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])));
})

describe("Unsqueeze", () => {
    const a = new Matrix([1, 2, 3, 4]);

    const b = Matrix.unsqueeze(a, 0);
    assert(equal(b, new Matrix([[1, 2, 3, 4]], [1, 4])));

    const c = Matrix.unsqueeze(a, 1);
    assert(equal(c, new Matrix([[1], [2], [3], [4]], [4, 1])));
})

describe("Broadcast to", () => {
    const a = new Matrix([1, 2, 3]);

    // const b = Matrix.broadcast_to(a, [3, 3]);
    const b = a.broadcast_to([3,3]);
    assert(equal(b, new Matrix([[1, 2, 3], [1, 2, 3], [1, 2, 3]], [3, 3])));
})

describe("Less than equal", () => {
    const a = new Matrix([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]);
    const b = new Matrix([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]);

    const c = a.lte(b);
    assert(equal(c, new Matrix([[false, true, true], [false, false, true], [false, false, false], [false, false, false]], [4, 3])));
})

describe("Tril", () => {
    const a = new Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);

    assert(equal(Matrix.tril(a), new Matrix([[1, 0, 0], [4, 5, 0], [7, 8, 9], [10, 11, 12]], [4, 3])));
})

describe("Triu", () => {
    const a = new Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);

    const b = Matrix.triu(a);
    assert(equal(b, new Matrix([[1, 2, 3], [0, 5, 6], [0, 0, 9], [0, 0, 0]], [4, 3])));
})

describe("Abs", () => {
    const a = new Matrix([-1.5, 1.5]);
    const b = Matrix.abs(a);

    assert(equal(b, new Matrix([1.5, 1.5], [2])));
})

describe("Permute", () => {
    const a = new Matrix([[1, 2], [3, 4]]);

    assert(equal(a.permute(), new Matrix([[1, 3], [2, 4]])))

    const b = new Matrix([1, 2, 3, 4]);
    assert(equal(b.permute(), new Matrix([1, 2, 3, 4])))

    const c = new Matrix([[[1, 2, 3], [4, 5, 6]]]);
    assert(equal(c.permute([1, 0, 2]), new Matrix([[[1, 2, 3]], [[4, 5, 6]]])))

    const d = Matrix.ones([2, 3, 4, 5]);
    assert(equal(d.permute(), Matrix.ones([5, 4, 3, 2])))

    const e = new Matrix([[1,2], [3,4]]);
    const eT = Matrix.permute(e);


    assert(equal(e.permute(), new Matrix([[1,3], [2,4]])));
    assert(equal(e.permute().reshape([1,4]), new Matrix([[1,3,2,4]])));
});

describe("Transpose", () => {
    const a = new Matrix([[1.0028, -0.9893, 0.5809], [-0.1669, 0.7299, 0.4942]]);

    assert(equal(a.transpose(0, 1), new Matrix([[1.0028, -0.1669],[-0.9893, 0.7299],[ 0.5809, 0.4942]])));
});

describe("Masked fill", () => {
    const a = new Matrix([1, 2, 3, 4, 5]);
    const mask = new Matrix([1, 0, 0, 0, 0]);

    const filled = Matrix.masked_fill(a, mask, 10);

    assert(equal(filled, new Matrix([10, 2, 3, 4, 5])));
});

describe("Slice", () => {
    const a = new Matrix([
        [
            [ 0,  1,  2,  3],
            [ 4,  5,  6,  7]
        ],
        [
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]
        ],
        [
            [16, 17, 18, 19],
            [20, 21, 22, 23]
        ]
    ]);
        
    assert(equal(Matrix.slice(a, [null, null, 0]), new Matrix([[0, 4], [8, 12], [16, 20]])));
    assert(equal(Matrix.slice(a, [null, null, 1]), new Matrix([[1, 5], [9, 13], [17, 21]])));
    assert(equal(Matrix.slice(a, [null, null, 2]), new Matrix([[2, 6], [10, 14], [18, 22]])));
    assert(equal(Matrix.slice(a, [null, null, [1,2]]), new Matrix([[[1], [5]], [[9], [13]], [[17], [21]]])));
    assert(equal(Matrix.slice(a, [null, [1,2], [1,2]]), new Matrix([[[5]], [[13]], [[21]]])));
});

describe("Cat", () => {
    const a = new Matrix([[1, 2], [3, 4]]);
    const b = new Matrix([[5, 6]]);

    console.log(`a ${Matrix.cat([a, b])}`)
    assert(equal(Matrix.cat([a, b]), new Matrix([[1, 2], [3, 4], [5, 6]])));
    // assert(equal(Matrix.cat([a, b], null), new Matrix([1, 2, 3, 4, 5, 6])));
    // assert(equal(Matrix.cat([a, b.permute()], 1), new Matrix([[1, 2, 5], [3, 4, 6]])));
});

describe("Softmax", () => {
    const a = Matrix.arange(0, 10);
    assert(equal(Matrix.softmax(a, 0), new Matrix([0.0001,0.0002,0.0006,0.0016,0.0043,0.0116,0.0315,0.0856,0.2326,0.6321]), 1e-4));

    const b = new Matrix([[5, 6, 3]]);
    assert(equal(Matrix.softmax(b, -1), new Matrix([[0.2595, 0.7054, 0.0351]]), 1e-4));
});

describe("Multinomial", () => {
    function computeProbs(events: Float32Array, numOutcomes: number): Matrix {
        const counts: number[] = [];
        for (let i = 0; i < numOutcomes; ++i) {
            counts[i] = 0;
        }
        const numSamples = events.length;
        for (let i = 0; i < events.length; ++i) {
            counts[events[i]]++;
        }
        // Normalize counts to be probabilities between [0, 1].
        for (let i = 0; i < counts.length; i++) {
            counts[i] /= numSamples;
        }
        return new Matrix(counts);
    }

    const NUM_SAMPLES = 1000;
    const EPS = 0.05;

    const fairCoin = new Matrix([1, 1]);
    assert(equal(computeProbs(Matrix.multinomial(fairCoin, NUM_SAMPLES).data, 2), new Matrix([0.5, 0.5]), EPS));

    const headsCoin = new Matrix([1, -100]);
    assert(equal(computeProbs(Matrix.multinomial(headsCoin, NUM_SAMPLES).data, 2), new Matrix([1, 0]), EPS));

    const tailsCoin = new Matrix([-100, 1]);
    assert(equal(computeProbs(Matrix.multinomial(tailsCoin, NUM_SAMPLES).data, 2), new Matrix([0, 1]), EPS));

    const tenSidedCoin = Matrix.full([10], 1);
    const tenSidedCoinProbs = computeProbs(Matrix.multinomial(tenSidedCoin, NUM_SAMPLES).data, 10);
    assert(tenSidedCoinProbs.data.length <= 10);
    assert(equal(tenSidedCoinProbs.sum(), new Matrix([1])));

    const threeSidedBiasedCoin = new Matrix([[-100, -100, 1], [-100, 1, -100], [1, -100, -100]]);
    const threeSidedBiasedCoinMultinomial = Matrix.multinomial(threeSidedBiasedCoin, NUM_SAMPLES)

    let outcomeProbs = computeProbs(threeSidedBiasedCoinMultinomial.data.slice(0, NUM_SAMPLES), 3);
    assert(equal(outcomeProbs, new Matrix([0, 0, 1])));

    outcomeProbs = computeProbs(threeSidedBiasedCoinMultinomial.data.slice(NUM_SAMPLES, 2 * NUM_SAMPLES), 3);
    assert(equal(outcomeProbs, new Matrix([0, 1, 0])));

    outcomeProbs = computeProbs(threeSidedBiasedCoinMultinomial.data.slice(2 * NUM_SAMPLES), 3);
    assert(equal(outcomeProbs, new Matrix([1, 0, 0])));
});

// describe("Split", () => {
//     const a = Matrix.arange(0, 10).reshape([5, 2]);

//     const b = Matrix.split(a, 2);
//     assert(b.length == 3);
//     assert(equal(b[0], new Matrix([[0, 1], [2, 3]])));
//     assert(equal(b[1], new Matrix([[4, 5], [6, 7]])));
//     assert(equal(b[2], new Matrix([[8, 9]])));

//     const c = Matrix.split(a, [1, 4]);
//     assert(c.length == 2);
//     assert(equal(c[0], new Matrix([[0, 1]])));
//     assert(equal(c[1], new Matrix([[2, 3], [4, 5], [6, 7], [8, 9]])));
// })