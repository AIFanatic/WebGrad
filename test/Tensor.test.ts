import { describe, assert, equal } from "./TestUtils";
import { Tensor } from "../src/Tensor";

import { Random } from "../src/Random";

Random.SetRandomSeed(1337);

describe("Tensor creation", () => {
    const a = new Tensor([[1, 2], [3, 4]]); // Compute shape
    assert(equal(a, new Tensor([[1, 2], [3, 4]])));
    assert(equal(a.shape, [2,2]));

    const b = new Tensor([1, 2, 3, 4, 5, 6]); // 1D
    assert(equal(b, new Tensor([1, 2, 3, 4, 5, 6])));
    assert(equal(b.shape, [6]));

    const c = new Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]); // 3D
    assert(equal(c, new Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])));
    assert(equal(c.shape, [2, 2, 2]));
})

describe("Zeros", () => {
    const a = Tensor.zeros([2, 2, 3]);
    assert(equal(a, new Tensor([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])));
    assert(equal(a.shape, [2, 2, 3]));
})

describe("Ones", () => {
    const a = Tensor.ones([5, 1, 2]);
    assert(equal(a, new Tensor([[[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]]])));
    assert(equal(a.shape, [5, 1, 2]));
})

describe("arange", () => {
    const a = Tensor.arange(10, 20, 2);
    assert(equal(a, new Tensor([10, 12, 14, 16, 18])));
    assert(equal(a.shape, [5]));
})

describe("rand", () => {
    const a = Tensor.rand([2, 2]);
    assert(equal(a, new Tensor([[0.1844118325971067, 0.2681861550081521], [0.6026948785874993, 0.05738111538812518]])));
    assert(equal(a.shape, [2, 2]));
    
    const b = Tensor.rand([3,1,3]);
    assert(equal(b, new Tensor([[[0.4702075123786926,0.6373465061187744,0.3192155063152313]],[[0.7714118361473083,0.441847562789917,0.3168673813343048]],[[0.5497839450836182,0.5445157885551453,0.6433277726173401]]])));
})

describe("Reshape", () => {
    const a = new Tensor([0, 1, 2, 3, 4, 5]);

    const b = a.reshape([3, 2]);
    assert(equal(b, new Tensor([[0, 1], [2, 3], [4, 5]])));
    assert(equal(b.shape, [3, 2]));

    const c = a.reshape([2, 3]);
    assert(equal(c, new Tensor([[0, 1, 2], [3, 4, 5]])));
    assert(equal(c.shape, [2, 3]));

    const d = new Tensor([[1, 2, 3], [4, 5, 6]]);
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

    const i = new Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const j = i.reshape([-1]);
    assert(equal(j, new Tensor([1, 2, 3, 4, 5, 6, 7, 8])));
    assert(equal(j.shape, [8]));

    const k = i.reshape(-1);
    assert(equal(k, new Tensor([1, 2, 3, 4, 5, 6, 7, 8])));
    assert(equal(k.shape, [8]));
})

describe("Broadcasting", () => {
    const a = new Tensor([[1, 1], [2, 2], [3, 3], [4, 4]]);
    const b = new Tensor([0.1]);
    const c = new Tensor([0.1, 0.2]);

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
})

describe("Binary Ops", () => {
    const a = new Tensor([[1, 1, 1], [2, 2, 2]]);
    const b = new Tensor([[3, 3, 3], [4, 4, 4]]);

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
})

describe("Binary Ops scalars", () => {
    const a = new Tensor([[1, 1, 1], [2, 2, 2]]);
    const b = a.add(10);

    assert(equal(b, new Tensor([[11, 11, 11], [12, 12, 12]])));
    assert(equal(b.shape, [2, 3]));
})

describe("Test add with broadcasting", () => {
    const a = new Tensor([[1], [2], [3], [4]]);
    const b = new Tensor([0.1]);
    const c = new Tensor([0.1, 0.2]);
    const d = a.add(b);

    assert(equal(d, new Tensor([[1.1], [2.1], [3.1], [4.1]])));
    assert(equal(d.shape, [4, 1]));

    const e = a.add(c);

    assert(equal(e, new Tensor([[1.1, 1.2], [2.1, 2.2], [3.1, 3.2], [4.1, 4.2]])));
    assert(equal(e.shape, [4, 2]));
})

describe("Matmul 1", () => {
    const a = new Tensor([[1, 2], [3, 4]]);
    const b = new Tensor([[5, 6], [7, 8]]);
    const c = a.matmul(b);

    assert(equal(c, new Tensor([[19, 22], [43, 50]])));
    assert(equal(c.shape, [2, 2]));
})

describe("Matmul 2", () => {
    const a = new Tensor([[1, 2], [3, 4], [5, 6]])
    const b = new Tensor([[7], [8]]);
    const c = a.matmul(b);

    assert(equal(c, new Tensor([[23], [53], [83]])));
    assert(equal(c.shape, [3, 1]));
})

describe("Matmul 3", () => {
    const a = new Tensor([-0.63, -0.46, 0.20]).reshape([3,1]);
    const b = new Tensor([2]).reshape([1,1]);
    const c = a.matmul(b);

    assert(equal(c, new Tensor([[-1.26], [-0.92], [0.4]])));
    assert(equal(c.shape, [3, 1]));
})

describe("Matmul 4", () => {
    const x = new Tensor([2, 3, -1]).reshape([1,3]);
    const w = new Tensor([-0.63, -0.46, 0.20]).reshape([3,1]);
    const d = x.matmul(w);

    assert(equal(d, new Tensor([[-2.8400000000000003]])));
    assert(equal(d.shape, [1, 1]));
})

describe("Matmul 5", () => {
    const x = new Tensor([[0.2, 0.3], [-0.4, 0.8], [-0.3, 0.9], [0.5, 0.3]]);
    const w = new Tensor([[-0.47595065], [-0.68263206]]);
    const xw = x.matmul(w);

    assert(equal(xw, new Tensor([[-0.299979748], [-0.3557253880000001], [-0.471583659], [-0.44276494299999997]])));
    assert(equal(xw.shape, [4, 1]));
});

describe("Matmul 6", () => {
    const a = new Tensor([[-2.0260, -2.0655, -1.2054], [-0.9122, -1.2502, 0.8032]]);
    const b = new Tensor([[-0.2071, 0.0544], [0.1378, -0.3889], [0.5133, 0.3319]]);
    const r = a.matmul(b);

    assert(equal(r, new Tensor([[-0.4837731200000001, 0.2929862900000002], [0.42892162, 0.7031611799999999]])));
    assert(equal(r.shape, [2, 2]));
});

describe("Matmul 7", () => {
    const a = new Tensor([[1, 1], [1, 1]]);
    const b = new Tensor([[-0.2071, 0.1378, 0.5133], [0.0544, -0.3889, 0.3319]]);
    const r = a.matmul(b);

    assert(equal(r, new Tensor([[-0.1527, -0.2511, 0.8452], [-0.1527, -0.2511, 0.8452]])));
    assert(equal(r.shape, [2, 3]));
});

describe("Matmul with permute", () => {
    const x = new Tensor([[1, 2], [3, 4]]);
    const w = new Tensor([[-0.5963, -0.0062],[ 0.1741, -0.1097],[-0.4237, -0.6666],[ 0.1204, 0.2781]]);
    const wP = w.permute([-1, -2]);
    const y = x.matmul(wP);

    assert(equal(y, new Tensor([[-0.6087, -0.0453, -1.7569,  0.6766], [-1.8137,  0.0835, -3.9375,  1.4736]])));
    assert(equal(y.shape, [2, 4]));
});

describe("Sum", () => {
    const a = new Tensor([0.5, 1.5]);
    const b = new Tensor([[1, 2], [3, 4]]);
    const c = new Tensor([[0, 1], [0, 5]]);

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


    // Keepdims
    const i = new Tensor([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0]]);

    assert(equal(i.sum(null, true), new Tensor([[6]])));

    console.log(`t ${i.sum(0, true)}`);
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
    ])

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
    ])

    assert(equal(y.sum(), new Tensor([378])));
    assert(equal(y.sum(0), new Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[10, 11, 12], [13, 14, 15], [16, 17, 18]]], [[[19, 20, 21], [22, 23, 24], [25, 26, 27]]]])));
    assert(equal(y.sum(1), new Tensor([[[[30, 33, 36], [39, 42, 45], [48, 51, 54]]]])));
    assert(equal(y.sum(2), new Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]])));
    assert(equal(y.sum(3), new Tensor([[[[12, 15, 18]], [[39, 42, 45]], [[66, 69, 72]]]])));
    assert(equal(y.sum(4), new Tensor([[[[6, 15, 24]], [[33, 42, 51]], [[60, 69, 78]]]])));
    assert(equal(y.sum(-1), new Tensor([[[[6, 15, 24]], [[33, 42, 51]], [[60, 69, 78]]]])));
    assert(equal(y.sum(-2), new Tensor([[[[12, 15, 18]], [[39, 42, 45]], [[66, 69, 72]]]])));
})

describe("Mean", () => {
    const a = new Tensor([[1, 2], [3, 4]]);

    assert(equal(a.mean(), new Tensor([2.5])));
    assert(equal(a.mean(null, true), new Tensor([[2.5]])));

    assert(equal(a.mean(0, true), new Tensor([[2, 3]])));
    assert(equal(a.mean(0, false), new Tensor([2, 3])));

    assert(equal(a.mean(1, true), new Tensor([[1.5], [3.5]])));
    assert(equal(a.mean(1, false), new Tensor([1.5, 3.5])));

    const b = new Tensor([[[0, 1, 2], [3, 4, 5]]]);
    assert(equal(b.mean(2, true), new Tensor([[[1], [4]]])));
    assert(equal(b.mean(-1, true), new Tensor([[[1], [4]]])));
    assert(equal(b.mean(-1, false), new Tensor([[1, 4]])));
});

describe("Var", () => {
    const a = new Tensor([[1, 2], [3, 4]]);

    assert(equal(a.var(), new Tensor([1.25])));
    assert(equal(a.var(null, true), new Tensor([[1.25]])));

    assert(equal(a.var(0, true), new Tensor([[1, 1]])));
    assert(equal(a.var(0, false), new Tensor([1, 1])));

    assert(equal(a.var(1, true), new Tensor([[0.25], [0.25]])));
    assert(equal(a.var(1, false), new Tensor([0.25, 0.25])));
});

describe("Abs", () => {
    const a = new Tensor([[-3, 3, 3], [4, -4, 4]]);
    assert(equal(a.abs(), new Tensor([[3, 3, 3], [4, 4, 4]])));
})

describe("Log", () => {
    const a = new Tensor([[1, 2, 3], [4, 5, 6]]);
    assert(equal(a.log(), new Tensor([[0,0.6931,1.0986],[1.3863,1.6094,1.7918]]), 1e-4));
})

describe("Equal", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor([0, 2, 2]);
    const c = a.eq(b);

    assert(equal(c, new Tensor([0, 1, 0])));

    const d = new Tensor([3]);
    const e = a.eq(d);
    assert(equal(e, new Tensor([0, 0, 1])));
})

describe("Not equal", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor([0, 2, 2]);
    const c = a.ne(b);

    assert(equal(c, new Tensor([1, 0, 1])));

    const d = new Tensor([3]);
    const e = a.ne(d);
    assert(equal(e, new Tensor([1, 1, 0])));
})

describe("Greater than", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor([0, 2, 2]);
    const c = a.gt(2);

    assert(equal(c, new Tensor([0, 0, 1])));

    const d = a.gt(b);
    assert(equal(d, new Tensor([1, 0, 1])));

    const e = a.gte(b);
    assert(equal(e, new Tensor([1, 1, 1])));
})

describe("Less than", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor([0, 2, 2]);
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
})

describe("Less than equal", () => {
    const a = new Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]);
    const b = new Tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]);

    const c = a.lte(b);
    assert(equal(c, new Tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]])));
})

describe("Where using scalar", () => {
    const a = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const c = Tensor.where(a.lt(5), a, a.mul(10));

    assert(equal(c, new Tensor([1, 2, 3, 4, 50, 60, 70, 80, 90])));
})

describe("Where using matrix", () => {
    const x = new Tensor([[1, 2], [3, 4]]);
    const y = new Tensor([[9, 8], [7, 6]]);

    const condition = new Tensor([[1, 0], [1, 1]]);

    const c = Tensor.where(condition, x, y);

    assert(equal(c, new Tensor([[1, 8], [3, 4]])));
})

describe("Where using permute", () => {
    const a = new Tensor([[1, 2], [3, 4]]);
    const b = new Tensor([[0, 0], [1, 1]]);
    
    const c = Tensor.where(b.eq(0), a, a.mul(10));
    assert(equal(c, new Tensor([[1,2],[30,40]])));

    const d = Tensor.where(b.eq(0), a.T, a.T.mul(10));
    assert(equal(d, new Tensor([[1,3],[20,40]])));
})

describe("Maximum", () => {
    const a = new Tensor([2, 3, 4]);
    const b = new Tensor([1, 5, 2]);
    const c = a.maximum(b);

    assert(equal(c, new Tensor([2, 5, 4])));
    assert(equal(c.shape, [3]));
})

describe("Expand", () => {
    const a = new Tensor([[1], [2], [3]]);

    assert(equal(a.expand([3,4]), new Tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]])));
    assert(equal(a.expand([-1,4]), new Tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]])));

    const b = new Tensor([[1, 2, 3], [4, 5, 6]]);
    assert(equal(b.expand([4,2,3]), new Tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])));

    const c = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    assert(equal(c.expand([3,3,3]), new Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])));

    const d = new Tensor([1, 2, 3]);
    assert(equal(d.expand([3,3]), new Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])));

    const e = new Tensor([[[1], [2]], [[3], [4]]]);
    assert(equal(e.expand([2,2,3]), new Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])));
})

describe("Unsqueeze", () => {
    const a = new Tensor([1, 2, 3, 4]);

    const b = a.unsqueeze(0);
    assert(equal(b, new Tensor([[1, 2, 3, 4]])));
    assert(equal(b.shape, [1, 4]));

    const c = a.unsqueeze(1);
    assert(equal(c, new Tensor([[1], [2], [3], [4]])));
    assert(equal(c.shape, [4, 1]));
})

describe("Masked fill", () => {
    const a = new Tensor([1, 2, 3, 4, 5]);
    const mask = new Tensor([1, 0, 0, 0, 0]);

    const filled = a.masked_fill(mask, 10);

    assert(equal(filled, new Tensor([10, 2, 3, 4, 5])));
});

describe("Permute", () => {
    const a = new Tensor([[1, 2], [3, 4]]);

    assert(equal(a.permute(), new Tensor([[1, 3], [2, 4]])))

    const b = new Tensor([1, 2, 3, 4]);
    assert(equal(b.permute(), new Tensor([1, 2, 3, 4])))

    const c = new Tensor([[[1, 2, 3], [4, 5, 6]]]);
    assert(equal(c.permute([1, 0, 2]), new Tensor([[[1, 2, 3]], [[4, 5, 6]]])))

    const d = Tensor.ones([2, 3, 4, 5]);
    assert(equal(d.permute(), Tensor.ones([5, 4, 3, 2])))

    const e = new Tensor([[1,2], [3,4]]);

    assert(equal(e.permute(), new Tensor([[1,3], [2,4]])));
    assert(equal(e.permute().reshape([1,4]), new Tensor([[1,3,2,4]])));
});

describe("Transpose", () => {
    const a = new Tensor([[1.0028, -0.9893, 0.5809], [-0.1669, 0.7299, 0.4942]]);

    assert(equal(a.transpose(0, 1), new Tensor([[1.0028, -0.1669],[-0.9893, 0.7299],[ 0.5809, 0.4942]])));
});

describe("Softmax", () => {
    const a = Tensor.arange(0, 10);
    assert(equal(a.softmax(0), new Tensor([0.0001,0.0002,0.0006,0.0016,0.0043,0.0116,0.0315,0.0856,0.2326,0.6321]), 1e-4));

    const b = new Tensor([[5, 6, 3]]);
    assert(equal(b.softmax(-1), new Tensor([[0.2595, 0.7054, 0.0351]]), 1e-4));
});

describe("Slice", () => {
    const a = new Tensor([
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
        
    assert(equal(a.slice([null, null, 0]), new Tensor([[0, 4], [8, 12], [16, 20]])));
    assert(equal(a.slice([null, null, 1]), new Tensor([[1, 5], [9, 13], [17, 21]])));
    assert(equal(a.slice([null, null, 2]), new Tensor([[2, 6], [10, 14], [18, 22]])));
    assert(equal(a.slice([null, null, [1,2]]), new Tensor([[[1], [5]], [[9], [13]], [[17], [21]]])));
    assert(equal(a.slice([null, [1,2], [1,2]]), new Tensor([[[5]], [[13]], [[21]]])));
});

describe("Contiguous", () => {
    const a = new Tensor([
        [
            [0, 1],
            [2, 3]
        ],
        [
            [4, 5],
            [6, 7]
        ]
    ]);

    const b = a.T;
    assert(equal(b, new Tensor([[[0,4], [2,6]], [[1,5], [3,7]]])));
    assert(equal(b.shape, [2,2,2]));
    assert(equal(b.strides, [1,2,4]));

    const c = b.contiguous()
    assert(equal(c, new Tensor([[[0,4], [2,6]], [[1,5], [3,7]]])));
    assert(equal(c.shape, [2,2,2]));
    assert(equal(c.strides, [4,2,1]));


    const d = new Tensor([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
    ]);

    const e = d.T;
    const f = e.reshape([2,4]);
    assert(equal(f, new Tensor([[0,2,4,6], [1,3,5,7]])));
    assert(equal(f.shape, [2,4]));
    assert(equal(f.strides, [4,1]));

    const g = a.reshape([2,4]);
    assert(equal(g, new Tensor([[0,1,2,3], [4,5,6,7]])));
    assert(equal(g.shape, [2,4]));
    assert(equal(g.strides, [4,1]));
})

describe("Tril", () => {
    const a = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);

    const b = a.tril();
    assert(equal(b, new Tensor([[1, 0, 0], [4, 5, 0], [7, 8, 9], [10, 11, 12]])));
    assert(equal(b.shape, [4, 3]));
})

describe("Triu", () => {
    const a = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
    
    const b = a.triu();
    assert(equal(b, new Tensor([[1, 2, 3], [0, 5, 6], [0, 0, 9], [0, 0, 0]])));
    assert(equal(b.shape, [4, 3]));
})