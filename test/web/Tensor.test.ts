import { describe, assert, equal } from "../TestUtils";
import { Tensor } from "../../src/Tensor";

import { Random } from "../../src/Random";
import { Device } from "../../src";
import { TestRunner } from "../run-web";

Random.SetRandomSeed(1337);

function TensorTest(device: Device) {
    TestRunner.describe("Tensor creation", () => {
        const a = new Tensor([[1, 2], [3, 4]], {device: device});
        assert(equal(a, new Tensor([[1, 2], [3, 4]])));
        assert(equal(a.shape, [2,2]));
    
        const b = new Tensor([1, 2, 3, 4, 5, 6], {device: device});
        assert(equal(b, new Tensor([1, 2, 3, 4, 5, 6])));
        assert(equal(b.shape, [6]));
    
        const c = new Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], {device: device});
        assert(equal(c, new Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])));
        assert(equal(c.shape, [2, 2, 2]));
    })

    TestRunner.describe("Tensor to", () => {
        const a = new Tensor([[1, 2], [3, 4]], {device: device});
        assert(a.device === device);

        // Only works because only two devices
        const otherDevice = device === Device.CPU ? Device.WEBGL : Device.CPU;
        const b = a.to(otherDevice);
        assert(b.device === otherDevice);
    })
    
    TestRunner.describe("Zeros", () => {
        const a = Tensor.zeros([2, 2, 3], {device: device});
        assert(equal(a, new Tensor([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])));
        assert(equal(a.shape, [2, 2, 3]));
    })
    
    TestRunner.describe("Ones", () => {
        const a = Tensor.ones([5, 1, 2], {device: device});
        assert(equal(a, new Tensor([[[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]]])));
        assert(equal(a.shape, [5, 1, 2]));
    })
    
    TestRunner.describe("arange", () => {
        const a = Tensor.arange(10, 20, 2, {device: device});
        assert(equal(a, new Tensor([10, 12, 14, 16, 18])));
        assert(equal(a.shape, [5]));
    })
    
    TestRunner.describe("rand", () => {
        Random.SetRandomSeed(1337);
        const a = Tensor.rand([2, 2], {device: device});
        assert(equal(a, new Tensor([[0.1844118325971067, 0.2681861550081521], [0.6026948785874993, 0.05738111538812518]])));
        assert(equal(a.shape, [2, 2]));
        
        const b = Tensor.rand([3,1,3], {device: device});
        assert(equal(b, new Tensor([[[0.4702075123786926,0.6373465061187744,0.3192155063152313]],[[0.7714118361473083,0.441847562789917,0.3168673813343048]],[[0.5497839450836182,0.5445157885551453,0.6433277726173401]]])));
    })
    
    TestRunner.describe("Reshape", () => {
        const a = new Tensor([0, 1, 2, 3, 4, 5], {device: device});
    
        const b = a.reshape([3, 2]);
        assert(equal(b, new Tensor([[0, 1], [2, 3], [4, 5]])));
        assert(equal(b.shape, [3, 2]));
    
        const c = a.reshape([2, 3]);
        assert(equal(c, new Tensor([[0, 1, 2], [3, 4, 5]])));
        assert(equal(c.shape, [2, 3]));
    
        const d = new Tensor([[1, 2, 3], [4, 5, 6]], {device: device});
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
    
        const i = new Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], {device: device});
        const j = i.reshape([-1]);
        assert(equal(j, new Tensor([1, 2, 3, 4, 5, 6, 7, 8])));
        assert(equal(j.shape, [8]));
    
        const k = i.reshape(-1);
        assert(equal(k, new Tensor([1, 2, 3, 4, 5, 6, 7, 8])));
        assert(equal(k.shape, [8]));
    })
    
    TestRunner.describe("Broadcasting", () => {
        const a = new Tensor([[1, 1], [2, 2], [3, 3], [4, 4]], {device: device});
        const b = new Tensor([0.1], {device: device});
        const c = new Tensor([0.1, 0.2], {device: device});
    
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
    
    TestRunner.describe("Binary Ops", () => {
        const a = new Tensor([[1, 1, 1], [2, 2, 2]], {device: device});
        const b = new Tensor([[3, 3, 3], [4, 4, 4]], {device: device});
    
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
    
    TestRunner.describe("Negative pow", () => {
        const a = new Tensor([-2, 2, -2], {device: device});
        const b = new Tensor([2, 2, 2], {device: device});
        const c = a.pow(b);
        assert(equal(c, new Tensor([4, 4, 4])));

        const d = new Tensor([3, 3, 3], {device: device});
        const e = a.pow(d);
        assert(equal(e, new Tensor([-8, 8, -8])));

        // // TODO: Fix
        // const f = new Tensor([0.1, 1, 1], {device: device});
        // const g = a.pow(f);
        // assert(equal(g, new Tensor([0, 2, -2])));
    })

    TestRunner.describe("Binary Ops scalars", () => {
        const a = new Tensor([[1, 1, 1], [2, 2, 2]], {device: device});
        const b = a.add(10);
    
        assert(equal(b, new Tensor([[11, 11, 11], [12, 12, 12]])));
        assert(equal(b.shape, [2, 3]));
    })
    
    TestRunner.describe("Test add with broadcasting", () => {
        const a = new Tensor([[1], [2], [3], [4]], {device: device});
        const b = new Tensor([0.1], {device: device});
        const c = new Tensor([0.1, 0.2], {device: device});
        const d = a.add(b);
    
        assert(equal(d, new Tensor([[1.1], [2.1], [3.1], [4.1]])));
        assert(equal(d.shape, [4, 1]));
    
        const e = a.add(c);
    
        assert(equal(e, new Tensor([[1.1, 1.2], [2.1, 2.2], [3.1, 3.2], [4.1, 4.2]])));
        assert(equal(e.shape, [4, 2]));
    })
    
    TestRunner.describe("Matmul 1", () => {
        const a = new Tensor([[1, 2], [3, 4]], {device: device});
        const b = new Tensor([[5, 6], [7, 8]], {device: device});
        const c = a.matmul(b);
    
        assert(equal(c, new Tensor([[19, 22], [43, 50]])));
        assert(equal(c.shape, [2, 2]));
    })
    
    TestRunner.describe("Matmul 2", () => {
        const a = new Tensor([[1, 2], [3, 4], [5, 6]], {device: device})
        const b = new Tensor([[7], [8]], {device: device});
        const c = a.matmul(b);
    
        assert(equal(c, new Tensor([[23], [53], [83]])));
        assert(equal(c.shape, [3, 1]));
    })
    
    TestRunner.describe("Matmul 3", () => {
        const a = new Tensor([-0.63, -0.46, 0.20], {device: device}).reshape([3,1]);
        const b = new Tensor([2], {device: device}).reshape([1,1]);
        const c = a.matmul(b);
    
        assert(equal(c, new Tensor([[-1.26], [-0.92], [0.4]])));
        assert(equal(c.shape, [3, 1]));
    })
    
    TestRunner.describe("Matmul 4", () => {
        const x = new Tensor([2, 3, -1], {device: device}).reshape([1,3]);
        const w = new Tensor([-0.63, -0.46, 0.20], {device: device}).reshape([3,1]);
        const d = x.matmul(w);
    
        assert(equal(d, new Tensor([[-2.8400000000000003]])));
        assert(equal(d.shape, [1, 1]));
    })
    
    TestRunner.describe("Matmul 5", () => {
        const x = new Tensor([[0.2, 0.3], [-0.4, 0.8], [-0.3, 0.9], [0.5, 0.3]], {device: device});
        const w = new Tensor([[-0.47595065], [-0.68263206]], {device: device});
        const xw = x.matmul(w);
    
        assert(equal(xw, new Tensor([[-0.299979748], [-0.3557253880000001], [-0.471583659], [-0.44276494299999997]])));
        assert(equal(xw.shape, [4, 1]));
    });
    
    TestRunner.describe("Matmul 6", () => {
        const a = new Tensor([[-2.0260, -2.0655, -1.2054], [-0.9122, -1.2502, 0.8032]], {device: device});
        const b = new Tensor([[-0.2071, 0.0544], [0.1378, -0.3889], [0.5133, 0.3319]], {device: device});
        const r = a.matmul(b);
    
        assert(equal(r, new Tensor([[-0.4837731200000001, 0.2929862900000002], [0.42892162, 0.7031611799999999]])));
        assert(equal(r.shape, [2, 2]));
    });
    
    TestRunner.describe("Matmul 7", () => {
        const a = new Tensor([[1, 1], [1, 1]], {device: device});
        const b = new Tensor([[-0.2071, 0.1378, 0.5133], [0.0544, -0.3889, 0.3319]], {device: device});
        const r = a.matmul(b);
    
        assert(equal(r, new Tensor([[-0.1527, -0.2511, 0.8452], [-0.1527, -0.2511, 0.8452]])));
        assert(equal(r.shape, [2, 3]));
    });
    
    TestRunner.describe("Matmul with permute", () => {
        const x = new Tensor([[1, 2], [3, 4]], {device: device});
        const w = new Tensor([[-0.5963, -0.0062],[ 0.1741, -0.1097],[-0.4237, -0.6666],[ 0.1204, 0.2781]], {device: device});
        const wP = w.permute([-1, -2]);
        const y = x.matmul(wP);
    
        assert(equal(y, new Tensor([[-0.6087, -0.0453, -1.7569,  0.6766], [-1.8137,  0.0835, -3.9375,  1.4736]])));
        assert(equal(y.shape, [2, 4]));
    });
    
    TestRunner.describe("Sum", () => {
        const a = new Tensor([0.5, 1.5], {device: device});
        const b = new Tensor([[1, 2], [3, 4]], {device: device});
        const c = new Tensor([[0, 1], [0, 5]], {device: device});
    
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
        const i = new Tensor([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0]], {device: device});
    
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
        ], {device: device})
    
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
        ], {device: device})
    
        assert(equal(y.sum(), new Tensor([378])));
        assert(equal(y.sum(0), new Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[10, 11, 12], [13, 14, 15], [16, 17, 18]]], [[[19, 20, 21], [22, 23, 24], [25, 26, 27]]]])));
        assert(equal(y.sum(1), new Tensor([[[[30, 33, 36], [39, 42, 45], [48, 51, 54]]]])));
        assert(equal(y.sum(2), new Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]])));
        assert(equal(y.sum(3), new Tensor([[[[12, 15, 18]], [[39, 42, 45]], [[66, 69, 72]]]])));
        assert(equal(y.sum(4), new Tensor([[[[6, 15, 24]], [[33, 42, 51]], [[60, 69, 78]]]])));
        assert(equal(y.sum(-1), new Tensor([[[[6, 15, 24]], [[33, 42, 51]], [[60, 69, 78]]]])));
        assert(equal(y.sum(-2), new Tensor([[[[12, 15, 18]], [[39, 42, 45]], [[66, 69, 72]]]])));
    })
    
    TestRunner.describe("Mean", () => {
        const a = new Tensor([[1, 2], [3, 4]], {device: device});
    
        assert(equal(a.mean(), new Tensor([2.5])));
        assert(equal(a.mean(null, true), new Tensor([[2.5]])));
    
        assert(equal(a.mean(0, true), new Tensor([[2, 3]])));
        assert(equal(a.mean(0, false), new Tensor([2, 3])));
    
        assert(equal(a.mean(1, true), new Tensor([[1.5], [3.5]])));
        assert(equal(a.mean(1, false), new Tensor([1.5, 3.5])));
    
        const b = new Tensor([[[0, 1, 2], [3, 4, 5]]], {device: device});
        assert(equal(b.mean(2, true), new Tensor([[[1], [4]]])));
        assert(equal(b.mean(-1, true), new Tensor([[[1], [4]]])));
        assert(equal(b.mean(-1, false), new Tensor([[1, 4]])));
    });
    
    TestRunner.describe("Var", () => {
        const a = new Tensor([[1, 2], [3, 4]], {device: device});
    
        assert(equal(a.var(), new Tensor([1.25])));
        assert(equal(a.var(null, true), new Tensor([[1.25]])));
    
        assert(equal(a.var(0, true), new Tensor([[1, 1]])));
        assert(equal(a.var(0, false), new Tensor([1, 1])));
    
        assert(equal(a.var(1, true), new Tensor([[0.25], [0.25]])));
        assert(equal(a.var(1, false), new Tensor([0.25, 0.25])));
    });
    
    TestRunner.describe("Abs", () => {
        const a = new Tensor([[-3, 3, 3], [4, -4, 4]], {device: device});
        assert(equal(a.abs(), new Tensor([[3, 3, 3], [4, 4, 4]])));
    })
    
    TestRunner.describe("Log", () => {
        const a = new Tensor([[1, 2, 3], [4, 5, 6]], {device: device});
        assert(equal(a.log(), new Tensor([[0,0.6931,1.0986],[1.3863,1.6094,1.7918]]), 1e-4));
    })
    
    TestRunner.describe("Equal", () => {
        const a = new Tensor([1, 2, 3], {device: device});
        const b = new Tensor([0, 2, 2], {device: device});
        const c = a.eq(b);
    
        assert(equal(c, new Tensor([0, 1, 0])));
    
        const d = new Tensor([3], {device: device});
        const e = a.eq(d);
        assert(equal(e, new Tensor([0, 0, 1])));
    })
    
    TestRunner.describe("Not equal", () => {
        const a = new Tensor([1, 2, 3], {device: device});
        const b = new Tensor([0, 2, 2], {device: device});
        const c = a.ne(b);
    
        assert(equal(c, new Tensor([1, 0, 1])));
    
        const d = new Tensor([3], {device: device});
        const e = a.ne(d);
        assert(equal(e, new Tensor([1, 1, 0])));
    })
    
    TestRunner.describe("Greater than", () => {
        const a = new Tensor([1, 2, 3], {device: device});
        const b = new Tensor([0, 2, 2], {device: device});
        const c = a.gt(2);
    
        assert(equal(c, new Tensor([0, 0, 1])));
    
        const d = a.gt(b);
        assert(equal(d, new Tensor([1, 0, 1])));
    
        const e = a.gte(b);
        assert(equal(e, new Tensor([1, 1, 1])));
    })
    
    TestRunner.describe("Less than", () => {
        const a = new Tensor([1, 2, 3], {device: device});
        const b = new Tensor([0, 2, 2], {device: device});
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
    
    TestRunner.describe("Less than equal", () => {
        const a = new Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], {device: device});
        const b = new Tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], {device: device});
    
        const c = a.lte(b);
        assert(equal(c, new Tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]])));
    })
    
    TestRunner.describe("Where using scalar", () => {
        const a = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], {device: device});
        const c = Tensor.where(a.lt(5), a, a.mul(10));
    
        assert(equal(c, new Tensor([1, 2, 3, 4, 50, 60, 70, 80, 90])));
    })
    
    TestRunner.describe("Where using matrix", () => {
        const x = new Tensor([[1, 2], [3, 4]], {device: device});
        const y = new Tensor([[9, 8], [7, 6]], {device: device});
    
        const condition = new Tensor([[1, 0], [1, 1]], {device: device});
    
        const c = Tensor.where(condition, x, y);
    
        assert(equal(c, new Tensor([[1, 8], [3, 4]])));
    })
    
    TestRunner.describe("Where using permute", () => {
        const a = new Tensor([[1, 2], [3, 4]], {device: device});
        const b = new Tensor([[0, 0], [1, 1]], {device: device});
        
        const c = Tensor.where(b.eq(0), a, a.mul(10));
        assert(equal(c, new Tensor([[1,2],[30,40]])));
    
        const d = Tensor.where(b.eq(0), a.T, a.T.mul(10));
        assert(equal(d, new Tensor([[1,3],[20,40]])));
    })
    
    TestRunner.describe("Maximum", () => {
        const a = new Tensor([2, 3, 4], {device: device});
        const b = new Tensor([1, 5, 2], {device: device});
        const c = a.maximum(b);
    
        assert(equal(c, new Tensor([2, 5, 4])));
        assert(equal(c.shape, [3]));
    })
    
    TestRunner.describe("Expand", () => {
        const a = new Tensor([[1], [2], [3]], {device: device});
    
        assert(equal(a.expand([3,4]), new Tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]])));
        assert(equal(a.expand([-1,4]), new Tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]])));
    
        const b = new Tensor([[1, 2, 3], [4, 5, 6]], {device: device});
        assert(equal(b.expand([4,2,3]), new Tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])));
    
        const c = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], {device: device});
        assert(equal(c.expand([3,3,3]), new Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])));
    
        const d = new Tensor([1, 2, 3], {device: device});
        assert(equal(d.expand([3,3]), new Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])));
    
        const e = new Tensor([[[1], [2]], [[3], [4]]], {device: device});
        assert(equal(e.expand([2,2,3]), new Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])));
    })
    
    TestRunner.describe("Unsqueeze", () => {
        const a = new Tensor([1, 2, 3, 4], {device: device});
    
        const b = a.unsqueeze(0);
        assert(equal(b, new Tensor([[1, 2, 3, 4]])));
        assert(equal(b.shape, [1, 4]));
    
        const c = a.unsqueeze(1);
        assert(equal(c, new Tensor([[1], [2], [3], [4]])));
        assert(equal(c.shape, [4, 1]));
    })
    
    TestRunner.describe("Masked fill", () => {
        const a = new Tensor([1, 2, 3, 4, 5], {device: device});
        const mask = new Tensor([1, 0, 0, 0, 0], {device: device});
    
        const filled = a.masked_fill(mask, 10);
    
        assert(equal(filled, new Tensor([10, 2, 3, 4, 5])));
    });
    
    TestRunner.describe("Permute", () => {
        const a = new Tensor([[1, 2], [3, 4]], {device: device});
    
        assert(equal(a.permute(), new Tensor([[1, 3], [2, 4]])))
    
        const b = new Tensor([1, 2, 3, 4], {device: device});
        assert(equal(b.permute(), new Tensor([1, 2, 3, 4])))
    
        const c = new Tensor([[[1, 2, 3], [4, 5, 6]]], {device: device});
        assert(equal(c.permute([1, 0, 2]), new Tensor([[[1, 2, 3]], [[4, 5, 6]]])))
    
        const d = Tensor.ones([2, 3, 4, 5], {device: device});
        assert(equal(d.permute(), Tensor.ones([5, 4, 3, 2])))
    
        const e = new Tensor([[1,2], [3,4]], {device: device});
    
        assert(equal(e.permute(), new Tensor([[1,3], [2,4]])));
        assert(equal(e.permute().reshape([1,4]), new Tensor([[1,3,2,4]])));
    });
    
    TestRunner.describe("Transpose", () => {
        const a = new Tensor([[1.0028, -0.9893, 0.5809], [-0.1669, 0.7299, 0.4942]], {device: device});
    
        assert(equal(a.transpose(0, 1), new Tensor([[1.0028, -0.1669],[-0.9893, 0.7299],[ 0.5809, 0.4942]])));
    });
    
    TestRunner.describe("Softmax", () => {
        const a = Tensor.arange(0, 10, 1, {device: device});
        assert(equal(a.softmax(0), new Tensor([0.0001,0.0002,0.0006,0.0016,0.0043,0.0116,0.0315,0.0856,0.2326,0.6321]), 1e-4));
    
        const b = new Tensor([[5, 6, 3]], {device: device});
        assert(equal(b.softmax(-1), new Tensor([[0.2595, 0.7054, 0.0351]]), 1e-4));
    });
    
    TestRunner.describe("Slice", () => {
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
        ], {device: device});
            
        assert(equal(a.slice([null, null, 0]), new Tensor([[0, 4], [8, 12], [16, 20]])));
        assert(equal(a.slice([null, null, 1]), new Tensor([[1, 5], [9, 13], [17, 21]])));
        assert(equal(a.slice([null, null, 2]), new Tensor([[2, 6], [10, 14], [18, 22]])));
        assert(equal(a.slice([null, null, [1,2]]), new Tensor([[[1], [5]], [[9], [13]], [[17], [21]]])));
        assert(equal(a.slice([null, [1,2], [1,2]]), new Tensor([[[5]], [[13]], [[21]]])));
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
        ], {device: device});
    
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
        ], {device: device});
    
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

    TestRunner.describe("Contiguous with slice", () => {
        const data = [
            [
                [-0.4617, 2.0794, -0.3864, -11.851, -9.0403, -0.9475, 0.8994, -2.2933, -0.8202, -5.4271],
                [-4.8378, -4.6949, -5.4385, -12.7535, -8.4812, -1.9216, -5.5221, -3.7003, -5.5581, -10.1265],
            ]
        ];
        const a = new Tensor(data, {device: device});
        const b = a.slice([null, [a.shape[1]-1, a.shape[1]], null]);
        assert(equal(b, new Tensor([[[-4.8378,-4.6949,-5.4385,-12.7535,-8.4812,-1.9216,-5.5221,-3.7003,-5.5581,-10.1265]]])));
        
        const c = b.contiguous();
        assert(equal(c, new Tensor([[[-4.8378,-4.6949,-5.4385,-12.7535,-8.4812,-1.9216,-5.5221,-3.7003,-5.5581,-10.1265]]])));
    })
    
    TestRunner.describe("Tril", () => {
        const a = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], {device: device});
    
        const b = a.tril();
        assert(equal(b, new Tensor([[1, 0, 0], [4, 5, 0], [7, 8, 9], [10, 11, 12]])));
        assert(equal(b.shape, [4, 3]));
    })
    
    TestRunner.describe("Triu", () => {
        const a = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], {device: device});
        
        const b = a.triu();
        assert(equal(b, new Tensor([[1, 2, 3], [0, 5, 6], [0, 0, 9], [0, 0, 0]])));
        assert(equal(b.shape, [4, 3]));
    })

    TestRunner.describe("Split", () => {
        const a = Tensor.arange(0, 10).reshape([5, 2]);

        const b = a.split(2);
        assert(b.length == 3);
        assert(equal(b[0], new Tensor([[0, 1], [2, 3]])));
        assert(equal(b[1], new Tensor([[4, 5], [6, 7]])));
        assert(equal(b[2], new Tensor([[8, 9]])));

        const c = a.split([1,4]);
        assert(c.length == 2);
        assert(equal(c[0], new Tensor([[0, 1]])));
        assert(equal(c[1], new Tensor([[2, 3], [4, 5], [6, 7], [8, 9]])));
    })
}

export const TensorTests = {category: "Tensor", func: TensorTest};