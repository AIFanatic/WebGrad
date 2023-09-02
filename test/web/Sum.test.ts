import { describe, assert, equal } from "../TestUtils";
import { Tensor } from "../../src/Tensor";

import { Random } from "../../src/Random";
import { Device } from "../../src";
import { TestRunner } from "../run-web";

Random.SetRandomSeed(1337);

function SumTest(device: Device) {
   
    TestRunner.describe("Sum", () => {
        const a = new Tensor([0.5, 1.5], {device: device});
        const b = new Tensor([[1, 2], [3, 4]], {device: device});
        const c = new Tensor([[0, 1], [0, 5]], {device: device});
    
        const d = a.sum();
        console.log(`d ${d}`)
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
    
    // TestRunner.describe("Mean", () => {
    //     const a = new Tensor([[1, 2], [3, 4]], {device: device});
    
    //     assert(equal(a.mean(), new Tensor([2.5])));
    //     assert(equal(a.mean(null, true), new Tensor([[2.5]])));
    
    //     assert(equal(a.mean(0, true), new Tensor([[2, 3]])));
    //     assert(equal(a.mean(0, false), new Tensor([2, 3])));
    
    //     assert(equal(a.mean(1, true), new Tensor([[1.5], [3.5]])));
    //     assert(equal(a.mean(1, false), new Tensor([1.5, 3.5])));
    
    //     const b = new Tensor([[[0, 1, 2], [3, 4, 5]]], {device: device});
    //     assert(equal(b.mean(2, true), new Tensor([[[1], [4]]])));
    //     assert(equal(b.mean(-1, true), new Tensor([[[1], [4]]])));
    //     assert(equal(b.mean(-1, false), new Tensor([[1, 4]])));
    // });
    
    // TestRunner.describe("Var", () => {
    //     const a = new Tensor([[1, 2], [3, 4]], {device: device});
    
    //     assert(equal(a.var(), new Tensor([1.25])));
    //     assert(equal(a.var(null, true), new Tensor([[1.25]])));
    
    //     assert(equal(a.var(0, true), new Tensor([[1, 1]])));
    //     assert(equal(a.var(0, false), new Tensor([1, 1])));
    
    //     assert(equal(a.var(1, true), new Tensor([[0.25], [0.25]])));
    //     assert(equal(a.var(1, false), new Tensor([0.25, 0.25])));
    // });
}

export const SumTests = {category: "Sum", func: SumTest};