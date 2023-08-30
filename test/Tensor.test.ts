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

describe("Unsqueeze", () => {
    const a = new Tensor([1, 2, 3, 4]);

    const b = a.unsqueeze(0);
    assert(equal(b, new Tensor([[1, 2, 3, 4]])));
    assert(equal(b.shape, [1, 4]));

    const c = a.unsqueeze(1);
    assert(equal(c, new Tensor([[1], [2], [3], [4]])));
    assert(equal(c.shape, [4, 1]));
})