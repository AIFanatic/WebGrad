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