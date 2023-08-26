# Micrograd-ts

## TODO/FIX
* In some cases (broadcast, tri, etc) the matrix needs to be recreated in order to "normalize" the strides, this is because some ops don't work properly with the strides, therefore a hack is implemented:
```typescript
// TODO: Fix
const aData = a.getData();
const bData = b.getData();
a = new Matrix(aData instanceof Array ? aData : [aData]);
b = new Matrix(bData instanceof Array ? bData : [bData]);
```

* Some operations rely on Math stuff (tanh, exp etc) this is kinda of a nono once we want to port to the gpu.

* Following the point above no operations should be performed on the Matrix class with the exception of binary ops. Later these ops will be part of a runtime that can either be the cpu/gpu/accel etc.

* Make reshape (implement view) behave like numpy. With "reshape" the original data cannot be modified, while with "view" it can, with view the data is shared across all instances.

* Port the Matrix class into the Tensor class. For now its easier to keep them separate, but all should be a Tensor.
* Rename `Model` to `Module` and update codebase accordingly.

* Add `ModuleDict` and `ModuleList`.

* Get rid of `this.layers.push(layer)` on `Model` constructor, this is needed for the forward calls