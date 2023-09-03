# WebGrad

[WebGrab web tests](https://aifanatic.github.io/WebGrad/dist/test/)

## TODO/FIX
* If requires_grad is made optional tests fail with grad = null when it shouldn't
* Add tests for all Tensor operations (Split/etc)
* Test ReduceOps with axis being an array, at the moment is only tested with single numbers
* Multinomial and cat (./test/networks/GPT2) should be tensor methods and use Tensor ops instead of working on the data
* Figure out a way for rand on webgl to match cpu
* Get all webgl tests to pass ./dist/test/index.html.
* Probably device come from TensorBuffer and not Tensor (From TensorBuffer is more reliable and can't be faked)
* Test tensor assign (probably it should copy stuff instead of assigning by reference)
* WEBGL reduce_op needs a reshape for the resulting shape, analyse ways around not using any more ops there