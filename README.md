# Micrograd-ts

## TODO/FIX
* If requires_grad is made optional tests fail with grad = null when it shouldn't
* Add tests for all Tensor operations (Split/etc)
* Test ReduceOps with axis being an array, at the moment is only tested with single numbers
* Multinomial and cat (./test/networks/GPT2) should be tensor methods and use Tensor ops instead of working on the data