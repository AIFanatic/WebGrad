# Micrograd-ts

## TODO/FIX
* If requires_grad is made optional tests fail with grad = null when it shouldn't
* Add forward/backward tests for all Operations (Equal/Maximum missing)
* Add tests for all Tensor operations (Split/reshape/etc)
* Softmax is fixed but analyse ReduceOps sum keepdim shape calculation (axis=axes[axes.length-1] sounds weird behaviour)
* Multinomial and cat (./test/networks/GPT2) should be tensor methods and use Tensor ops instead of working on the data