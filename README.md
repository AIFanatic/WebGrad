# Micrograd-ts

## TODO/FIX
* Matrix still being used all over, (in tensor grad should be a TensorBuffer instead of Matrix)
* If requires_grad is made optional tests fail with grad = null when it shouldn't
* Add forward/backward tests for all Operations (Equal/Maximum missing)
* Softmax is fixed but analyse ReduceOps sum keepdim shape calculation (axis=axes[axes.length-1] sounds weird behaviour)
* Start porting some Matrix methods to Tensor (GPT2 needs a few, tril/slice/split/multinomial/cat)