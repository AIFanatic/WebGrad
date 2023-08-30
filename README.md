# Micrograd-ts

## TODO/FIX
* Matrix still being used all over, (in tensor grad should be a TensorBuffer instead of Matrix)
* If requires_grad is made optional tests fail with grad = null when it shouldn't
* Softmax is fixed but analyse ReduceOps sum keepdim shape calculation (axis=axes[axes.length-1] sounds weird behaviour)