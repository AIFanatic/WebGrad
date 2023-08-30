# Micrograd-ts

## TODO/FIX
* Softmax is using matrix sum (Tensor sum must have bugs).
* If requires_grad is made optional tests fail with grad = null when it shouldn't
* Matrix still being used all over, (in tensor grad should be a TensorBuffer instead of Matrix)