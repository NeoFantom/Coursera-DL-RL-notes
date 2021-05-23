# Neural Networks and Deep Learning

- [Neural Networks and Deep Learning](#neural-networks-and-deep-learning)
  - [W2A1: Python Basics with Numpy](#w2a1-python-basics-with-numpy)

## W2A1: Python Basics with Numpy

- $\displaystyle \mathrm{sigmoid}(x) = \sigma(x) = \frac{1}{1+e^{-x}}, \quad \frac{d\sigma}{dx} = \sigma(1-\sigma) .$
- $\displaystyle \mathrm{softmax}(x_i) = f(x_i; x_1, \dots, x_n) = \frac{e^{x_i}}{\sum_j e^{x_j}}$
- `np.ndarray.reshape(*shape)` is often used. Parameter `-1` means auto-shape. E.g. 
  - `v.reshape(-1, 1)` generates column vector `v.shape = [N, 1]`
  - `v.reshape(-1)` generates 1d array `v.shape = [N,]`
- `np.linalg.norm(x, ord=2, axis=1, keepdims=True)` calculates 2-norm. See [documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).
- `axis=1` in parameters means, for matrix, reduce along rows and generate a column vector.
- **Always** `keepdims=True`.
- Make use of **broadcasting**.
- `np.random.rand(*shape)` gives random ndarray of given shape.
- `np.multiply()` is equivalent to elementwise `A*B`
