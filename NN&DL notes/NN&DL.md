# Neural Networks and Deep Learning

- [Neural Networks and Deep Learning](#neural-networks-and-deep-learning)
  - [Numpy functions](#numpy-functions)
    - [Random sampling](#random-sampling)
  - [Matplotlib usage](#matplotlib-usage)
  - [W2A1: Python Basics with Numpy](#w2a1-python-basics-with-numpy)
  - [W2A2: Logistic Regression with a Neural Network mindset](#w2a2-logistic-regression-with-a-neural-network-mindset)
  - [2-layer NN calculation](#2-layer-nn-calculation)
    - [NN notation setup](#nn-notation-setup)
    - [Forward](#forward)
    - [Cost](#cost)
    - [Backward](#backward)
  - [W3A1 Planar data classification with one hidden layer](#w3a1-planar-data-classification-with-one-hidden-layer)

## Numpy functions

- `np.ndarray.reshape(*shape)` is often used. Parameter `-1` means auto-shape. E.g.
  - `v.reshape(-1, 1)` generates column vector `v.shape = [N, 1]`
  - `v.reshape(-1)` generates 1d array `v.shape = [N,]`
- `np.linalg.norm(x, ord=2, axis=1, keepdims=True)` calculates 2-norm. See [documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).
- `np.zeros(shape: tuple)` shape is a tuple.
- `np.multiply()` is equivalent to elementwise `A * B`
- `np.squeeze(ndarray, axis=None)` remove axes if this axis has only one element

### Random sampling

See [oneline documentaion](https://numpy.org/doc/stable/reference/random/index.html).

- New version:
  ```python
  # Do this (new version)
  from numpy.random import default_rng
  rng = default_rng()
  vals = rng.standard_normal(10)
  more_vals = rng.standard_normal(10)

  # instead of this (legacy version)
  from numpy import random
  vals = random.standard_normal(10)
  more_vals = random.standard_normal(10)
  ```
- `np.random.rand(dim1, dim2, ...)` uniform distribution $U[0, 1)$
- `np.random.randn(dim1, dim2, ...)` standard normal distribution $N(0, 1)$

## Matplotlib usage

- Show image:
  ```python
  img = tensor(height, width, channels)
  # or
  img = matrix(height, width)
  plt.imgshow(img)
  ```
- Plot curve [see more](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html):
  ```python
  # plot() common usage
  plt.plot(x, y)        # plot x and y, default line style and color
  plt.plot(x, y, 'bo')  # plot x and y, blue circle markers
  plt.plot(y)           # plot y using x as index array 0..N-1
  plt.plot(y, 'r+')     # ditto, but with red plusses
  
  x = np.linspace(0, 10, 1000)
  y = np.sin(x)

  plt.plot(x, y, 'r', label='$\sin x$')

  plt.legend(loc='upper right')
  plt.xlabel(r"$x$")
  plt.ylabel(r"$f(x)$")

  plt.show()
  ```

## W2A1: Python Basics with Numpy

- $\displaystyle \operatorname{sigmoid}(x) = \sigma(x) = \frac{1}{1+e^{-x}}, \quad \frac{d\sigma}{dx} = \sigma(1-\sigma) .$
- $\displaystyle \mathrm{softmax}(x_i) = f(x_i; x_1, \dots, x_n) = \frac{e^{x_i}}{\sum_j e^{x_j}}$
- `axis=1` in parameters means, for matrix, reduce along rows and generate a column vector. E.g. see [np.linalg.norm](#numpy-functions)
- **Always** `keepdims=True`.
- Make use of **broadcasting**.

## W2A2: Logistic Regression with a Neural Network mindset

- Reshape dataset:
  ```python
  # data.shape = (num_imgs, img_height, img_width, img_channels)
  data = data.reshape(data.shape[0], -1).T
  # data.shape = (size_of_img_vec, num_imgs)
  ```
- Preprocessing common steps:
  - Figure out the dimensions
  - Reshape
  - Standardize, for images, devide every image vector by 255
- Algorithm for **logistic regression** (single feature neural):
  - Neuron: $\hat{Y} = A = \sigma(Z) = \sigma(\mathbf{w}^T \mathbf{X}+b)$
  - Cost : $\displaystyle J = \frac{1}{m} \sum_i {L}(\hat y ^{(i)}, y ^{(i)}) = -\frac{1}{m}\sum_{i=1}^{m}\Big(y^{(i)}\ln(\hat y^{(i)})+(1-y^{(i)})\ln(1-\hat y^{(i)})\Big)$
  - Derivatives : $\displaystyle \frac{dL}{dz} = \hat y-y, \quad \frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m}X(\hat{Y}-Y)^T , \quad \displaystyle \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)}-y^{(i)})$
- Functions in this assignment:
  - `initialize(dim) => w: of shape(dim, 1), b: number`
  - `propagate(w, b, X, Y) => grads{dw, db}, cost: number`
  - `optimize(w, b, X, Y, num_iterations, learning_rate, print_cost) => params{w, b}, grads{dw, db}, cost: number`
  - `predict(w, b, X) => Yhat`
  - Model:
    ```python
    def model(X_train, Y_train, X_test, Y_test, 
      num_iterations=2000, learning_rate=0.5, print_cost=False):
      
        d = {
          "costs": costs,
          "Y_prediction_test": Y_prediction_test, 
          "Y_prediction_train" : Y_prediction_train, 
          "w" : w, 
          "b" : b,
          "learning_rate" : learning_rate,
          "num_iterations": num_iterations`
        }
        return d
    ```

## 2-layer NN calculation

### NN notation setup

1. $X=(x ^{(1)}, x ^{(2)}, \dots, x ^{(m)})$ has shape $(n_x \times m)$ with $n_x$ being size of features, $m$ being number of training examples.
1. $Y=(y ^{(1)}, y ^{(2)}, \dots, y ^{(m)})$ has shape $(1\times m)$, each a boolean variable.
1. $\displaystyle x^{(i)}$ is the $i$-th training example
1. $\square^{[j](i)}_k$ means the $k$-th neuron in $j$-th layer, acting on $i$-th training example.
   1. $\square^{[i]}$ means for $i$-th hidden layer (input layer counted as 0-th)

### Forward

$$ \begin{aligned}
  a ^{[0]} &= {X} \\
  z ^{[1]} &= W ^{[1]} a ^{[0]} + b ^{[1]} \\
  a ^{[1]} &= \tanh ( z ^{[1]} ) \\
  z ^{[2]} &= W ^{[2]} a ^{[1]} + b ^{[2]} \\
  a ^{[2]} &= \sigma ( z ^{[2]} ) \\
  \hat {Y} &= a ^{[2]}
\end{aligned} $$

### Cost

$$ \begin{aligned}
  J & = - \frac{1}{m} \sum\limits_{i = 1}^{m} \large{(} \small y^{(i)}\ln\left(a^{[2] (i)}\right) + (1-y^{(i)})\ln\left(1- a^{[2] (i)}\right) \large{)} \small\tag{13} \\
  & = -\frac{1}{m} \Big(\mathbf{Y} \cdot \ln ( \mathbf{a} ^{[2]} )^T + (1 - \mathbf{Y}) \cdot \ln ( 1 - \mathbf{a} ^{[2]} )^T )
\end{aligned} $$

### Backward

Notation: $\circ$ denotes Hadamard elementwise product.

- First layer

  $$ \newcommand{\pdiff}[1]{\frac{\partial J}{\partial #1}} \begin{aligned}
    \pdiff{z ^{[2]}} &= a ^{[2]} - Y \\
    \pdiff{w ^{[2]}} &= \frac{1}{m} \pdiff{z ^{[2]}}  {A ^{[1]}} ^T \\
    \pdiff{b ^{[2]}} &= \frac{1}{m} (1, \dots, 1)  \pdiff{z ^{[2]}} \\
  \end{aligned} $$
- Second layer:

  $$ \newcommand{\pdiff}[1]{\frac{\partial J}{\partial #1}} \begin{aligned}
    \boxed{\frac{\partial a ^{[1]}}{\partial z ^{[1]}}} &= 1 - a ^{[1]} \circ a ^{[1]} \\
    \pdiff{z ^{[1]}} &= {W ^{[2]}}^T \pdiff{z ^{[2]}} \circ \boxed{\frac{\partial a ^{[1]}}{\partial z ^{[1]}}} \\
    \pdiff{w ^{[1]}} &= \frac{1}{m} \pdiff{z ^{[1]}} X^T \\
    \pdiff{b ^{[1]}} &= \frac{1}{m} (1, \dots, 1)  \pdiff{z ^{[2]}}
  \end{aligned} $$

Math details:

With $L(a,y)=-y\ln a - (1-y)\ln(1-a)$, we have $\frac{\partial L}{\partial a} = -\frac{y}{a} - \frac{1-y}{1-a}$. Use chain rule $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \frac{\partial a}{\partial z}$, and $\frac{\partial a}{\partial z} = \sigma' (z)=a(1-a)$, we get $\frac{\partial L}{\partial z} = a - y$.

Andrew Ng's vectorization summary:

![Summary of gradient descent by Andrew Ng](./image/2021-0529-192453.png)

## W3A1 Planar data classification with one hidden layer

> This is an implementation of shallow neural network.

Variables:

- `parameters = {W1, b1, W2, b2}`
- `grads = {dW1, db1, dW2, db2}`

Functions:

- `initialize_parameters(n_x, n_h, n_y) => parameters`
- `forward_propagation(X, parameters) => Yhat, cache{Z1, A1, Z2, A2}`
- `compute_cost(A2, Y) => cost: number`
- `backward_propagation(parameters, cache, X, Y) => grads`
- `update_parameters(parameters, grads, learning_rate = 1.2) => parameters`
- `nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False) => paramete