# Neural Networks and Deep Learning

- [Neural Networks and Deep Learning](#neural-networks-and-deep-learning)
  - [numpy functions](#numpy-functions)
  - [W2A1: Python Basics with Numpy](#w2a1-python-basics-with-numpy)
  - [W2A2: Logistic Regression with a Neural Network mindset](#w2a2-logistic-regression-with-a-neural-network-mindset)

## numpy functions

- `np.ndarray.reshape(*shape)` is often used. Parameter `-1` means auto-shape. E.g. 
  - `v.reshape(-1, 1)` generates column vector `v.shape = [N, 1]`
  - `v.reshape(-1)` generates 1d array `v.shape = [N,]`
- `np.linalg.norm(x, ord=2, axis=1, keepdims=True)` calculates 2-norm. See [documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).
- `np.random.rand(*shape)` shape is given one by one.
- `np.zeros(shape: tuple)` shape is a tuple.
- `np.multiply()` is equivalent to elementwise `A*B`

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
  - Cost : $\displaystyle J = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$
  - Derivatives : $\displaystyle \frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m}X(\hat{Y}-Y)^T , \; \displaystyle \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)}-y^{(i)})$
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
          "num_iterations": num_iterations
        }
        return d
    ```
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
