# Python Packages Usage

- [Python Packages Usage](#python-packages-usage)
  - [Numpy functions](#numpy-functions)
    - [Glossary](#glossary)
    - [Array manipulations](#array-manipulations)
    - [Array arithmetics](#array-arithmetics)
    - [Generate specific type array](#generate-specific-type-array)
    - [Random sampling](#random-sampling)
    - [Working with images](#working-with-images)
  - [Matplotlib usage](#matplotlib-usage)

## Numpy functions

### Glossary

See definition of [along an axis](https://numpy.org/doc/stable/glossary.html)

### Array manipulations

- `reshape(*shape)` is often used. Parameter `-1` means auto-shape. E.g.
  - `v.reshape(-1, 1)` generates column vector `v.shape = [N, 1]`
  - `v.reshape(-1)` generates 1d array `v.shape = [N,]`
- `squeeze(axis=None)` remove axes if this axis has only one element.
- `item(index)` returns index-th element, if `None`, the array must be single element.

### Array arithmetics

- `np.linalg.norm(x, ord=2, axis=1, keepdims=True)` calculates 2-norm. See [documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).
- `np.multiply()` is equivalent to elementwise `A * B`.

### Generate specific type array

- `np.zeros(shape: tuple)` shape is a tuple.

### Random sampling

See [oneline documentaion](https://numpy.org/doc/stable/reference/random/index.html).

- New version with random number generator:
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
- `random.choice(a, size=None, replace=True, p=None)` Generates a random sample from a given 1-D array

### Working with images

- `numpy.pad(array, pad_width, mode='constant', **kwargs)` do padding. [See documentaion](https://numpy.org/doc/stable/reference/generated/numpy.pad.html)
  - `pad_width` takes form `((p1_front, p1_end), ... (pn_front, pn_end))`, pads the array at `j`-th axis with `pj_front` at front and `pj_end` at end.

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
