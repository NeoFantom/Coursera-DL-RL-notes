# 1. Computer Vision and Convolutional Neural Networks

- [1. Computer Vision and Convolutional Neural Networks](#1-computer-vision-and-convolutional-neural-networks)
  - [1.1. Convolution operation](#11-convolution-operation)
    - [1.1.1. Convolution kernels](#111-convolution-kernels)
    - [1.1.2. Padding](#112-padding)
    - [1.1.3. Strided convolution](#113-strided-convolution)
    - [1.1.4. Input-output size](#114-input-output-size)
    - [1.1.5. Notations](#115-notations)
    - [1.1.6. 3 types of CNN layer](#116-3-types-of-cnn-layer)
    - [1.1.7. Pooling](#117-pooling)
    - [1.1.8. Reason of using CNN](#118-reason-of-using-cnn)
  - [1.2. W1A1](#12-w1a1)
    - [1.2.1. Functions](#121-functions)
    - [1.2.2. Mistakes made](#122-mistakes-made)

## 1.1. Convolution operation

Also as cross correlation.

### 1.1.1. Convolution kernels

- Vertical edge filter $\displaystyle \begin{bmatrix}
  1&0&-1\\1&0&-1\\1&0&-1
\end{bmatrix}$
- Horizontal edge filter $\displaystyle \begin{bmatrix}
  1&1&1\\0&0&0\\-1&-1&-1
\end{bmatrix}$
- Sobel filter $\displaystyle \begin{bmatrix}
    1&0&-1\\2&0&-2\\1&0&-1
  \end{bmatrix}$
- Sharr filter $\displaystyle \begin{bmatrix}
    3&0&-3\\10&0&-10\\3&0&-3
  \end{bmatrix}$

### 1.1.2. Padding

Add pixels to the edges of image.

Paddin types:

- *Valid convolution*: no padding
- *Same convolution*: $\frac{f-1}{2}$ padding, $f$ being filter size

### 1.1.3. Strided convolution

Jump $s$ steps instead of 1 step every time you move on.

### 1.1.4. Input-output size

**Input** image width $n$ with filter $f$ padding $p$ stride $s$ $\to$ **Output** width $\lceil \frac{n+2p-f+1}{s} \rceil = \lfloor \frac{n+2p-f}{s} + 1 \rfloor$

### 1.1.5. Notations

- $f ^{[l]}$ filter size
- $p ^{[l]}$ padding
- $s ^{[l]}$ stride
- Input sizes: $n_H ^{[l-1]} \times n_W ^{[l-1]} \times n_C ^{[l-1]}$
  - With $m$ training examples: $m \times n_H ^{[l-1]} \times n_W ^{[l-1]} \times n_C ^{[l-1]}$
- Filter sizes: $f ^{[l]}\times f ^{[l]} \times n_C ^{[l-1]}$, which is also the number of parameters in the filter.
  - $n_C ^{[l]}$ is equal to number of filter in layer $l$, because filters and their results are stacked in the channels dimension.
  - Whole layer stacked together: $f ^{[l]}\times f ^{[l]} \times n_C ^{[l-1]} \times n_C^{[l]}$
- Output sizes: $n_H ^{[l]} \times n_W ^{[l]} \times n_C ^{[l]}$
  - With $m$ training examples: $m\times n_H ^{[l]} \times n_W ^{[l]} \times n_C ^{[l]}$
- Size relation between layers: $n_{H,W} ^{[l]} = \lfloor \frac{n_{H,W} ^{[l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} +1 \rfloor$
- Each **neuron** (containing one filter $\mathcal{F}$) has form $\; \mathbf{a}^{[l]} = f(\mathbf{a}^{[l-1]}*\mathcal{F} + b)$ where $*$ means convolution.
  - Each neuron has $f ^{[l]}\times f ^{[l]} \times n_C ^{[l-1]}+1$ trainable parameters.
  - Each layer has $f ^{[l]}\times f ^{[l]} \times n_C ^{[l-1]} \times n_C ^{[l]}+ n_C ^{[l]}$, because each layer has $n_C ^{[l]}$ filters, or neurons.

### 1.1.6. 3 types of CNN layer

Convolution, pooling, fully connected layer.

### 1.1.7. Pooling

Hyperparameters: filter size, stride.

Types: max pooling, average pooling.

Pooling layer has **no parameters** to learn.

### 1.1.8. Reason of using CNN

1. Parameter sharing: a filter useful here, might be useful there.
1. Sparsity of connections: each output relies on a small part of input.

## 1.2. W1A1

### 1.2.1. Functions

- `zero_pad(X, pad_width) => X_padded`
- `conv_single_step(a_slice_prev, W, b) => Z`
- `conv_forward(A_prev, W, b, hparameters)`

### 1.2.2. Mistakes made

The final convolution step should be `Z[i,h,w,c] = (np.sum(a_slice * weights) + bias).item()`,\
~~but I wrote `Z[i,h,w,c] = (np.sum(a_slice * weights + bias)).item()`~~
