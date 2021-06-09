# 1. Computer Vision and Convolutional Neural Networks

- [1. Computer Vision and Convolutional Neural Networks](#1-computer-vision-and-convolutional-neural-networks)
  - [1.1. Convolution](#11-convolution)
    - [1.1.1. Convolution kernels](#111-convolution-kernels)
    - [1.1.2. Padding](#112-padding)
    - [1.1.3. Strided convolution](#113-strided-convolution)
    - [1.1.4. Input-output size](#114-input-output-size)
  - [Notations](#notations)

## 1.1. Convolution

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

## Notations

- $f ^{[l]}$ filter size
- $p ^{[l]}$ padding
- $s ^{[l]}$ stride
- Input sizes: $n_H ^{[l-1]} \times n_W ^{[l-1]} \times n_C ^{[l-1]}$
- Filter sizes: $f ^{[l]}\times f ^{[l]} \times n_C ^{[l-1]}$
- Output sizes: $n_H ^{[l]} \times n_W ^{[l]} \times n_C ^{[l]}$
- Size relation between layers: $n_{H,W} ^{[l]} = \lfloor \frac{n_{H,W} ^{[l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} +1 \rfloor$
- $n_C ^{[l]}$ is equal to number of filter in layer $l$
