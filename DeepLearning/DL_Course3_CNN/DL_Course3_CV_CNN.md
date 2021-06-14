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
  - [1.2. W1A1 Convolution model Step by Step](#12-w1a1-convolution-model-step-by-step)
    - [1.2.1. Functions](#121-functions)
    - [1.2.2. Mistakes made](#122-mistakes-made)
    - [1.2.3. What you should remember](#123-what-you-should-remember)
    - [1.2.4. Back propagation](#124-back-propagation)
  - [1.3. W1A2 Convolutional Neural Networks: Application](#13-w1a2-convolutional-neural-networks-application)
    - [1.3.1. Keras](#131-keras)
    - [1.3.2. Functions](#132-functions)
    - [1.3.3. Plot train history](#133-plot-train-history)
  - [1.4. Read-world solution](#14-read-world-solution)
    - [1.4.1. Papers](#141-papers)
    - [1.4.2. Modern networks](#142-modern-networks)
    - [1.4.3. Transfer learning](#143-transfer-learning)
    - [1.4.4. Data augmentation](#144-data-augmentation)
    - [1.4.5. Benchmarking and competition tips](#145-benchmarking-and-competition-tips)

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

## 1.2. W1A1 Convolution model Step by Step

### 1.2.1. Functions

- `zero_pad(X, pad_width) => X_padded`
- `conv_single_step(a_slice_prev, W, b) => Z`
- `conv_forward(A_prev, W, b, hparameters) => Z, cache(A_prev, W, b, hparameters)`
- `pool_forward(A_prev, hparameters, mode = "max") => A, cache(A_prev, hparameters)`
- `conv_backward(dZ, cache) => dA_prev, dW, db`
- `create_mask_from_window(x) => mask`
- `distribute_value(dz: number, shape) => average_matrix`
- `pool_backward(dA, cache, mode = "max") => dA_prev`

### 1.2.2. Mistakes made

The final convolution step should be `Z[i,h,w,c] = (np.sum(a_slice * weights) + bias).item()`,\
~~but I wrote `Z[i,h,w,c] = (np.sum(a_slice * weights + bias)).item()`~~

### 1.2.3. What you should remember

- A convolution extracts features from an input image by taking the dot product between the input data and a 2D array of weights (the filter).
- The 2D output of the convolution is called the *feature map*
- A convolution layer is where the filter slides over the image and computes the dot product
  - This transforms the input volume into an output volume of different size
- Zero padding helps keep more information at the image borders, and is helpful for building deeper networks, because you can build a CONV layer without shrinking the height and width of the volumes
- Pooling layers gradually reduce the height and width of the input by sliding a 2D window over each specified region, then summarizing the features in that region

### 1.2.4. Back propagation

Mathematical calculations:

$$dA \mathrel{+}= \sum _{h=0} ^{n_H} \sum_{w=0} ^{n_W} W_c \times dZ_{hw} \tag{1}$$

$$dW_c \mathrel{+}= \sum_{h=0} ^{n_H} \sum_{w=0} ^ {n_W} a_{slice} \times dZ_{hw}  \tag{2}$$

$$db = \sum_h \sum_w dZ_{hw} \tag{3}$$

In python code, for one training example:

```python
da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += 
    W[:,:,:,c] * dZ[i, h, w, c]
dW[:,:,:,c] \mathrel{+}= a_slice * dZ[i, h, w, c]
db[:,:,:,c] += dZ[i, h, w, c]
```

## 1.3. W1A2 Convolutional Neural Networks: Application

### 1.3.1. Keras

- See keras.io
- If you want to print summary of layers, you need to [specify the input shape in advance](https://keras.io/guides/sequential_model/#specifying-the-input-shape-in-advance).
  - Easiest way is to use `input_shape` kwarg.

### 1.3.2. Functions

```python
import tensorflow.keras.layers as tfl
# Sequantial model
model = tf.keras.Sequantial([tfl.Conv2D(...), ...])
# Functional API
inputs = tf.keras.Input(shape=(...))
t = tfl.Conv2D(...)(inputs)
t = tfl...
output = tfl.Dense(...)(t)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

### 1.3.3. Plot train history

```python
df_loss_acc = pd.DataFrame(history.history)
df_loss = df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc = df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```

## 1.4. Read-world solution

### 1.4.1. Papers

- LeNet-5
  - Just read section II and III.
  - Has some old-fashioned techniques with number of channels, no ReLU activations, nonlinearity after pooling.
  - A small model, input dimension: (32,32,1)
- AlexNet
  - Not important things like multiple GPUs training, local response normalization (not so effective)
  - Easier paper to read.
  - First paper to have the CV scientist to take a godd look at DL.
  - Input dimension: (227,227,3)
- VGG-16
  - Simple architechture, few hyperparameters, relatively deep, many trainable parameters.
  - Half the height and width every layer, double the channels.
  - Input dimension: (224,224,3)

### 1.4.2. Modern networks

- ResNet
  - Basic block: residual network.
  - Reason it does well going deeper: Because the worst it does is just to learn identity function and become a shallow network. If any luck, it'll learn some useful information.
- Inception (GoogLeNet)
  - *1×1 convolutions* reduce channel dimension, also called *network in network*
  - Motivations
    - Use 1×1 convolutions to reduce channels and then do convolution, thus reducing parameters ten times fewer.
    - Instead of choosing a convolution filter size, let's do them all, stack them together, then apply a 1×1 convolution.
  - Name origin: in movie *Inception*, Leo says "We need to go deeper".
  - Also has some side branches to ensure the middle layers calculate good enough features for prediction.
- MobileNet
  - MobileNet v1 core idea: depthwise seperable convolutions (as compared to normal convolutions)
    - First do a single channel convolution on every channel.
    - Then do a 1×1 convolution on every pixel.
  - MobileNet v2 bottleneck:
    - Skip connection in parallel.
    - Expansion 1×1 convolution, depthwise convolution, projection 1×1 convolution.
- EfficientNet: A good way to trade off between $r$ (resolution), $d$ (network depth), $w$ (layer width). Look into the source code.

### 1.4.3. Transfer learning

### 1.4.4. Data augmentation

- Geometric distortion.
- Color shifting.
  - PCA color augmentation.

### 1.4.5. Benchmarking and competition tips

- Ensenbling: train multiple networks and average their **results.**
- Test time multi-cropping: feed multiple cropped verisons of a test image, average their results.
