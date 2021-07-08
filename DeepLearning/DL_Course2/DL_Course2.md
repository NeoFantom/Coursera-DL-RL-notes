# 1. Improving Deep Neural Networks

Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization

- [1. Improving Deep Neural Networks](#1-improving-deep-neural-networks)
  - [1.1. Week 1](#11-week-1)
    - [1.1.1. Basic recipe](#111-basic-recipe)
    - [1.1.2. $L^2$ Regularizaiton](#112-l2-regularizaiton)
    - [1.1.3. Dropout](#113-dropout)
    - [1.1.4. Other methods reducing overfitting](#114-other-methods-reducing-overfitting)
    - [1.1.5. Normalizing inputs](#115-normalizing-inputs)
    - [1.1.6. Initializing weights](#116-initializing-weights)
    - [1.1.7. Gradient checking](#117-gradient-checking)

## 1.1. Week 1

**Bias**: training performance.

**Variance**: validation performance.

### 1.1.1. Basic recipe

- High bias (wrong prediction):
  - bigger network
  - train longer
  - NN architechture search
- High variance (overfitting):
  - more data
  - regularization
  - NN architechture search

### 1.1.2. $L^2$ Regularizaiton

Intuition:

> By making weight matrices small, you zero out a lot of hidden units. Maybe most units will be killed, leaving a few. Then it'll be more "linear", thus reduce overfitting.

### 1.1.3. Dropout

The most commonly used implementation: **Inverted Dropout**.

- Use a `keep_prob` constant to control the probability of keeping a neuron
  ```python
  mask =  np.random.rand(*a.shape) < keep_prob
  a = a * mask / keep_prob
  ```
- Don't use dropout at test time.

### 1.1.4. Other methods reducing overfitting

1. Data augmentation
   - distortion
   - shift
   - clip
   - rotation
1. Early stopping\
   The train error always goes down, while at some time the validation error goes up. So stop earlier before validation error going up.
1. Orthogonalization
   1. Don't intertwine different tasks together, do them orthogonally.

### 1.1.5. Normalizing inputs

Make the inputs have 0 mean and 1 variance.

Purpose: make every feature to the same scale, speed up learning process.

$$\begin{gather*}
  \mu = \frac{1}{m} \sum_{i=1}^{m} x ^{(i)}, \quad \sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x ^{(i)} - \mu)^2 \\
  x = \frac{x - \mu}{\sigma}
\end{gather*}$$

### 1.1.6. Initializing weights

We often have the problem of **gradient vanishing** or **gradient exploding**. To solve this, a not perfect yet relatively effective way is to initialize your weights carefully.

Formally, we want to make $z=w^T x$ about the same size as each component $x_i$. So we make the variance $\operatorname{Var}(w_i) = \frac{1}{n}$ where $n$ is the length of vector $x$. So we do
```python
w[l] = np.random.randn(shape) * np.sqrt(1 / len(w[l-1]))
```
Or we can choose other variances:

- $\displaystyle \operatorname{Var}(w_i) = \frac{2}{n^{[l-1]}}$ for ReLU activation
- $\displaystyle \operatorname{Var}(w_i) = \sqrt{ \frac{1}{n^{[l-1]}} }$ for tanh activation
- $\displaystyle \operatorname{Var}(w_i) = \sqrt{ \frac{2}{n^{[l-1]} + n^{[l]}} }$ , called Xavier initialization.

### 1.1.7. Gradient checking

A better derivative/gradient numerical approximation:

$$ f'(x) = \frac{ f(x+\delta) - f(x-\delta) }{2\delta} + o(\delta^2) $$

Sometimes it is helpful to do gradient checking to find out possible bugs. The goal is to check whether the calculated gradients are close to real gradients. The real gradient w.r.t. $\Theta_i$ is

$$ d\tilde{\Theta}_i = J(\Theta_i + \delta) - J(\Theta_i - \delta) / 2\delta$$

and we compare it to our calculated gradient by making sure that

$$ \frac{\lVert d\tilde{\Theta} - d\Theta \rVert_2}{\lVert d\tilde{\Theta} \rVert_2 + \lVert d\Theta \rVert_2} \le 10^{-7} $$

If this error is of order $10^{-3}$, then one should be worried.

Also:

- Remember regularizition
- Doesn't work with dropout
- Sometimes code works bad when $w,b$ are sufficient large, that case, run multiple times with random initialization.
