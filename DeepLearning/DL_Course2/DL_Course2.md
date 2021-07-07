# 1. Improving Deep Neural Networks

Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization

- [1. Improving Deep Neural Networks](#1-improving-deep-neural-networks)
  - [1.1. Week 1](#11-week-1)
    - [1.1.1. Basic recipe](#111-basic-recipe)
    - [1.1.2. Regularizaiton](#112-regularizaiton)
    - [Dropout](#dropout)

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

### 1.1.2. Regularizaiton

Intuition:

> By making weight matrices small, you zero out a lot of hidden units. Maybe most units will be killed, leaving a few. Then it'll be more "linear", thus reduce overfitting.

### Dropout

The most commonly used implementation: **Inverted Dropout**.

- Use a `keep_prob` constant to control the probability of keeping a neuron
  ```python
  mask =  np.random.rand(*a.shape) < keep_prob
  a = a * mask / keep_prob
  ```
- Don't use dropout at test time.
