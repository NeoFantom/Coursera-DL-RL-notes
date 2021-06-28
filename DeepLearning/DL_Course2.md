# Improving Deep Neural Networks

Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization

- [Improving Deep Neural Networks](#improving-deep-neural-networks)
  - [Week 1](#week-1)
    - [Regularizaiton](#regularizaiton)

## Week 1

**Bias**: training performance.

**Variance**: validation performance.

Basic recipe:

- High bias (wrong prediction):
  - bigger network
  - train longer
  - NN architechture search
- High variance (overfitting):
  - more data
  - regularization
  - NN architechture search

### Regularizaiton

Intuition:

> By making weight matrices small, you zero out a lot of hidden units. Maybe most units will be killed, leaving a few. Then it'll be more "linear", thus reduce overfitting.
