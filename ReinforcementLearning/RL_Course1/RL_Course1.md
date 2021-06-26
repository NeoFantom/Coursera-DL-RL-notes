# RL Course 1

- [RL Course 1](#rl-course-1)
  - [Assignment 1: Bandits and Exploration/Exploitation](#assignment-1-bandits-and-explorationexploitation)
  - [Week 2](#week-2)

## Assignment 1: Bandits and Exploration/Exploitation

1. `argmax(q_values) => action: int` random choice of tied maximal `q_values`
1. `GreedyAgent.agent_step(self, reward, observation=None) => currentAction`

## Week 2

Question: The reward hypothesis states “that all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).” Can you think of a situation that is **not** well-modeled by maximizing a scalar reward signal?

I think multiple-goal optimization problems are not well modeled by scalar reward signal. Consider this simple example: A computer needs to *learn how to distribute execution time among users*. Now if we simply assign each action a reward, where an action means to excute some user's task, this policy will result in some user's task not excuted at all, because the computer doesn't distinguish between different users. We can, however, change our reward and goal to some complicated functions that takes into account user's waiting time. Actually this is a typical problem in operating systems designs. We can do pretty well, but certainly not with single goal, or single scalar reward, because when our priorities changes, it'll be hard to change our algorithm accordingly.
