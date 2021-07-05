# 1. RL Course 1

- [1. RL Course 1](#1-rl-course-1)
  - [1.1. Assignment 1: Bandits and Exploration/Exploitation](#11-assignment-1-bandits-and-explorationexploitation)
  - [1.2. Week 2](#12-week-2)
    - [1.2.1. Markov decesion process (MDP)](#121-markov-decesion-process-mdp)
      - [1.2.1.1. MDP types](#1211-mdp-types)
    - [1.2.2. Two types of tasks](#122-two-types-of-tasks)
    - [1.2.3. Discussion](#123-discussion)
    - [1.2.4. Peer graded assignments](#124-peer-graded-assignments)

## 1.1. Assignment 1: Bandits and Exploration/Exploitation

1. `argmax(q_values) => action: int` random choice of tied maximal `q_values`
1. `GreedyAgent.agent_step(self, reward, observation=None) => currentAction`

## 1.2. Week 2

### 1.2.1. Markov decesion process (MDP)

![MDP model](RL_Course1-images/2021-0628-105654.png)

- In *MDP*, actions influence not just immediate rewards like in *bandit problems*, but also subsequent situations, or states, and through those future rewards.
- There is a tradeoff: immediate and delayed rewards.
- Estimated values are different for *MDP* and *bandit*
  - In *bandit problems*, we estimate the value of action $q_*(a)$.
  - In *MDP*, we estimate the value of an action $a$ in given state $s$, aka $q_*(s, a)$. <span style='color:red'> or we estimate the value $v_*(s)$ of each state given optimal action selections.</span>
- An MDP is completely described by a set of probabilities, or the **dynamics** of the MDP
  $$p(s_{t+1}, r_{t+1} \mid s_t, a_t)$$
- A state is said to have **Markov property**, if it contains information about all aspects of the past agent–environment interaction that make a di↵erence for the future.

#### 1.2.1.1. MDP types

1. *Finite MDP* \
   The sets of states, actions, and rewards $(S, A, R)$ all have a finite number of elements.

### 1.2.2. Two types of tasks

1. Episodic.
   1. The task ends at some point, so it has an *ending state*. From start to end, it's called an *episode*. E.g. playing games, solving mazes.
   1. Expected return:
      $$G_t := R_{t+1} + R_{t+2} + \dots + R_{T}$$
1. Continuing.
   1. The task never ends. The value of an action and state should take into account all future rewards.
   1. Expected return
      $$G_t := \sum_{k=1}^{\infty} \gamma^{k-1}R_{t+k}$$
      where $\gamma$ is the *discount rate*.
   1. Recursive expected return

### 1.2.3. Discussion

Question: The reward hypothesis states “that all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).” Can you think of a situation that is **not** well-modeled by maximizing a scalar reward signal?

> I think multiple-goal optimization problems are not well modeled by scalar reward signal. Consider this simple example: A computer needs to *learn how to distribute execution time among users*. Now if we simply assign each action a reward, where an action means to excute some user's task, this policy will result in some user's task not excuted at all, because the computer doesn't distinguish between different users. We can, however, change our reward and goal to some complicated functions that takes into account user's waiting time. Actually this is a typical problem in operating systems designs. We can do pretty well, but certainly not with single goal, or single scalar reward, because when our priorities changes, it'll be hard to change our algorithm accordingly.

### 1.2.4. Peer graded assignments

Neo's MDP examples

1. Consider a government funding problem. If we would like an agent to decide annual invest in each institute across the country, then this is an MDP.  The environment is the unity of all institutes, each competing to win more funding. The actions are each year's investment in each institute, which is a vector. The states are each institutes's personnel, hardware and software resources, and projects started by this institute. The reward is a research index, which is calculated from every year's publishment and prizes gained by each institute. This is a finite MDP, and also a continuing task, as long as the countrie holds is sovereignty.
1. A **superconducting quantum chip** is a chip used for quantum computing. We feed a specific form of microwave pulse into the chip, then the quantum state of the qubit (quantum analogy of "bit" in classical computer) evolves accordingly, the process of which is called a **quantum operation**. But there might be some errors and noises due to the fragility of quantum states. Deciding the waveform microwaves is an MDP. The actions are waveforms each time we feed into the chip. The states are the quantum states after each operation. The reward is the difference between the resulted quantum state and the desired quantum state. Our goal here is to do a series of quantum operations that are correct and noise-insensitive as much as possible.
1. Consider our whole life. We have many decisions to make, small to big, daily decisions to annual decisions. We would like to maximize the "benefit" we get through our whole lives. The actions are every small or big decisions we make, e.g. what to have for breakfast, whom to marry, how many children do we raise. The states are the comprehensive situations we put ourselves in after each decision we make. The reward after each decision, is a complicated, custimized to everyone, changing over time, **living condition function** of many indicators of our lives, such as salary, savings, happiness, health condition, reputation, knowledges etc. So for each individual, it is the key that we find our own *living condition function* to know what we want, and the *dynamics* of the MDP, which is the probability of a state and reward after taking an action in current situation.
