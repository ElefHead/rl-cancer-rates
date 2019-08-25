# Policy : Simulator

This branch contains code for a completely simulated environment that tries to
somewhat model the main QuBBD problem statement using random variables and vectors.

## Methodology
The functioning of the environment can be summarized as:
1. Assume that there are A different actions/decisions, and one state is a D dimension vector.
2. For each action _a_, sample a diagonal (D x D) matrix W<sub>_a_</sub> and a diagonal matrix
    Sigma<sub>_a_</sub> such that _P_(s<sub>t+1</sub> - s<sub>t</sub> | s<sub>t</sub>, a<sub>t</sub>) is
    a Gaussian distribution with mean W<sub>a</sub>*s<sub>t</sub> and covariance Sigma<sub>a</sub>.
    > This just means we model the environment as a Gaussian and each next state depends on a small multinomial noise by the use of action chosen on current state.
3. The reward and episode ends right now are calculated as a sigmoid of the dot product of a random vector v and the current state. The function then
    has a threshold for "death" and "recovery".

## Using the Policy
To use the policy:
1. Import the class Policy from policy.py
```python
    from simulator.policy import Policy
```
2. Create the environment as
```python
    env = Policy()
```
3. Get the initial state or reset environment as
```python
    state = env.reset()
```
4. To take a step
```python
   new_state, current_reward, done, message = env.step()
```

See code comments for function descriptions

---
The attempt using this code is to see if we can model the QuBBD data problem and how well deep-RL techniques perform in this setting.
