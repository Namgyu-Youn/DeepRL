# Introduction to Deep Reinforcement Learning
## What is RL (Reinforcement Learning)?
- RL : Framework for solving control tasks (also called decision problems) by building agents that **learn from the environment**t by **interacting** with it through trial and error and **receiving rewards** (positive or negative) as unique feedback.
- As a result, **without any supervision**, agents will get better and better at upcoming trial.
- This RL loop outputs a sequence of **state(S), action(A), reward(R) and next state.**
	![|600](https://i.imgur.com/jusbTl8.png)

## MDP (Markov Decision Property)
- MP implies that our agent needs **only the current state to decide** what action to take and **not the history of all the states and actions** they took before.

## Observations & Action Space
- Observation : **Partial** information our agent gets **from the environment**
- state(S) : **Complete description of the state of the world** (no hidden information) -> Fully observed environment (ex. In a chess game, whole check board information should be environment)
- If **the number of possible actions** is finite/infinite, it is discrete/continuous space. (ex. Super Mario/Self driving)

## Rewards and the Discounting
- `Reward` is **the only feedback** for the agent. And agent knows if the action taken was good or not by reward
- The `cumulative reward` can be written as :
$$
R(t) = r(t+1) + r(t+2) + r(t+3) + \dots = \sum^{\infty}_{k=0} r(t)+k+1
$$
- However, in reality, **we can’t just add them like that.** **The cumulative reward only means short-term future**, so long-term future reward is needed. -> discount factor = `Gamma`
- If `gamma` close to 1, it encourages the agent to consider long-term reward, while 0 encourages short-term rewards. After all, `discounted expected cumulative reward` is :
$$
R(t) = r(t+1) + \gamma*r(t+2) + \gamma^2*r(t+3) + \gamma^3*r(t+4) + \dots
$$

##  Episodic and the Continuing Tasks
- Task is an instance of RL problem
- `Episodic task` : Terminal state (starting and ending point) exist. This creates a episode : S, A, R
- `Continuing task` : No terminal state -> continue forever

## The Exploration/Exploitation trade-off
- `Exploration` : Trying new actions to discover potentially better options. **While it might result in lower immediate rewards**, it opens up possibilities for **discovering better strategies in the long run**.
- `Exploitation` : Selecting the best action based on **currently available information**.
- `Trade-off` : Balancing these two strategies for best choice.
	- Pure exploitation can lead to getting stuck in **local optima**, missing out on potentially better solutions elsewhere
	- Pure exploration, while discovering good strategies, **fails to capitalize on the knowledge gained**.

## Two main approaches for solving RL problems
- `Policy(π)` : It **defines agent's action at a state (given time).**
- `π(a|s)` :  The probability of selecting `a(Action)` in `s(State)`
- Approaches to find the optimal `Policy(π) :
	- `Policy-based` (ex. `REINFORCE`) : Teach agent **which action to take** by defining mapping function.
		![[../9. AttachedFiles/Pasted image 20250223191904.png|400]]
	- `Value-based` (ex. `Q-learning, DQN`): Teach agent **which state is more valuable**. "Act according to our policy" just means that our policy is **"going to the state with the highest value"**.
		![[../9. AttachedFiles/Pasted image 20250223192345.png|400]]