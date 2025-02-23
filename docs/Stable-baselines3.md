#DeepRL
## A2C (Advantage Actor-Critic)
- Policy-based RL algorithm using `Actor-Critic` structure. It trains multiple env and update by synchronization.
- `Actor` : Policy function **which action to decide**
- `Critic` : **Evaluation function** for current state
- `make_vec_env` : Utility function for multiple env (parallel action)
- `model.predict` : Agent choose the `action`. `Exploration` can be adjusted.
```python
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Build parallel environments
env = make_vec_env('CartPole-v1', n_envs=4)

# Build & Train : A2C model
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=25000) # Update policy

# Model prediction
obs = env.reset()
for _ in range(1000):
	# deterministic : Choose best choice without exploration
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```
- Cons :
	- Unstable learning : Policy update occurs too often, so it could be unstable. (Past policy could be useful)
	- Sample's inefficiency : Collected data is **discarded after only one step**.

## PPO (Proximal Policy Optimization)
- It combines `A2C` (having multiple workers) and `TRPO` (it uses a trust region to improve the actor -> **stable**).
- It aims to "the new (updated) policy should be not too far from the old (past) policy" and
### Clipping mechanism
- Update policy with two approaches:
	- Safety : It uses `Probability Ratio (rt)`. It limits ratio for steep policy change.
	- Efficiency : `TRPO` uses Second-Order Differential Equation. `PPO` uses min/max.
$$
Probability \ Ratio (rt) : rt(i) = π_{i}(at|st) / π_{i-1}(at|st)
$$
```python
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# n_envs allows Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

# Build & Train : MlpPolicy model
model = PPO("MlpPolicy", vec_env)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

model = PPO.load("ppo_cartpole")

# Model prediction
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
```

## Evaluation : evaluate_policy
- `env`, `model` can be **customized** (It means various comparison can be experienced).
```python
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import gymnasium as gym

# Build env & model(algorithm)
env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env)

# Model evaluation
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
```