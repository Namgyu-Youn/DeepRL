- [docs](https://gymnasium.farama.org/), [GitHub](https://github.com/Farama-Foundation/Gymnasium/) #DeepRL
# Gymnasium Env
``` python
import gymnasium as gym

# First, we create env(Agent) called LunarLander-v2
env = gym.make("LunarLander-v2")

# Then we reset this environment
observation, info = env.reset()

for _ in range(20):
	# All actions should be contained within the space
	action = env.action_space.sample()
	print("Action taken:", action)

	# Update the state using action
	observation, reward, terminated, truncated, info = env.step(action)
	
	# Agent reaches the terminal state(ex. finish, crash)
	if terminated or truncated:
		# Reset the environment
		print("Environment is reset")
		observation, info = env.reset()

env.close()
```
- `step()` : Updates an `env` with actions returning the next agent **observation**
- `terminated` : It means agent reached to the finish or crash point

## Make and register
- It allows automatically load environments, pre-wrapped with several important wrappers