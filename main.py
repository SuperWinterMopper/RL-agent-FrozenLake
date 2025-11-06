import gymnasium as gym
import numpy as np
import time
import sys

env = gym.make('FrozenLake-v1')
STATES = env.observation_space.n 
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))

print(Q)

epochs = 2000
max_steps = 100

alpha = .80
discount_rate = .9
epsilon = .9

rewards = []
for epoch in range(epochs):
    state = env.reset()[0]
    # print(state)
    # sys.exit()
    for step in range(max_steps):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        
        next_state, reward, done, _, _ = env.step(action)
        Q[state, action] = Q[state, action] + alpha * (float(reward) + discount_rate * np.argmax(Q[next_state, :]) - Q[state, action])
        state = next_state

        if done:
            state = next_state
            rewards.append(reward)
            epsilon -= .001
            break
print(Q)


print("DONE")