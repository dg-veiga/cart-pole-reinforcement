# import gym
# print('aqui')
# env = gym.make('CartPole-v1')

import gym
import time
import numpy as np

env = gym.make('CartPole-v1')
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)

reward_sum = 0
tries = []
_try = []

# Training:
for _ in range(1000):
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action=action)
    
    env.render(mode='human')

    time.sleep(0.01)

    reward_sum += reward

    print(observation, action)

    if done:
        observation, info = env.reset(return_info=True)
        reward_sum = 0
        break

env.close()