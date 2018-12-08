import gym
import numpy as np
"""Evaluate agent's performance after Q-learning"""

env = gym.make("Taxi-v2").env
q_table = np.load('qtable.npy')
total_epochs, total_penalties = 0, 0
MaxNrEpochs=1000 #Wait this long until stopping the episode (in case the agent is unsuccessful)

episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done and epochs <= MaxNrEpochs:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward <= -2:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

