import gym
import numpy as np
import random
from IPython.display import clear_output

# Init Taxi-V2 Env
env = gym.make("Taxi-v2").env

# Init arbitary values
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.5
minepsilon = 0.5
NrEpisodes = 30000


all_epochs = []
all_penalties = []
print('Start training.')

for i in range(1, NrEpisodes+1):
    state = env.reset()

    # Init Vars
    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        epsilon = max(minepsilon,(1-1e-4)*epsilon)
        if random.uniform(0, 1) < epsilon:
            # Check the action space
            action = env.action_space.sample()
        else:
            # Check the learned values
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Update the new value
        new_value = (1 - alpha) * old_value + alpha * \
            (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward <= -2:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 10000 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")
print("Testing")
for i in range(100):
    action = np.argmax(q_table[state])

    next_state, reward, done, info = env.step(action)
    state = next_state
    env.render()
    print(f'action {action} resulted in reward {reward}')
    print('--------------------------------------------')
    if done:
        print('done. Starting new environment')
        state = env.reset()
np.save('qtable',q_table)
print('saved qtable.')
print(q_table.shape)
print(np.around(q_table,1))
print(q_table)
