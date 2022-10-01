import gym
import obm_gym
import time
import chess

# env = gym.make("Chess-v0", render_mode='human')

env = gym.make("TicTacToe-v0", render_mode='human')
print('reset')
env.reset()
terminal = False
while not terminal:
    action = env.action_space.sample()
    observation, reward, terminal, truncated, info = env.step(action)
time.sleep(1)

env.close()