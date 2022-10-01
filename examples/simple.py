import gym
import adversarial_gym

# env = gym.make("Chess-v0", render_mode='human')
env = gym.make("TicTacToe-v0", render_mode='human')
print('reset')
env.reset()
terminal = False
while not terminal:
    action = env.action_space.sample()
    observation, reward, terminal, truncated, info = env.step(action)
env.close()