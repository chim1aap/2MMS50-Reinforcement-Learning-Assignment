import gym
import numpy
import random
from tqdm import tqdm




env = gym.make("Centipede-ram-v0")
env = gym.make("Centipede-v0") #image version.

env = gym.make("Breakout-v0")
env.reset()


alpha = 0.1
gamma = 1
epsilon = 0.1

n = 10000
prevAction = 0
prevState = 0
states = []
actions = env.action_space
## First step
act = actions.sample()
observation, reward, done, info = env.step(act )
prevAction = act
prevOvservation = observation
## Suppose Q only takes the action in consideration.
Q = [1]*env.action_space.n
for t in range(n):
    # select according to Epsilon Greedy.
    if epsilon < random.uniform(0, 1):
        # select at random
        act = actions.sample()
    else:
        # select greedy.
        act = numpy.argmax(Q)  # todo


    # Take Action, R=reward, S'=observation
    observation, reward, done, info = env.step(act )
    # Q(S,a) <- Q(S,A) + \alpha ( R + \gamma Q(S', A' ) - Q(S,A) )
    Q[act] = Q[act] + alpha *( reward + gamma * Q[prevAction] - Q[act])
