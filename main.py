import gym
import numpy
import random


# beunen

def td ():
    # init game:
    env = gym.make("Centipede-ram-v0")
    env.reset()
    # numpy
    actions = env.action_space
    # init
    reward = 0
    lastaction = 0
    observation = 0
    n = 1000
    epsilon = 0.1
    vS = [env.action_space.n]*env.action_space.n
    S = [0]*env.action_space.n
    print(S)
    alpha = 0.3
    gamma = 0.3
    vsPrime = 0
    # Run the game
    for t in range(n):
        # select according to Epsilon Greedy.
        if epsilon < random.uniform(0, 1):
            # select at random
            act = actions.sample()
        else:
            # select greedy.
            act = numpy.argmax(vS)  # todo

        # Take Action, R=reward, S'=observation
        observation, reward, done, info = env.step(act )


        loc = 0
        # V(S) <- V(S) + \alpha ( R + \gamma V(S') - V(S) )
        vS[loc] = vS[loc] + alpha * (reward + gamma * vsPrime - S[act])
        vsPrime = vS[loc]
        print(vS)

td()
