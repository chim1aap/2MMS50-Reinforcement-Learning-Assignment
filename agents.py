import gym
import numpy
import random


# TODO:
class Agents:
    def __init__(self,
                 actions: gym.Space,
                 alpha: float,
                 gamma: float,
                 epsilon: float
                 ):

        self.S = []  # Array of encountered States.
        self.vS = []  # V(s) estimated award, given state S.
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    # See p 142
    def td_learning(self, observation, action, reward):
        # Sprime =  self.S.contains(observation) # observation is S'

        # check if a state was found previously:
        if not observation in self.S:
            self.S.append(observation)
            self.vS.append(reward)  # initial value: 1

        # location of observation in S:
        loc = self.S.index(observation)

        Sprime = self.vS[action]

        # V(S) <- V(S) + \alpha ( R + \gamma V(S') - V(S) )
        self.vS[loc] = self.vS[loc] + self.alpha * (reward + self.gamma * Sprime - self.S[action])

        # select according to Epsilon Greedy.
        if self.epsilon < random.uniform(0, 1):
            # select at random
            act = self.actions.sample()
        else:
            # select greedy.
            act = numpy.argmax(self.vS)  # todo
        return (act)

    # TODO
    def q_learning(self):
        action = "bla"
        return (action)

    # TODO
    def sarsa(self):
        return (1)
