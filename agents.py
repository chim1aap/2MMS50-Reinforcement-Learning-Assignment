import gym
import numpy
import random

#TODO:
class Agents:
    def __init__ (self,
                  actions: gym.Space,
                  alpha: float,
                  gamma: float,
                  epsilon: float
                  ):

        self.S = numpy.zeros(actions.n)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    # See p 142
    def td_learning(self, observation, action, reward ):
        Sprime =  self.S.contains(observation) # observation is S'

        # V(S) <- V(S) + \alpha ( R + \gamma V(S') - V(S) )
        self.S[action] = self.S[action] + self.alpha ( reward + self.gamma * Sprime - self.S[action]  )


        # select according to Epsilon Greedy.
        if self.epsilon < random.uniform(0,1) :
            #select at random
            act = action.sample()
        else:
            #select greedy.
            act = numpy.argmax(self.S)
        return(act)

    #TODO
    def q_learning():
        action = "bla"
        return(action)

    # TODO
    def sarsa():
        return(1)

