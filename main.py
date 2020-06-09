import gym
import agents #imports other py file in this dir.

##### Global Variables #####
env = gym.make( "Centipede-ram-v0" )




print( agents.sarsa() + agents.q_learning() )
