
import gym
import tqdm
import numpy as np
from plotnine import *  # like Tidyverse in R.
import pandas
from  agents import Agents

def simulateGame(n):
    env = gym.make("Centipede-ram-v0")
    env.reset()
    # numpy
    actions = []
    agent = Agents(env.action_space , 0.5, 0.5, 0.1)

    # init
    reward = 0
    lastaction = 0
    observation = 0

    for t in range(n):
        # initialeze arrays
        actionList = np.empty(n)
        rewardList = np.empty(n)

        # select action:
        # actionList[t] = env.action_space.sample()
        actionList[t] = agent.td_learning(observation, lastaction, reward)

        # observation, reward, done, info = env.step(action)
        observation, rewardList[t], done, info = env.step(int(actionList[t] ))
        lastAction = actionList[t]
        if done :
            break
    env.close()
    return (actionList, rewardList)


if __name__ == '__main__':
    actions, rewards = simulateGame(10000)
    df = pandas.DataFrame({"actions": actions, "rewards": rewards, "Time": range(10000)})
    print(df)
    g = (ggplot(df)
         + geom_point(aes(x=rewards, y=actions))
         )
    g = (
        ggplot(df)
        + geom_point(aes(x = "rewards", y ="Time") )
    )
    print(g)
