
import gym
import tqdm
import numpy as np
from plotnine import *  # like Tidyverse in R.
import pandas


def simulateGame(n):
    env = gym.make("Centipede-ram-v0")
    env.reset()
    # numpy
    actions = []
    for t in range(n):
        # initialeze arrays
        actionList = np.empty(n)
        rewardList = np.empty(n)

        # select action:
        actionList[t] = env.action_space.sample()

        # observation, reward, done, info = env.step(action)
        observation, rewardList[t], done, info = env.step(int(actionList[t] ))
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
