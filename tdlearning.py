import gym
import time
import numpy
import random
import os
import pickle
from tqdm import tqdm

# Choose an environment from https :// gym . openai . com / envs /# atari
env = gym.make("Centipede-ram-v0")
env = gym.make("SpaceInvaders-ram-v0")
env = gym.make("Breakout-v0")
maxX = 209
maxY = 159

### Q states.
# There are three parameters: player x, ball x, ball y. And the Action
# print(env.observation_space.shape[0], env.observation_space.shape[0], env.observation_space.shape[1],env.action_space.n)
Q = numpy.ones(shape=(
    env.observation_space.shape[0], env.observation_space.shape[0], env.observation_space.shape[1]))
for i in range(len(Q)):
    for j in range(len(Q[0, 0])):
        Q[i, maxX, maxY] = 0
# print(Q)
# width , height = env.observation_space[1], env.observation_space[2]
width, height = 210, 160

Qstart = numpy.copy(Q)

checkpointdistance = 100
def monteCarlo():
    global Q
    rewards = []
    k = 0  # iteration to load from .

    # load Q from file
    try:
        Q = numpy.load("./tdlearning/save" + str(k) + ".npy")

        # load rewards from file
        with open("./tdlearning/rewards"+str(k) +".pkl", 'rb') as fp:
            print(fp)
            rewards = pickle.load(fp)
        print(len(rewards))
        print(sum(rewards))

        # Sanity check
        print(sum(sum(sum(sum(Q == Qstart)))))
        print(210 * 210 * 160 * 4)
        print(Q.shape)
    except FileNotFoundError as e:
        print(e)
        k = 0
    # rewardvectors and recording stuff:

    # Begin looping and training.
    for i in tqdm(range(k, k +1 + 10000) , desc= "Td: "):
        Q, totalReward = run(Q)

        rewards.append(totalReward)
        # Save results in case the PC crashes *again*, after every 10 games.
        if i % checkpointdistance == 0:
            # store Q
            numpy.save("./tdlearning/save" + str(i) + ".npy", Q)
            # store rewardvector.
            with open("./tdlearning/rewards" + str(i) + ".pkl", 'wb') as fp:
                pickle.dump(rewards, fp)

            # remove old ones because they are 200Mb in size.
            try:
                os.remove("./tdlearning/save" + str(i - 5 * checkpointdistance) + ".npy")
                os.remove("./tdlearning/rewards" + str(i - 5 * checkpointdistance) + ".pkl")
            except FileNotFoundError as e:  # ignore errors and continue
                # pass # do nothing
                print(e)
    '''
    Save stuff after simulations - just in case. 
    '''
    numpy.save("./tdlearning/saveEnd" + str(i) + ".npy", Q)
    with open("./tdlearning/rewards" + str(i) + ".pkl", 'wb') as fp:
        pickle.dump(rewards, fp)


'''
Runs a game 
'''


def run(Q):
    # initialize
    env.reset()
    xMinBall, xMaxBall = 0, maxX
    yMinBall, yMaxBall = 0, maxY

    n = 100
    tBallSearch = n
    searchTimePaused = 10
    state = 0
    Qold = 1
    alpha = 0.7
    gamma = 0.7
    epsilon = 0.1
    totalReward = 0
    # Run the Game
    for t in (range(n)):
        # time.sleep(1 / 30)
        # env.render()

        # Choose Random Action.
        if random.uniform(0, 1) < epsilon:  # t bigger than 2 else Qarray is not initialized.
            action = env.action_space.sample()
        elif t < 31:
            action = env.action_space.sample()
        else:
            Qarray = Q[xmin, xMinBall, yMinBall]  # array of the rewards of the actions in this particular state.
            action = numpy.argmax(Qarray)

        # Take the Action, make an observation from the environment and obtain a reward.
        observation, reward, done, info = env.step(action)

        # There are three states, This is to speed up the scanning
        # If the ball is lost, it can spawn everywhere, so we need to check *everywhere*
        # It also takes a while before it spawns, so we skip some of the scans to speed it up.
        # The downside is that it is going to react slower when a ball spawns.
        # There are three states: Ball in game (2) -> Ball out and pauze (0) -> Ball out and resume (1) .
        if state == 0:  # Ball out and pause
            tBallSearch = t + searchTimePaused  # search for the ball after some time steps.
            state = 1

        if state == 1:  # Ball out and resume
            if tBallSearch < t:
                state = 2
        if state == 2:  # Ball in the game.
            xMinBall, xMaxBall, yMinBall, yMaxBall = ballmove(xMinBall, xMaxBall, yMinBall, observation)
            if abs(xMaxBall - xMinBall) > 10:
                state = 0

        '''
        update the player position
        It is always at a certain y position, so we only need to update x.
        '''
        xmin, xmax = playermove(observation)

        '''
        Update the Q function (tdlearning)
        These lines should be changed to have other policies (Qlearning, Sarsa)
        '''
        # V(S) <- V(S) + \alpha ( R + \gamma V(S') - V(S) )
        # S,a = (xPlayer, xBall, yBall, action )
        Q[xmin][xMinBall][yMinBall] = Q[xmin, xMinBall, yMinBall] + alpha * (
                    reward + gamma * Qold - Q[xmin, xMinBall, yMinBall])
        Qold = Q[xmin, xMinBall, yMinBall]

        '''
        Logging
        '''
        totalReward = totalReward + reward



        if done:
            # print(" Episode finished after {} timesteps ".format(t + 1))
            break

    env.close()
    return (Q, totalReward)


def ballmove(x, xmax, y, observation):
    if abs(x - xmax) > 5:  # ball gone: search entire board.
        xMaxBall, xMinBall = 0, maxX
        yMaxBall, yMinBall = 0, maxY
        # for x in range(max ( 0, xlast -searchRange ), min(xlast + searchRange, width) ):
        for x in range(0, width):
            if x in range(57, 63):  # die rode balk
                continue  # go to next loop
            if x in range(189, 195):  # Player area
                continue
            # for y in range(max( 0,ylast - searchRange  ) , min( ylast + searchRange, height-1 ) ) :
            for y in range(height):
                if (observation[x, y] == [200, 72, 72]).all():
                    xMaxBall = max(x, xMaxBall)
                    xMinBall = min(x, xMinBall)
                    yMaxBall = max(y, yMaxBall)
                    yMinBall = min(y, yMinBall)
    else:  # ball is near previous position. - search just that small area
        searchRange = 5
        xlast = x
        ylast = y
        xMaxBall, xMinBall = 0, maxX
        yMaxBall, yMinBall = 0, maxY
        for x in range(xlast - searchRange, min(xlast + searchRange, 190)):
            if x in range(57, 63):  # die rode balk
                continue  # go to next loop
            if x in range(189, 195):  # Player area
                continue
            if x in range(-110, 7):  # left of the board
                continue
            # for y in range(height):
            for y in range(ylast - searchRange, ylast + searchRange):
                if y in range(200, 300):
                    continue
                if (observation[x, y] == [200, 72, 72]).all():
                    xMaxBall = max(x, xMaxBall)
                    xMinBall = min(x, xMinBall)
                    yMaxBall = max(y, yMaxBall)
                    yMinBall = min(y, yMinBall)
    # print(xMinBall, xMaxBall, yMinBall, yMaxBall)
    return xMinBall, xMaxBall, yMinBall, yMaxBall


def playermove(observation):
    ymax, ymin = 0, 210
    x = 190
    # for x in range(max(0, xlast -searchRange ) , min(  xlast + searchRange, width  ) ):
    for y in range(8, 151):
        if (observation[x, y] == [200, 72, 72]).all():
            ymax = max(y, ymax)
            ymin = min(y, ymin)
    return ymin, ymax


'''
prints a observation. Also pauses the game. 
'''


def print_observation(observation):
    for x in range(len(observation)):
        for y in range(len(observation[x])):
            if (observation[x, y] == [200, 72, 72]).all():
                print(observation[x, y], "at ", x, y)
    input()


# grey = [142 , 142, 142]
# black = [0,0,0]
# lower green = [66,158,130]
# lower red = [200, 72, 72]
# player = [200, 72, 72] at (189-193, 8-151)
# ball = [200, 72, 72]
# enemies = De rest

monteCarlo()
