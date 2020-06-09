import gym
import time
from tqdm import tqdm # This makes a nice progress bar.
# Choose an environment from https :// gym . openai . com / envs /# atari
env = gym.make( "Centipede-ram-v0" )

print(env.action_space)
print(env.observation_space)

#

env.reset() #Resets the game

def run():
    #Run the Game
    for t in tqdm(range(1000)) :
        time.sleep(1/120) #slows so you see what is happening.
        env.render() # draws the game state

        #Choose Random Action.
        #This is where we choose a certain action.
        action = env.action_space.sample()

        #Take the Action, make an observation from the environment and obtain a reward.
        # observation : object
        # reward: float
        # done: boolean
        # info: dict, debugging infromation.

        observation, reward, done, info = env.step(action)

        # Example time - this freezes the game however...
        # if t == 684 & False:
        #     print ( " At time " ,t , " , we obtained reward " , reward , " , and observed : " )
        #     print ( observation )
        #     print("Press Enter to Continue...")
        #     input()

        if done:
            print ( " Episode finished after {} timesteps " . format ( t +1) )
            break

    env.close()


if __name__ == '__main__':
    # stuff to run if *this* file is called.
    run()
