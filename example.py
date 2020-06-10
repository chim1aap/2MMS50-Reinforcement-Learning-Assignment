import gym

# Choose an environment from https :// gym . openai . com / envs /# atari
env = gym.make( "Centipede-ram-v0" )


print(env.action_space)
print(env.observation_space)

print("Press Enter to Continue...")
input()

env.reset()

#Run the Game
for t in range(1000):
    env.render()

    #Choose Random Action.
    action = env.action_space.sample()

    #Take the Action, make an observation from the environment and obtain a reward.
    observation, reward, done, info = env.step(action)

    print ( " At time " ,t , " , we obtained reward " , reward , " , and observed : " )
    print ( observation )

    if done:
        print ( " Episode finished after {} timesteps " . format ( t +1) )
        break

env.close()
