import os
import gym
import gym_donkeycar
import numpy as np
from DFP import DFPAgent

speed = 0.3
                    #3072
agent = DFPAgent(7, (3072,), (2,), (2,), "donkeytrack.h5", True)

os.environ['DONKEY_SIM_PATH'] = "/home/mike/Programs/DonkeySimLinux/donkey_sim.x86_64"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0) # "1" is headless

env = gym.make("donkey-generated-roads-v0")
for episode in range(50000):
    obs = env.reset()
    mes = np.array([1,1])

    goal = np.array([1, 1])
    done = False
    obs, mes = agent.reshape(obs, mes, reset=True) 
    while not done:
        action = agent.act(obs, mes, goal)
        step = [agent.actionToTurn(action), 0.3] 
        obs, reward, done, info = env.step(step)
        deviation = 10 - abs(info['cte'])
        if done:
            reward -= 30
            deviation -= 30
        mes = np.array([reward, deviation])
        obs, mes = agent.reshape(obs, mes)       
        agent.remember(obs, mes, action, done)
        agent.train(goal)

    print(f"Epsilon: {agent.epsilon}")
    print(f"Mem size: {agent.memory.getSize()}")

    agent.save("Pretrained.h5")
