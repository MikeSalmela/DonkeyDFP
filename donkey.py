import os
import gym
import gym_donkeycar
import numpy as np
from DFP import DFPAgent

speed = 0.3
                    #120, 180, 3
                    #3072
agent = DFPAgent(7, (60, 160,3), (2,), (2,6))

os.environ['DONKEY_SIM_PATH'] = "/home/mike/Programs/DonkeySimLinux/donkey_sim.x86_64"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0) # "1" is headless

env = gym.make("donkey-generated-roads-v0")
for episode in range(50000):
    obs = env.reset()
    mes = np.array([1,1])

    goal = np.array([[1,1,1,1,1,1],[1,1,1,1,1,1]])
    done = False
    obs, mes = agent.reshape(obs, mes, reset=True)
    while not done:
        action = agent.act(obs, mes, goal)
        step = [agent.actionToTurn(action), 0.3]
        obs, reward, done, info = env.step(step)
        deviation = (10 - abs(info['cte']))/10
        if done:
            reward -= 3
            deviation -= 3
        mes = np.array([reward, deviation])
        obs, mes = agent.reshape(obs, mes)
        agent.remember(obs, mes, action, done)
        agent.train(goal)
    agent.info()
    agent.save("Pretrained.h5")
