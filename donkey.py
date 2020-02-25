import os
import gym
import gym_donkeycar
import numpy as np
import skimage
from DFP import DFPAgent
import skimage.io
import cv2
import functions as f
import math
import time

imgShape = (100, 50)
def modImg(img):
    img = skimage.color.rgb2gray(img)
    img = f.splitImage(img, 0.4)
    img = cv2.resize(img, imgShape)
    img = f.normalize(img)
    return img.reshape((1, *(imgShape), 1))

def makeMes(info):
    deviation = round(math.ceil(10 - info['cte']**2)/10, 1)
    pos = round((10 *info['cte'])/50, 1)
    return deviation, pos

speed = 0.3
agent = DFPAgent(3, (*imgShape,4), (3,), (18,))

os.environ['DONKEY_SIM_PATH'] = "/home/walker/Programs/DonkeySimLinux/donkey_sim.x86_64"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0) # "1" is headless


env = gym.make("donkey-generated-roads-v0")
for episode in range(10000):
    img = modImg(env.reset())
    state = np.stack([img]*4, axis=2)
    state = state.reshape((1, *imgShape, 4))
    mes = np.array([1, 0, 0])
    mes = mes.reshape((1,3))
    goal = np.array([1, 0, -1]*6)
    t = 0
    done = False
    tm = time.perf_counter()
    while not done:
        t += 1
        if (t%24 == 0):
            
            print("24 time: ", time.perf_counter() - tm)
            tm = time.perf_counter()
        crash = 0
        action = agent.act(state, mes, goal)
        step = [agent.actionToTurn(action), 0.3]
        img, reward, done, info = env.step(step)
        img = modImg(img)
        state = np.append(img, state[:,:,:,:3], axis=3)
        deviation, pos = makeMes(info)
        if done:
            crash = 10
        mes = np.array([deviation, pos, crash])
        #print(mes)
        mes = mes.reshape((1, 3))
        agent.remember(state, mes, action, done)
        agent.train(goal)
    print("Episode :", episode)
    agent.info()
    agent.save("Pretrained.h5")
    if (episode%5 == 0):
        agent.decayLearningRate()
