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
import matplotlib.pyplot as plt

imgShape = (100, 50)
def modImg(img):
    img = skimage.color.rgb2gray(img)
    img = f.splitImage(img, 0.4)
    img = cv2.resize(img, imgShape)
    img = f.normalize(img)
    return img.reshape((1, *(imgShape), 1))

def makeMes(info):
    deviation = round(math.ceil(10 - info['cte']**2)/10, 1)
    deviation = deviation if deviation > 0 else 0
    pos = round((info['cte']*2)/10, 1)
    return deviation, pos
imgFrames = 4
speed = 0.3
f_vec = [1, 2, 4, 8, 16, 32]
l = len(f_vec)
agent = DFPAgent(3, (*imgShape,imgFrames), (3,), (3*l,), f_vec)

os.environ['DONKEY_SIM_PATH'] = "/home/walker/Programs/DonkeySimLinux/donkey_sim.x86_64"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0) # "1" is headless
steps = 0
avrgsteps = []

env = gym.make("donkey-generated-track-v0")
try:
    for episode in range(5000):
        img = modImg(env.reset())
        state = np.stack([img]*imgFrames, axis=2)
        state = state.reshape((1, *imgShape, imgFrames))
        mes = np.array([1, 0, 0])
        mes = mes.reshape((1,3))
        goal = np.array([1, 0, -1]*l)
        t = 0
        done = False
        tm = time.perf_counter()
        while not done:
            steps += 1
            t += 1
            if (t%f_vec[-1] == 0):
                
                print(f_vec[-1], " time: ", time.perf_counter() - tm)
                tm = time.perf_counter()
            crash = 0
            action = agent.act(state, mes, goal)
            step = [agent.actionToTurn(action), 0.3]
            img, reward, done, info = env.step(step)
            img = modImg(img)
            state = np.append(img, state[:,:,:,:imgFrames-1], axis=3)
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
        if (episode%10 == 0):
            avrgsteps.append(steps/5)
            steps = 0
        #if (episode%5 == 0):
            #agent.decayLearningRate()
    
    plt.plot(avrgsteps)
    plt.show()
    plt.savefig("average32.png")
    a = np.array(avrgsteps)
    np.savetxt("avg32.csv", a, delimiter=",")

except:
    plt.plot(avrgsteps)
    plt.show()
    plt.savefig("average32.png")
    a = np.array(avrgsteps)
    np.savetxt("avg32.csv", a, delimiter=",")


