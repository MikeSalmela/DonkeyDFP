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
from keras.models import load_model

imgShape = (48, 128, 3)
encoded = 3072
encoder = load_model("donkeygreen.h5")

def modImg(img):
    img = f.splitImage(img)
    img = cv2.resize(img, (128, 48), interpolation=cv2.INTER_CUBIC)
    img = f.normalize(img)
    img = encoder.predict(np.array([img]))[0]
    return img.reshape((1, encoded, 1))

def makeMes(info):
    deviation = min(round((info['cte']**2)/10, 1),3)
    speed = round(info['speed']/10, 1)
    return deviation, speed
imgFrames = 4
speed = 0.3
f_vec = [1, 2, 4, 8, 16, 32]
l = len(f_vec)
mes_c = 4
agent = DFPAgent(3, (encoded, imgFrames), (mes_c,), (mes_c*l,), f_vec, True)

os.environ['DONKEY_SIM_PATH'] = "/home/walker/Programs/DonkeySimLinux/donkey_sim.x86_64"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0) # "1" is headless
steps = 0
avrgsteps = []
env = gym.make("donkey-generated-track-v0")
#try:
for episode in range(1500):
    img = modImg(env.reset())
    state = np.stack([img]*imgFrames, axis=2)
    state = state.reshape((1, encoded, imgFrames))
    #mes = deviation, reward, crash, speed
    mes = np.array([0, 1, 0, 0])
    mes = mes.reshape((1,mes_c))
    goal = np.array([-1, 0.8, -10, 0.5]*l)
    t = 0
    done = False
    tm = time.perf_counter()
    while not done:
        steps += 1
        t += 1
        if (t%f_vec[-1] == 0):
            print(f_vec[-1], " time: ", time.perf_counter() - tm)
            tm = time.perf_counter()
        if (t%2000):
            done = True

        crash = 0
        action = agent.act(state, mes, goal)
        turn, speed = agent.actionToTurn(action)
        step = [turn, speed]
        img, reward, done, info = env.step(step)
        img = modImg(img)
        state = np.append(img, state[:,:,:imgFrames-1], axis=2)
        deviation, speed = makeMes(info)
        if done:
            crash = 1
        mes = np.array([deviation, reward, crash, speed])
        mes = mes.reshape((1, mes_c))
        agent.remember(state, mes, action, done, goal)

    print("Episode :", episode)
    agent.info()
    env.reset()
    avrgsteps.append(int(steps))
    steps = 0
    if (episode%20 == 0):
        agent.save("Pretrained.h5")
    if (episode == 150):
        agent.decayLearningRate()

agent.save("pretrained_encoder_32.h5") 
plt.plot(avrgsteps)
plt.savefig("autoencoder_32.png")
a = np.asarray(avrgsteps)
np.savetxt("autoencoder_32.csv", a, delimiter=",")
plt.show()
"""
except:
    agent.save("pretrained_encoder_32.h5") 
    plt.plot(avrgsteps)
    plt.savefig("autoencoder_32.png")
    a = np.asarray(avrgsteps)
    np.savetxt("autoencoder_32.csv", a, delimiter=",")
    plt.show()
"""
