import os
import gym
import numpy as np
import vizdoom
from DFP import DFPAgent
import cv2
import skimage
import matplotlib.pyplot as plt
import functions as f

imgShape = (84, 84)

def modImg(img):
    #img = skimage.transform.resize(img, imgShape)
    #img = skimage.color.rgb2gray(img)
    #img = img.astype(np.float32)
    #return img.reshape((*imgShape,3))
    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, imgShape)
    img = skimage.color.rgb2gray(img)
    img = f.normalize(img)
    #print(img)
    #cv2.imwrite("test.png", np.asarray(img)*255)
    return np.reshape(img , (*img.shape, 1))

    #img = skimage.color.rgb2gray(img)
    #img = np.array(img)
    #img = cv2.resize(img, imgShape) 
    #return img

def append(x, a):
    x = np.reshape(x, (1, *imgShape, 1))
    a = np.reshape(a, (1, *imgShape, 2))
    return np.append(x, a[:, :, :, :1], axis=3)

def addImg(s, x):
    x = modImg(x)
    s = append(x, s)
    return s 

# [left, right, shoot]
def actionToInput(action):
    #print(action)
    action = np.uint8(action)
    a = np.unpackbits(action, bitorder='little', count=3)
    #print(a)
    return a.tolist()

game = vizdoom.DoomGame()
game.load_config("health_gathering_supreme.cfg")
game.set_sound_enabled(True)
game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
game.set_window_visible(True)
game.init()

agent = DFPAgent(6, (*imgShape, 2), (3,), (3*6,))
#agent.load("pretrained.h5")
lifes = []
health_v = []
m_steps = 0
medkits = []

try:
    for episode in range(2000000):
        lifetime = 0
        game.new_episode()
        done = False
        state = game.get_state()
        health = state.game_variables[0]
        medkit = 0
        poison = 0
        s = modImg(state.screen_buffer)
        s = np.stack([s]*2)
        s = addImg(s, state.screen_buffer) 
        m = np.array([health/30.0, medkit, poison])
        g = np.array([1.0, 1.0, -1.0]*6)
        m = m.reshape((1,3))

        while not done:
            m_steps += 1
            lifetime += 1
            action = agent.act(s, m, g)
            game.set_action(actionToInput(action))
            game.advance_action(4)
            if ((agent.trains+1) % 500 == 0):
                print("decayLearningRate")
                agent.decayLearningRate()
            done = game.is_episode_finished()
            if not done:
                state = game.get_state()
                if state.game_variables[0] - 2 > health:
                    print("Medkit")
                    medkit += 1
                if health - state.game_variables[0] > 8:
                    print("Poison")
                    poison += 1
                health = state.game_variables[0]
                #s = modImg(state.screen_buffer)
                s = addImg(s, state.screen_buffer)
            m = np.array([health/30.0, medkit, poison])
            m = m.reshape((1,3))
            agent.remember(s, m, action, done, g)
            if (m_steps % 32 == 0):
                loss = agent.train()
        medkits.append(medkit)
        lifes.append(lifetime)
        health_v.append(health)
        agent.info()
        print(f"Episode: {episode}")
        #print(f"Loss: {loss}")
        print(f"Trains: {agent.trains}")
        if (agent.trains*64 == 50000000):
            break


    agent.save("pretrained.h5")
    a = np.asarray(medkits)
    np.savetxt("medkits.csv", a, delimiter=",")
    b = np.asarray(lifes)
    np.savetxt("doom.csv", b, delimiter=",")

    c = np.asarray(health_v)
    np.savetxt("health.csv", c, delimiter=",")

except:
    plt.plot(lifes)
    plt.savefig('life_length.pdf')

    agent.save("pretrained.h5")
    a = np.asarray(medkits)
    np.savetxt("medkits.csv", a, delimiter=",")
    a = np.asarray(lifes)
    np.savetxt("doom.csv", a, delimiter=",")

    c = np.asarray(health_v)
    np.savetxt("health.csv", c, delimiter=",")
