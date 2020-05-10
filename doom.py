import os
import gym
import numpy as np
import vizdoom
from DFP import DFPAgent
import cv2
import skimage
import matplotlib.pyplot as plt

imgShape = (84, 84)

def modImg(img):
    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, imgShape)
    #print(img)
    #cv2.imwrite("test.png", np.asarray(img)*255)
    #return np.reshape(img , (1, *img.shape))
    return skimage.color.rgb2gray(img)

    #img = skimage.color.rgb2gray(img)
    #img = np.array(img)
    #img = cv2.resize(img, imgShape) 
    #return img

def append(x, a):
    x = np.reshape(x, (1, *imgShape, 1))
    a = np.reshape(a, (1, *imgShape, 3))
    return np.append(x, a[:, :, :, :2], axis=3)

def addImg(s, x):
    x = modImg(x)
    s = append(x, s)
    return s 

# [left, right, shoot]
def actionToInput(action):
    #action = np.uint8(action) 
    #a = np.unpackbits(action, bitorder='little', count=3)
    a = np.zeros(3)
    a[action] = 1
    return a.tolist()

game = vizdoom.DoomGame()
game.load_config("health_gathering.cfg")
game.set_sound_enabled(True)
game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
game.set_window_visible(True)
game.init()

agent = DFPAgent(3, (*imgShape, 3), (3,), (3*6,))
lifes = []
m_steps = 0
medkits = []
try:
    for episode in range(5000):
        lifetime = 0
        game.new_episode()
        done = False
        state = game.get_state()
        health = state.game_variables[0]
        medkit = 0
        poison = 0
        s = modImg(state.screen_buffer)
        s = np.stack([s]*3)
        s = addImg(s, state.screen_buffer) 
        m = np.array([health/30.0, medkit/10.0, poison])
        g = np.array([1.0, 1.0, -1.0]*6)
        m = m.reshape((1,3))

        while not done:
            m_steps += 1
            lifetime += 1
            action = agent.act(s, m, g)
            game.set_action(actionToInput(action))
            game.advance_action(4)
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
            m = np.array([health/30.0, medkit/10.0, poison])
            m = m.reshape((1,3))
            agent.remember(s, m, action, done, g)
            loss = agent.train()
        medkits.append(medkit)
        lifes.append(lifetime)
        agent.info()
        print(f"Episode: {episode}")
        print(f"Loss: {loss}")
        print(m_steps)
        if episode == 200:
            agent.decayLearningRate()

    a = np.asarray(medkits)
    np.savetxt("medkits.csv", a, delimiter=",")
    b = np.asarray(lifes)
    np.savetxt("doom.csv", b, delimiter=",")

except:
    plt.plot(lifes)
    plt.savefig('life_length.pdf')

    a = np.asarray(medkits)
    np.savetxt("medkits.csv", a, delimiter=",")
    a = np.asarray(lifes)
    np.savetxt("doom.csv", a, delimiter=",")
