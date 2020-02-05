import os
import gym
import numpy as np
import vizdoom
from DFP import DFPAgent
import cv2
import skimage
import matplotlib.pyplot as plt

imgShape = (80, 80)

def modImg(img):
    img = skimage.color.rgb2gray(img)
    img = np.array(img)
    img = cv2.resize(img, imgShape) 
    return img

def append(x, a):
    x = np.reshape(x, (1, *imgShape, 1))
    a = np.reshape(a, (1, *imgShape, 4))
    return np.append(x, a[:, :, :, :3], axis=3)

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
game.load_config("health_gathering_supreme.cfg")
game.set_sound_enabled(True)
game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
game.set_window_visible(True)
game.init()

agent = DFPAgent(3, (80, 80, 4), (3,), (3,6))
lifes = []
#try:
for episode in range(50000):
    lifetime = 0
    game.new_episode()
    done = False
    state = game.get_state()
    health = state.game_variables[0]
    medkit = 0
    poison = 0
    img = modImg(state.screen_buffer)
    s = np.stack([img]*4)
    s = addImg(s, state.screen_buffer) 
    m = np.array([health/30.0, medkit/10.0, poison])
    g = np.array([1.0, 1.0, -1.0])
    g = np.repeat(g,6)
    g = g.reshape((3,6))
    s, m = agent.reshape(s, m)
    while not done:
        lifetime += 1
        action = agent.act(s, m, g)
        game.set_action(actionToInput(action))
        game.advance_action(4)
        done = game.is_episode_finished()
        agent.remember(s, m, action, done)
        if not done:
            state = game.get_state()
            if state.game_variables[0] - 2 > health:
                print("Medkit")
                medkit += 1
            if health - state.game_variables[0] > 8:
                print("Poison")
                poison += 1
            health = state.game_variables[0]
            s = addImg(s, state.screen_buffer)
        if done:
            health -= 10
            medkit -= 10
        m = np.array([health/30.0, medkit/10.0, poison])
        s, m = agent.reshape(s, m)
        #agent.remember(s, m, action, done)
        agent.train(g)
    lifes.append(lifetime)
    #agent.decayLearningRate()
    agent.info()
    print(f"Episode: {episode}")
    if episode % 20 == 0:
        agent.decayLearningRate()
        agent.save("Pretrained.h5")
#except:
#    plt.plot(lifes)
#    plt.savefig('life_length.pdf')
