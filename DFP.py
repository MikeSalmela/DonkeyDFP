import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.layers import BatchNormalization, Reshape, Subtract, Add
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import functions as f
from collections import deque
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import numpy as np
import random
import imageprocessor as ip

class Memory:
    def __init__(self, futureVec, mesCount, maxSize, stateShape):
        self.stateShape = stateShape
        self.timesteps = len(futureVec)
        self.futurePreds = futureVec
        self.mesCount = mesCount
        self.maxTimestep = futureVec[-1]
        self.mem = deque(maxlen=maxSize)

    def append(self, state, measurement, action, done):
        self.mem.append((state, measurement, action, done))

    def getSize(self):
        return len(self.mem)

    # Get random array of f_vector recieved from the according state and action
    def randomSample(self, count):
        states = np.zeros((count, *self.stateShape))
        measurements = np.zeros((count, self.mesCount))
        actions = np.zeros((count))
        f_vec = np.zeros((count, self.mesCount, self.timesteps))

        randomInds = np.random.choice(len(self.mem)-(self.maxTimestep+1), count)

        for i, ind in enumerate(randomInds):
            if self.mem[ind][3]:
                ind += 1
            isDone = False
            future = np.zeros((self.timesteps, self.mesCount))
            k = 0
            for j in range(self.maxTimestep+1):
                if not self.mem[ind+j][3] and (j in self.futurePreds) and not isDone:
                    f = self.mem[ind+j][1] - self.mem[ind][1]
                    future[k] = f
                    k += 1
                elif (j in self.futurePreds):
                    if not isDone:
                        offset = j
                    isDone = True
                    f = self.mem[ind+offset][1] - self.mem[ind][1]
                    future[k] = f
                    k += 1
            states[i]       = self.mem[ind][0]
            measurements[i] = self.mem[ind][1]
            actions[i]      = self.mem[ind][2]
            f_vec[i]        = future.reshape((self.mesCount, self.timesteps))

        return states, measurements, actions, f_vec


class DFPAgent:
    # To stack the Image input, it must be a 1d vector representation of the image.
    def __init__(self, num_actions, I_shape, M_shape, G_shape, \
            encoderName=None, stackI=False, stackSize = 4):
        # Possible pretrained encoder for Image data
        if encoderName != None:
            self.encoder = load_model(encoderName)
        else:
            self.encoder = None
        # If Imagedata is stacked to get some information of movement:
        self.stackI = stackI
        self.stackSize = stackSize
        if stackI:
            I_shape = np.prod(I_shape)
            I_shape = (I_shape, stackSize)
            self.previousStates = None

        self.mesCount = np.prod(M_shape)
        self.epsilon = 1.0
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.99995
        self.learningRate = 0.0001
        self.actionCount = num_actions
        self.batchSize = 64
        self.futureTargets = [4, 8, 16, 24, 32, 48]
        self.startPoint = 3*self.futureTargets[-1]
        # Memory
        self.memory = Memory(self.futureTargets, np.prod(M_shape), 30000, I_shape)
        self.timesteps = len(self.futureTargets)
        self.model = self.makeModel(I_shape, M_shape, G_shape)


                       #Image, Measurement, Goal
    def makeModel(self, I_shape, M_shape, G_shape):

        input_Image       = Input(shape=I_shape, name='image')
        input_Measurement = Input(shape=M_shape, name='measurements')
        input_Goal        = Input(shape=G_shape, name='goal')

        if (self.encoder != None):
            print("Using encoder model")
            i = Flatten()(input_Image)
            i = Dense(512, activation='relu')(i)
            i = Dense(512, activation='relu')(i)
            i = Dense(256, activation='relu')(i)
            i = Dense(256, activation='linear')(i)

        else:
            print("Using Convolutional model")
            i = Conv2D(32, (3, 3), activation='relu', padding='same')(input_Image)
            i = MaxPooling2D((2,2), padding='same')(i)
            i = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
            i = MaxPooling2D((2,2), padding='same')(i)
            i = Conv2D(64, (3, 3), activation='relu', padding='same')(i)
            i = MaxPooling2D((2,2), padding='same')(i)
            i = Conv2D(128, (3, 3), activation='relu', padding='same')(i)
            i = MaxPooling2D((2,2), padding='same')(i)
            i = Flatten()(i)
            i = Dense(1024, activation='relu')(i)
            i = Dense(512, activation='relu')(i)


        m = Flatten()(input_Measurement)
        m = Dense(64, activation='relu')(m)
        m = Dense(64, activation='relu')(m)
        m = Dense(64, activation='linear')(m)

        g = Flatten()(input_Goal)
        g = Dense(64, activation='relu')(g)
        g = Dense(64, activation='relu')(g)
        g = Dense(64, activation='linear')(g)

        merged = Concatenate()([i,m,g])
        pred_size = self.actionCount * np.prod(M_shape) * self.timesteps

        #Expectation stream
        expectation = Dense(128, activation='relu', name='expectation_1')(merged)
        expectation = Dense(np.prod(M_shape)*self.timesteps \
                    , activation='linear', name='expectation_2')(expectation)

        expectation = Concatenate()([expectation]*self.actionCount)

        #Action stream
        action = Dense(512, activation='relu', name='action_1')(merged)
        action = Dense(256, activation='relu', name='action_2')(action)
        action = Dense(256, activation='relu', name='action_3')(action)
        action = Dense(pred_size, activation='linear', name='action_4')(action)
        action = BatchNormalization()(action)

        action_expectation = Add()([action, expectation])

        output = Reshape((self.actionCount, M_shape[0], self.timesteps))(action_expectation)

        model = Model(inputs=(input_Image, input_Measurement, input_Goal), outputs=output)
        opt = Adam(lr=self.learningRate)
        model.compile(loss="mse", optimizer=opt)
        model.summary()
        plot_model(model, to_file='DFPNetwork.png')
        return model

    def encodeState(self, state):
        state = np.array([ip.reshape(state)])
        return np.array(self.encoder.predict(state))

    def remember(self, state, measurement, action, done):
        self.memory.append(state, measurement, action, done)

    def printInfo(self, state, mes, action, f, f_target):
            print(f"state: {state.shape}")
            print(f"mes: {mes.shape}")
            print(f"action: {action.shape}")
            print(f"f: {f.shape}")
            print(f"f_target: {f_target.shape}")

    def train(self, goal):
        if (self.memory.getSize() > self.startPoint):
            state, mes, action, f = self.memory.randomSample(self.batchSize)
            f = self.useGoal(f, goal)
            goal = np.repeat(np.array([goal]), self.batchSize, axis=0)
            f_target = np.array(self.pred(state, mes, goal))
            for i in range(self.batchSize):
                f_target[i][int(action[i])] = f[i]
            #self.printInfo(state, mes, action, f, f_target)
            self.model.train_on_batch([state, mes, goal], f_target)
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay

    def actionToTurn(self, step):
        turn = (step)/int(self.actionCount/2) - 1
        return turn

    def pred(self, state, mes, goal):
        return self.model.predict([state, mes, goal])

    def act(self, state, mes, goal):
        if len(self.memory.mem) < self.startPoint or np.random.rand() <= self.epsilon:
            return random.randrange(0,self.actionCount)
        prediction = np.array(self.pred(state, mes, np.array([goal])))
        prediction = np.reshape(prediction, (self.actionCount, self.timesteps * self.mesCount))
        prediction = np.sum(prediction, axis=1)
        return np.argmax(prediction)

    def useGoal(self, f_vec, goal):
        for i in range(f_vec.shape[0]):
            f_vec[i] = f_vec[i]*goal
        return f_vec

    def reshape(self, state, mes, splitImage=True, reset=False):
        if self.encoder != None:
            state = self.encodeState(state)
        if(reset == True and self.stackI):
            self.previousStates = np.stack((state, state, state, state), axis=0)
            self.previousStates = self.previousStates.reshape(1, state.shape[1], 4)
            state = self.previousStates
        elif self.stackI:
            state = state.reshape(1, state.shape[1], 1)
            self.previousStates = np.append(state, self.previousStates[:,:,:3], axis=2)
            state = self.previousStates
        else:
            state = ip.normalize(state)
            state = np.array([state])
        mes = np.array(mes)
        mes = mes.reshape((1, self.mesCount))
        return state, mes

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
