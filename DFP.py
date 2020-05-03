#import tensorflow.keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers import BatchNormalization, Reshape, Subtract, Add
from keras.layers import LeakyReLU, Concatenate, Dense, LSTM, Input, Concatenate
from keras.optimizers import Adam
from keras.models import load_model
import functions as f
from collections import deque
from keras.utils import plot_model
import keras.backend as K
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

    def append(self, state, measurement, action, done, goal):
        self.mem.append((state, measurement, action, done, goal))

    def getSize(self):
        return len(self.mem)

    # Get random array of f_vector recieved from the according state and action
    def randomSample(self, count):
        states = np.zeros(((count,) + self.stateShape))
        measurements = np.zeros((count, self.mesCount))
        actions = np.zeros((count))
        f_vec = np.zeros((count, self.mesCount * self.timesteps))
        goals = np.zeros((count, self.mesCount * self.timesteps))
        randomInds = np.random.choice(len(self.mem)-(self.maxTimestep+1), count)

        for i, ind in enumerate(randomInds):
            if self.mem[ind][3]:
                ind += 1
            isDone = False
            future = np.zeros((self.mesCount, self.timesteps))
            k = 0
            for j in range(self.maxTimestep+1):
                if not self.mem[ind+j][3] and (j in self.futurePreds) and not isDone:
                    for m in range(self.mesCount):
                        future[m][k] = self.mem[ind+j][1][0][m]# - self.mem[ind][1][0][m]
                    k += 1
                elif (j in self.futurePreds):
                    if not isDone:
                        offset = j
                    isDone = True
                    for m in range(self.mesCount):
                        future[m][k] = self.mem[ind+offset][1][0][m]# - self.mem[ind][1][0][m]
                    k += 1
            states[i]       = self.mem[ind][0]
            measurements[i] = self.mem[ind][1]
            actions[i]      = self.mem[ind][2]
            f_vec[i]        = future.ravel(order='F')
            goals[i]        = self.mem[ind][4]
        #print(goals.shape)    
        return states, measurements, actions, f_vec, goals


class DFPAgent:
    # To stack the Image input, it must be a 1d vector representation of the image.
    def __init__(self, num_actions, I_shape, M_shape, G_shape, pred_v=[1, 2, 4, 8, 16, 32], \
            encoded=False):
        # Possible pretrained encoder for Image data
        self.encoded = encoded
        self.mesCount = np.prod(M_shape)
        self.epsilon =          1.0
        self.epsilon0 =         1.0
        self.epsilonMin =       0.001
        self.epsilonDecay =     30000
        self.learningRate =     0.00001
        self.maxLearningRate =  0.00015
        self.minLearningRate =  0.000002
        self.learningRateDecay= 0.9995
        self.actionCount = num_actions
        self.batchSize = 32
        self.futureTargets = pred_v
        self.startPoint = 100
        # Memory
        self.memory = Memory(self.futureTargets, np.prod(M_shape), 30000, I_shape)
        self.timesteps = len(self.futureTargets)
        self.model = self.makeModel(I_shape, M_shape, G_shape)


                       #Image, Measurement, Goal
    def makeModel(self, I_shape, M_shape, G_shape):

        input_Image       = Input(shape=I_shape, name='Encoded_input')
        input_Measurement = Input(shape=M_shape, name='Measurement_input')
        input_Goal        = Input(shape=G_shape, name='Goal_input')

        if (self.encoded):
            print("Using encoder model")
            i = Flatten()(input_Image)
            i = Dense(512, activation='relu')(i)
            i = Dense(512, activation='relu')(i)
            i = Dense(256, activation='relu')(i)
            i = Dense(256, activation='linear')(i)

        else:
            print("Using Convolutional model")
            i = Conv2D(32, (8, 8), strides=(4, 4))(input_Image)
            i = LeakyReLU()(i)
            i = Conv2D(64, (4, 4), strides=(2, 2))(i)
            i = LeakyReLU()(i)
            i = Conv2D(64, (3, 3))(i)
            i = LeakyReLU()(i)
            i = Flatten()(i)
            i = Dense(512, activation='relu')(i)


        m = Dense(128)(input_Measurement)
        m = LeakyReLU()(m)
        m = Dense(128)(m)
        m = LeakyReLU()(m)
        m = Dense(128, activation='relu')(m)

        g = Dense(128)(input_Goal)
        g = LeakyReLU()(g)
        g = Dense(128)(g)
        g = LeakyReLU()(g)
        g = Dense(128, activation='relu')(g)

        merged = Concatenate()([i,m,g])
        pred_size = np.prod(M_shape) * self.timesteps

        #Expectation stream
        expectation = Dense(512, name='Expectation_1')(merged)
        expectation = LeakyReLU()(expectation)
        expectation = Dense(pred_size \
                    , activation='relu', name='Expectation_2')(expectation)

        #Action stream
        actions = Dense(1024, name='Action_1')(merged)
        actions = LeakyReLU()(actions)
        actions = Dense(self.actionCount*pred_size,activation='relu', name='Action_2')(actions)
        actions = Reshape((self.actionCount, pred_size))(actions)
        actions = BatchNormalization()(actions)
        """
        actions = Dense(self.actionCount*pred_size,activation='relu', name='Action_1')(merged)
        actions = Reshape((self.actionCount, pred_size))(actions)
        """

        predictions = []
        for i in range(self.actionCount):
            action = Lambda(lambda x: x[:,i,:])(actions)
            out = Add()([action, expectation])
            predictions.append(out)

        model = Model([input_Image, input_Measurement, input_Goal], predictions)
        opt = Adam(lr=self.learningRate, decay=1e-6)
        model.compile(loss="mse", optimizer=opt)
        model.summary()
        plot_model(model, to_file='DFPNetwork.png')
        return model

    def info(self):
        print(f"Epsilon: {self.epsilon}")
        print(f"Mem size: {self.memory.getSize()}")
        lr = self.model.optimizer.get_config()['learning_rate']
        print(f"Learning rate: {lr}")

    def remember(self, state, measurement, action, done, goal):
        self.memory.append(state, measurement, action, done, goal)

    def train(self):
        if (self.memory.getSize() > self.startPoint):
            state, mes, action, f, goal = self.memory.randomSample(self.batchSize)
            f_target = self.pred(state, mes, goal)
            for i in range(self.batchSize):
                f_target[int(action[i])][i, :] = f[i]
            
            loss = self.model.train_on_batch([state, mes, goal], f_target)
            if self.epsilon > self.epsilonMin:
                self.epsilon -= (self.epsilon0 - self.epsilonMin)/self.epsilonDecay
            return loss

    def actionToTurn(self, step):
        turn = step%3
        turn = (turn) - 1
        speed = 0.4
        if (step >= 3):
            speed = 0.6
        return turn, speed

    def pred(self, state, mes, goal):
        return self.model.predict([state, mes, goal])

    def act(self, state, mes, goal):
        self.train()
        if len(self.memory.mem) < self.startPoint or np.random.rand() <= self.epsilon:
            return random.randrange(0,self.actionCount)
        prediction = np.array(self.pred(state, mes, np.array([goal])))
        prediction = np.vstack(prediction)
        sums = np.sum(np.multiply(prediction, goal), axis=1)
        

        return np.argmax(sums)

    def decayLearningRate(self):
        if self.learningRate > self.minLearningRate:
            self.learningRate = self.minLearningRate
            K.set_value(self.model.optimizer.lr, self.learningRate)

    def useGoal(self, f_vec, goal):
        for i in range(f_vec.shape[0]):
            f_vec[i] = f_vec[i]*goal
        return f_vec

    def reshape(self, state, mes):
        state = ip.normalize(state)
        #mes = np.array(mes)
        mes = mes.reshape((1, self.mesCount))
        return state, mes

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


