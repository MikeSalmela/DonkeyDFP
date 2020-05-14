#import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Multiply
from tensorflow.keras.layers import BatchNormalization, Reshape, Subtract, Add, RepeatVector
from tensorflow.keras.layers import LeakyReLU, Concatenate, Dense, LSTM, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import functions as f
from collections import deque
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import numpy as np
import random
import imageprocessor as ip
import threading

class Memory:
    def __init__(self, futureVec, mesCount, maxSize, stateShape):
        self.stateShape = stateShape
        self.timesteps = len(futureVec)
        self.futurePreds = futureVec
        self.mesCount = mesCount
        self.maxTimestep = futureVec[-1]
        self.mem = deque(maxlen=maxSize)
        self.lock = threading.Lock()

    def append(self, state, measurement, action, done, goal):
        self.mem.append((state, measurement, action, done, goal))

    def appendBatch(self, batch):
        for i in range(len(batch)):
            self.mem.append((batch[i][0], batch[i][1], batch[i][2], batch[i][3], batch[i][4]))

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
                        future[m][k] = self.mem[ind+j][1][0][m] - self.mem[ind][1][0][m]
                    k += 1
                elif (j in self.futurePreds):
                    if not isDone:
                        offset = j
                    isDone = True
                    for m in range(self.mesCount):
                        future[m][k] = self.mem[ind+offset][1][0][m] - self.mem[ind][1][0][m]
                    k += 1
            states[i]       = self.mem[ind][0]
            measurements[i] = self.mem[ind][1]
            actions[i]      = self.mem[ind][2]
            f_vec[i]        = future.ravel(order='F')
            #print(f_vec[i])
            goals[i]        = self.mem[ind][4]
        #print(goals.shape)    
        return states, measurements, actions, f_vec, goals


class DFPAgent:
    # To stack the Image input, it must be a 1d vector representation of the image.
    def __init__(self, num_actions, I_shape, M_shape, G_shape, pred_v=[1, 2, 4, 8, 16, 32], \
            encoded=False, name=""):
        # Possible pretrained encoder for Image data
        print("Create agent")
        self.encoded = encoded
        self.mesCount = np.prod(M_shape)
        self.epsilon =          1
        self.epsilon0 =         1
        self.epsilonMin =       0.05
        self.epsilonDecay =     2000
        self.learningRate =     0.00001
        self.maxLearningRate =  0.00001
        self.minLearningRate =  0.000005
        self.learningRateDecay= 0.9995
        self.actionCount = num_actions
        self.batchSize = 64
        self.futureTargets = pred_v
        self.startPoint = 100
        self.train_cycle = 32
        self.lock = threading.Lock()
        self.trains = 0
        # Memory
        self.memory = Memory(self.futureTargets, np.prod(M_shape), 20000, I_shape)
        self.timesteps = len(self.futureTargets)
        self.predSize = self.timesteps * np.prod(M_shape)
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

        j = Concatenate()([i,m,g])
        pred_size = np.prod(M_shape) * self.timesteps

        #Expectation stream
        expectation = Dense(512, name='Expectation_1')(j)
        expectation = LeakyReLU()(expectation)
        expectation = Dense(pred_size \
                    , activation='linear', name='Expectation_2')(expectation)
        expectation = Concatenate()([expectation]*self.actionCount)

        #Action stream
        actions = Dense(1024, name='Action_1')(j)
        actions = LeakyReLU()(actions)
        actions = Dense(self.actionCount*pred_size,activation='relu', name='Action_2')(actions)
        #actions = BatchNormalization()(actions)
        #actions = Reshape((self.actionCount, pred_size))(actions)
        """
        actions = Dense(self.actionCount*pred_size,activation='relu', name='Action_1')(merged)
        actions = Reshape((self.actionCount, pred_size))(actions)
        """
        """
        predictions = Add()([actions, expectation])
        """
        """
        predictions = []
        for i in range(self.actionCount):
            #action = Lambda(lambda x: x[:,i,:])(actions)
            action = Dense(pred_size, activation='relu')(merged)
            out = Add()([action, expectation])
            predictions.append(out)
        """

        predictions = Add()([actions, expectation])
        predictions = Reshape((self.actionCount, pred_size))(predictions)
        Mask_shape = (self.actionCount, pred_size)
        input_mask = Input(shape=Mask_shape, name='mask_input')
        predictions = Multiply()([predictions, input_mask])

        model = Model([input_Image, input_Measurement, input_Goal, input_mask], predictions)
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

    def rememberBatch(self, batch):
        self.lock.acquire()
        self.memory.appendBatch(batch)
        self.lock.release()
    
    def train(self):
        self.lock.acquire()
        if (self.memory.getSize() > self.startPoint):
            state, mes, action, f, goal = self.memory.randomSample(self.batchSize)
            #f_target = self.lpred(state, mes, goal)
            f_target = np.zeros((self.batchSize, self.actionCount, self.predSize))
            mask = np.zeros((self.batchSize, self.actionCount, self.predSize))
            #print(f_target.shape)

            for i in range(self.batchSize):
                f_target[i][int(action[i])][:] = f[i]
                mask[i] = self.makeMask(action[i])
                #print(mask[i])
            #print("f_target")
            #print(f_target[0]) 
            self.trains += 1
            loss = self.model.train_on_batch([state, mes, goal, mask], f_target)
            self.lock.release() 
            if self.epsilon > self.epsilonMin:
                self.epsilon *= 0.99#9#6
            return loss
        self.lock.release() 

    def actionToTurn(self, step):
        turn = step%3
        turn = (turn) - 1
        speed = 0.2
        if (step >= 3):
            speed = 0.6
        return turn, speed

    def lpred(self, state, mes, goal):
        return self.model.predict([state, mes, goal, self.makeOnes()])


    def pred(self, state, mes, goal):
        self.lock.acquire() 
        pred = self.model.predict([state, mes, goal, self.makeOnes()])
        self.lock.release() 
        return pred

    def act(self, state, mes, goal):
        if len(self.memory.mem) < self.startPoint or np.random.rand() <= self.epsilon:
            return random.randrange(0, self.actionCount)
        self.lock.acquire() 
        prediction = np.array(self.lpred(state, mes, np.array([goal])))
        prediction = np.vstack(prediction)
        #print(prediction.shape)
        #print(prediction)
        #mp = np.multiply(prediction, goal)
        #print("multiply")
        #print(mp)
        sums = np.sum(np.multiply(prediction, goal), axis=1)
        #print("sums")
        #print(sums)
        self.lock.release() 
        return np.argmax(sums)

    def decayLearningRate(self):
        self.lock.acquire()
        if self.learningRate > self.minLearningRate:
            if self.learningRate < 0.00001:
                self.learningRate *= 0.99
            else:
                self.learningRate *= 0.9
            self.model.optimizer.lr.assign(self.learningRate)
        self.lock.release()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def makeOnes(self):
        return np.ones((1, self.actionCount, self.predSize))

    def makeMask(self, action):
        mask = np.zeros((self.actionCount, self.predSize))
        mask[int(action)] = np.ones((self.predSize))
        return mask.reshape(1, self.actionCount, self.predSize)
