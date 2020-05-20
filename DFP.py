#import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Multiply
from tensorflow.keras.layers import BatchNormalization, Reshape, Subtract, Add, RepeatVector
from tensorflow.keras.layers import LeakyReLU, Concatenate, Dense, LSTM, Input, Concatenate
from tensorflow.keras.initializers import TruncatedNormal
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
import math

def msra_stddev(x, k_h, k_w): 
    return 1/math.sqrt(0.5*k_w*k_h*x.shape[-1])


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
        self.epsilonMin =       0.01
        self.epsilonDecay =     2000
        self.learningRate =     0.0001
        self.maxLearningRate =  0.0001
        self.minLearningRate =  0.000001
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
            #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(i, 1, 1)))(i)
            i = Dense(512, activation='relu')(i)
            #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(i, 1, 1)))(i)
            i = Dense(256, activation='relu')(i)
            #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(i, 1, 1)))(i)
            i = Dense(256, activation='linear')(i)
            #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(i, 1, 1)))(i)

        else:
            print("Using Convolutional model")
            i = Conv2D(32, (8, 8), strides=4, padding="same",
                bias_initializer="ones",
                kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(input_Image, 8, 8)))(input_Image)
            i = LeakyReLU(alpha=0.2)(i)
            
            i = Conv2D(64, (4, 4), strides=2,
                bias_initializer="ones",
                kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(i, 4, 4)))(i)
            i = LeakyReLU(alpha=0.2)(i)
            
            i = Conv2D(64, 3, strides=1,
                bias_initializer="ones",
                kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(i, 3, 3)))(i)
            i = LeakyReLU(alpha=0.2)(i)
            
            i = Flatten()(i)
            i = Dense(512, activation='linear',
                kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(i, 1, 1)))(i)
            

        # Measurement
        m = Dense(128)(input_Measurement)
        #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(input_Measurement, 1, 1)))(input_Measurement)
        m = LeakyReLU(alpha=0.2)(m)

        m = Dense(128)(m)
        #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(m, 1, 1)))(m)
        m = LeakyReLU()(m)
        
        m = Dense(128, activation='linear')(m)
        #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(m, 1, 1)))(m)

        # Goal
        g = Dense(128)(input_Goal)
        #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(input_Goal, 1, 1)))(input_Goal)
        g = LeakyReLU(alpha=0.2)(g)

        g = Dense(128)(g)
        #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(g, 1, 1)))(g)
        g = LeakyReLU()(g)
        
        g = Dense(128, activation='linear')(g)
        #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(g, 1, 1)))(g)

        j = Concatenate()([i,m,g])
        pred_size = np.prod(M_shape) * self.timesteps

        #Expectation stream
        expectation = Dense(512, name='Expectation_1')(j)
        #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(j, 1, 1)))(j)
        expectation = LeakyReLU(alpha=0.2)(expectation)

        expectation = Dense(pred_size \
                    , activation='linear', name='Expectation_2')(expectation)
                    #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(expectation, 1, 1)))(expectation)
        
        expectation = Concatenate()([expectation]*self.actionCount)

        #Action stream
        actions = Dense(512, name='Action_1')(j)
            #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(j, 1, 1)))(j)
        actions = LeakyReLU(alpha=0.2)(actions)

        actions = Dense(self.actionCount*pred_size,activation='linear')(actions)
            #kernel_initializer=TruncatedNormal(stddev=0.9*msra_stddev(actions, 1, 1)),
            #name='Action_2')(actions)
        #actions = BatchNormalization()(actions)
       
        predictions = Add()([actions, expectation])
        predictions = Reshape((self.actionCount, pred_size))(predictions)
        Mask_shape = (self.actionCount, pred_size)
        input_mask = Input(shape=Mask_shape, name='mask_input')
        predictions = Multiply()([predictions, input_mask])

        model = Model([input_Image, input_Measurement, input_Goal, input_mask], predictions)
        #model = Model([input_Image, input_Measurement, input_Goal], predictions)
        opt = Adam(lr=self.learningRate, beta_1=0.95, beta_2=0.999)
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

            for i in range(self.batchSize):
                f_target[i][int(action[i])][:] = f[i]
                mask[i] = self.makeMask(action[i])
            self.trains += 1
            loss = self.model.train_on_batch([state, mes, goal, mask], f_target)
            #loss = self.model.train_on_batch([state, mes, goal], f_target)
            self.lock.release() 
            if self.epsilon > self.epsilonMin:
                self.epsilon *= 0.995
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
        #return self.model.predict([state, mes, goal])


    def pred(self, state, mes, goal):
        self.lock.acquire() 
        pred = self.lpred(state, mes, goal)
        self.lock.release() 
        return pred

    def act(self, state, mes, goal):
        if len(self.memory.mem) < self.startPoint or np.random.rand() <= self.epsilon:
            return random.randrange(0, self.actionCount)
        prediction = np.array(self.pred(state, mes, np.array([goal])))
        prediction = np.vstack(prediction)
        sums = np.sum(np.multiply(prediction, goal), axis=1)
        return np.argmax(sums)

    def decayLearningRate(self):
        self.lock.acquire()
        if self.learningRate > self.minLearningRate:
            self.learningRate *= 0.998
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
