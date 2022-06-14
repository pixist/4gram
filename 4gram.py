# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:59:42 2022

@author: novar
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

class fourgramNN:
    def __init__(self, Lembed, Lhid, Lfeature):
        self.Lembed = Lembed
        self.Lhid = Lhid
        self.Lfeature = Lfeature
        
    def initializeLayers(self, N):
        self.embed1 = np.zeros((self.Lembed, N))
        self.embed2 = np.zeros((self.Lembed, N))
        self.embed3 = np.zeros((self.Lembed, N))
        self.hidden = np.zeros((self.Lhid, N))
        self.out = np.zeros((self.Lfeature, N))
        
    def initializeWeights(self):
        self.weights1 = np.random.normal(0, 0.01, (self.Lembed, self.Lfeature))
        self.weights21 = np.random.normal(0, 0.01, (self.Lhid, self.Lembed))
        self.weights22 = np.random.normal(0, 0.01, (self.Lhid, self.Lembed))
        self.weights23 = np.random.normal(0, 0.01, (self.Lhid, self.Lembed))
        self.weights3 = np.random.normal(0, 0.01, (self.Lfeature, self.Lhid))
        self.b2 = np.random.normal(0, 0.01, (self.Lhid,1))
        self.b3 = np.random.normal(0, 0.01, (self.Lfeature,1))
        
        self.gradW1, self.gradW21, self.gradW22, self.gradW23, self.gradW3, self.gradb2, self.gradb3 = 0,0,0,0,0,0,0
        self.gradW1New, self.gradW21New, self.gradW22New, self.gradW23New, self.gradW3New, self.gradb2New, self.gradb3New = 0,0,0,0,0,0,0
    
    
    def forward(self, data):
        self.embed1 = np.dot(self.weights1,idxToOH(data[0]))
        self.embed2 = np.dot(self.weights1,idxToOH(data[1]))
        self.embed3 = np.dot(self.weights1,idxToOH(data[2]))
        A11 = np.dot(self.weights21,self.embed1)+self.b2
        A12 = np.dot(self.weights22,self.embed2)+self.b2
        A13 = np.dot(self.weights23,self.embed3)+self.b2
        self.hidden = sigmoid(A11+A12+A13)
        A2 = np.dot(self.weights3,self.hidden)+self.b3
        self.out = softmax(A2).T # make the shape same with data
        return self.out
    
    def forwardWord(self):
        return np.argmax(self.out,axis=1) + 1
    
    def crossEntropy(self, ground):
        """
        onehot = np.zeros((ground.size, self.Lfeature))
        onehot[np.arange(ground.size),ground-1] = 1
        return -np.sum(onehot*np.log(self.out))
        """
        return -np.log(self.out.T[ground-1])
        
        
    def calculateGrad(self, data, ground):
        #wrong gradient
        out = self.out
        hid = self.hidden
        em1 = self.embed1
        em2 = self.embed2
        em3 = self.embed3
        #dJ_do__do_dA22 = (-(1/out[gnd-1])*(np.diag(self.out[i])-out*out.T))[:,gnd-1][:,None] long way
        dJ_do__do_dA2 = out.T - idxToOH(ground)
        dJ_dA1 = np.dot(self.weights3.T,dJ_do__do_dA2)*(hid*(1-hid))
        dJ_dE1 = np.dot(self.weights21.T,dJ_dA1)
        dJ_dE2 = np.dot(self.weights22.T,dJ_dA1)
        dJ_dE3 = np.dot(self.weights23.T,dJ_dA1)
        gradW3 = dJ_do__do_dA2*hid.T
        gradW21 = dJ_dA1*em1.T
        gradW22 = dJ_dA1*em2.T
        gradW23 = dJ_dA1*em3.T
        gradW1 = (dJ_dE1*idxToOH(data[0]).T + dJ_dE2*idxToOH(data[1]).T + dJ_dE3*idxToOH(data[2]).T)/3
        gradb3 = dJ_do__do_dA2
        gradb2 = dJ_dA1
        self.gradW1New = self.gradW1New + gradW1
        self.gradW21New = self.gradW21New + gradW21
        self.gradW22New = self.gradW22New + gradW22
        self.gradW23New = self.gradW23New + gradW23
        self.gradW3New = self.gradW3New + gradW3
        self.gradb2New = self.gradb2New + gradb2
        self.gradb3New = self.gradb3New + gradb3
        
    def updateWeights(self, learningRate, momentum):
        #run after grad calculation is done for enough samples depending on the batch size
        self.gradW1 = momentum*self.gradW1 + self.gradW1New
        self.gradW21 = momentum*self.gradW21 + self.gradW21New
        self.gradW22 = momentum*self.gradW22 + self.gradW22New
        self.gradW23 = momentum*self.gradW23 + self.gradW23New
        self.gradW3 = momentum*self.gradW3 + self.gradW3New
        self.gradb2 = momentum*self.gradb2 + self.gradb2New
        self.gradb3 = momentum*self.gradb3 + self.gradb3New
        
        self.weights1 = self.weights1 - learningRate*self.gradW1
        self.weights21 = self.weights21 - learningRate*self.gradW21
        self.weights22 = self.weights22 - learningRate*self.gradW22
        self.weights23 = self.weights23 - learningRate*self.gradW23
        self.weights3 = self.weights3 - learningRate*self.gradW3
        self.b2 = self.b2 - learningRate*self.gradb2
        self.b3 = self.b3 - learningRate*self.gradb3
        
        self.gradW1New, self.gradW21New, self.gradW22New, self.gradW23New, self.gradW3New, self.gradb2New, self.gradb3New = 0,0,0,0,0,0,0
    
    
    def trainStep(self, sample, target):
        # calls required methods sequentally except updateWeights()
        self.forward(sample)
        guess, _ = self.forwardOut()
        loss = self.crossEntropy(target.reshape(target.shape[1]))
        self.calcGrad(sample, target.reshape(target.shape[1]))
        return loss, guess

def idxToOH(idx, word = 250):
    oneHot = np.zeros((word,1))
    oneHot[idx-1] = 1
    return oneHot

def trainMiniBatch(nnModel, data, ground, valdat, valgnd, epoch, learningRate, momentum, batchSize = 200):
    lossListT, lossListV = [], []
    totalSamples = len(ground)
    batchCount = totalSamples//batchSize
    remainder = totalSamples % batchSize
    remLimit = totalSamples - remainder
    for e in range(epoch):
        permutation = list(np.random.permutation(totalSamples))
        shuffled_samples = data[permutation]
        shuffled_grounds = ground[permutation]
        samples = np.array_split(shuffled_samples[:remLimit], batchCount)
        grounds = np.array_split(shuffled_grounds[:remLimit], batchCount)
        samples.append(shuffled_samples[remLimit:])
        grounds.append(shuffled_grounds[remLimit:])
        loss = 0
        for j in range(len(grounds)):
            bSize = len(grounds[j])
            for i in range(bSize):
                nnModel.forward(samples[j][i])
                loss += nnModel.crossEntropy(grounds[j][i])
                nnModel.calculateGrad(samples[j][i], grounds[j][i])
            nnModel.updateWeights(learningRate,momentum)
        lossListT.append(loss)
        loss = 0
        for i in range(len(valgnd)):
            nnModel.forward(valdat[i])
            loss += nnModel.crossEntropy(valgnd[i])
        loss = loss/len(valgnd)
        print(f"Training and Validation Loss in epoch {e+1}: {lossListT[e]}, {lossListV[e]}")
        lossListV.append(loss)
        
        if loss > 1.2*lossListT[0]: 
            print("Terminated due to increased loss")
            return lossListV, lossListT
        elif (e > 1) & (lossListT[e-1] - lossListT[e] < 0.01):
            print("Terminated due to convergence")
            return lossListV, lossListT
    return lossListV, lossListT

def estimateForward(model, testX, testD, words, best=10):
    global probs
    for i in range(len(testD)):
        probs = model.forward(testX[i])
        idx = probs.argsort()[0,-best:][::-1]
        s = testX[i]
        gnd = testD[i]
        print(f"\nTrigram: {words[s[0]]} {words[s[1]]} {words[s[2]]} [{words[gnd]}], Guess: ",end="")
        for j in range(best):
            print(words[idx[j]+1],end=" ")
    print()
    return probs, idx
        

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def plotParameter(metric, labels, metricName):
    plt.figure(figsize = (12,6))
    xlabel = [str(i) for i in range(len(metric[0]))]
    for i in range(len(labels)):
        plt.plot(xlabel, metric[i], marker='o', markersize=6, linewidth=2, label=labels[i])
    plt.ylabel(metricName[0])
    plt.title(f'{metricName[1]} with {metricName[2]} Embed Size, {metricName[3]} Hidden Neurons, Learning Rate: {metricName[4]}, Momentum: {metricName[5]}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
# In[] Read the data
filename = "data2.h5"

with h5py.File(filename, "r") as f:
    groupKeys = list(f.keys())
    sets = []
    for key in groupKeys:
        sets.append(list(f[key]))
# In[]
testD = np.array(sets[0])
testX = np.array(sets[1])
trainD = np.array(sets[2][:])
trainX = np.array(sets[3][:])
valD = np.array(sets[4][:])
valX = np.array(sets[5][:])
words = [0]
words.extend(sets[6])
words = np.array(words, dtype=str)
# In[]
D, P = 8,64 #(32,256) (16,128) (8,64)
model = fourgramNN(D, P, 250)
model.initializeLayers(1)
model.initializeWeights()
lossT, lossV = [], []
# In[]
lr = 0.015
mm = 0.5
epoch = 50
print(f"Started Training with learning rate = {lr}, momentum = {mm}")
l1, l2 = trainMiniBatch(model, trainX, trainD, valX, valD, epoch, lr, mm)
lossV.extend(l1)
lossT.extend(l2)
# In[]
idx = np.random.permutation(len(testD))
estimateForward(model, testX[idx][:5], testD[idx][:5], words)
# In[plot]
plotParameter([lossT, lossV], ["Training", "Validation"], ["Loss","4-Gram Model",D,P,lr,mm])
