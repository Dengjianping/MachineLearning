import math
import random
import unittest

class Neuron:
    def __init__(self):
        self.input = None
        self.output = None
        self.linearValue = None
        self.weights = []
        
    def linearInput(self):
        t = 0
        for i in range(len(self.input)):
            t += self.weights[i]*self.input[i]
        return t
        
    def sigmoid(self, input):
        self.input = input
        self.output = 1 / (1 + math.exp(-self.linearInput()))
        return self.output
        
    def totalError(self, target):
        return 0.5*math.pow(target - self.output, 2)
        
    def partitialDerivationOfOutputError(self, target):
        return self.output - target
        
    def partitialDerivationOfSigmoid(self):
        return self.output*(1 - self.output)
        
    def partitialDerivationOfToalError(self, target):
        return self.partitialDerivationOfSigmoid()*self.partitialDerivationOfOutputError(target)
        
    def indexWeights(self, index):
        return self.weights[index]
        
    def indexInput(self, index):
        return self.input[index]

class NeuronLayer:
    def __init__(self, numHidden):
        self.neurons = self.initlayer(numHidden)
        self.output = []
    
    def initlayer(self, numHidden):
        neurons = []
        for i in range(numHidden):
            neurons.append(Neuron())
        return neurons
    
    def feedForward(self, inputs):
        layerOutput = []
        for neuron in self.neurons:
            layerOutput.append(neuron.sigmoid(inputs))
        
        self.output = layerOutput
        return layerOutput
        
    def getOutput(self, index):
        return self.output[index]
        
    def totalError(self, targets):
        t = 0
        for i in range(len(targets)):
            t += self.neurons[i].totalError(targets[i])
        return t

class NeuronNetwork:
    def __init__(self, numInputs, inputs, numHidden, numOutput, learnRate, precision, targets):
        self.numInputs = numInputs
        self.inputs = inputs
        self.learnRate = learnRate
        self.precision = precision
        self.targets = targets
        self.hiddenLayer = NeuronLayer(numHidden)
        self.outputLayer = NeuronLayer(numOutput)
        # initialize hidden layer weights
        self.initHiddenLayerWeights()
        # initialize output layer weights
        self.initOutputLayerWeights()
    
    def initHiddenLayerWeights(self):
        w = [[0.15, 0.2], [0.25, 0.3]]
        for i, neuron in enumerate(self.hiddenLayer.neurons):
            weights = []
            for j in range(self.numInputs):
                weights.append(random.random())
            self.hiddenLayer.neurons[i].weights = w[i]
            print weights
        
    def initOutputLayerWeights(self):
        w = [[0.4, 0.45], [0.5, 0.55]]
        for i, neuron in enumerate(self.outputLayer.neurons):
            weights = []
            for j, neuron in enumerate(self.hiddenLayer.neurons):
                weights.append(random.random())
            self.outputLayer.neurons[i].weights = w[i]
            print weights
        
    def feedForward(self):
        inputForOutput = self.hiddenLayer.feedForward(self.inputs)
        return self.outputLayer.feedForward(inputForOutput)
        
    def backPropagation(self):
        # update output layer weights
        for i in range(len(self.outputLayer.neurons)):
            for j in range(len(self.outputLayer.neurons[i].weights)):
                self.outputLayer.neurons[i].weights[j] -= self.learnRate * self.outputLayer.neurons[i].partitialDerivationOfToalError(self.targets[i]) * self.hiddenLayer.getOutput(j)
                
        # update hidden layer weights
        for i in range(len(self.hiddenLayer.neurons)):
            for j in range(len(self.hiddenLayer.neurons[i].weights)):
                derivation = 0
                # calculate toal error of derivation
                for m in range(len(self.outputLayer.neurons)):
                    derivation += self.outputLayer.neurons[m].partitialDerivationOfToalError(self.targets[m]) * self.outputLayer.neurons[m].weights[i]
                # print derivation
                self.hiddenLayer.neurons[i].weights[j] -= self.learnRate * self.inputs[i] * self.hiddenLayer.neurons[i].partitialDerivationOfSigmoid()
                        
    def train(self):
        self.feedForward()
        print self.outputLayer.output
        self.backPropagation()
        i = 0
        while (True):
            i += 1
            print i
            if (self.outputLayer.totalError(self.targets) <= self.precision):
                break
            print self.outputLayer.totalError(self.targets)
            self.backPropagation()
            self.feedForward()
        
if __name__ == '__main__':
    numInputs = 2
    inputs = [0.5, 0.1]
    numHidden = 2
    numOutput = 2
    learnRate = 0.3
    precision = 1e-4
    targets = [0.1, 0.99]
    nn = NeuronNetwork(numInputs, inputs, numHidden, numOutput, learnRate, precision, targets)
    nn.train()