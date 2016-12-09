import math
import random
import unittest

class Neuron:
    def __init__(self):
        self.input = None
        self.output = None
        self.linearValue = None
        self.weights = []
        
    # @property
    # def weights(self):
        # return self.weights
        
    # @weights.setter
    # def weihgts(self, weights):
        # if type(weights) != list:
            # raise TypeError("please assign a list")
        # self.weights = weights
        
        
    def linearInput(self):
        t = 0
        print len(self.input)
        for i in range(len(self.input)):
            t += self.weights[i]*self.input[i]
        return t + self.bias
        
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
    
    def initlayer(self, numHidden):
        neurons = []
        for i in range(numHidden):
            neurons.append(Neuron())
        return neurons
    
    def feedForward(self, inputs):
        layerOutput = []
        for neuron in self.neurons:
            layerOutput.append(neuron.sigmoid(inputs))
        return layerOutput
        
    def getOutput(self):
        output = []
        for neuron in self.neurons:
            output.append(neuron.output)
        return output
        
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
    
    # def __repr__(self):
        # return 'NeuronNetwork({0.numInputs}, {0.inputs}, {0.numHidden}, {0.numOutput}, {0.learnRate}, {0.precision}, {0.targets})'.format(self)
        
    # def __str__(self):
        # pass
    
    # def __enter__(self):
        # pass
        
    # def __exit__(self):
        # pass
    
    # def __format__(self):
        # pass
    
    def initHiddenLayerWeights(self):
        for i, neuron in enumerate(self.hiddenLayer.neurons):
            weights = []
            for j in range(self.numInputs):
                weights.append(random.random())
            self.hiddenLayer.neurons[i].weights = weights
        
    def initOutputLayerWeights(self):
        for i, neuron in enumerate(self.outputLayer.neurons):
            weights = []
            for j, neuron in enumerate(self.hiddenLayer.neurons):
                weights.append(random.random())
            self.outputLayer.neurons[i].weights = weights
        
    def feedForward(self):
        inputForOutput = self.hiddenLayer.feedForward(self.inputs)
        return self.outputLayer.feedForward(inputForOutput)
        
    def backPropagation(self):
        # update output layer weights
        for i in range(len(self.outputLayer.neurons)):
            for j in range(len(self.outputLayer.neurons[i].weights)):
                self.outputLayer.neurons[i].weights[j] = self.outputLayer.neurons[i].weights[j] - self.learnRate * \
                    self.outputLayer.neurons[i].partitialDerivationOfToalError(self.targets[i]) * self.hiddenLayer.neurons.getOutput()[j]
                
        # update hidden layer weights
        for i in range(len(self.hiddenLayer.neurons)):
            for j in range(len(self.hiddenLayer.neurons[i].weights)):
                derivation = 0
                # calculate toal error of derivation
                for m in range(len(self.outputLayer.neurons)):
                    derivation += self.outputLayer.neurons[m].partitialDerivationOfToalError(self.targets[m]) * self.outputLayer.neurons[m].weights[i]
                self.hiddenLayer.neurons[i].weights[j] = self.hiddenLayer.neurons[i].weights[j] - self.learnRate * self.inputs[i] * self.hiddenLayer.neurons[i].partitialDerivationOfSigmoid()
                        
    def train(self):
        self.feedForward()
        while (True):
            if (self.outputLayer.totalError(self.targets) <= self.precision):
                break
            self.backPropagation()
            self.feedForward()
        
if __name__ == '__main__':
    numInputs = 2
    inputs = [2.5, 5,7]
    numHidden = 2
    numOutput = 2
    learnRate = 0.3
    precision = 1e-4
    targets = [0.3, 0.6]
    nn = NeuronNetwork(numInputs, inputs, numHidden, numOutput, learnRate, precision, targets)
    nn.train()