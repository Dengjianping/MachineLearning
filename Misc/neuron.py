import math
import random
import unittest

class Neuron:
    def __init__(self, bias=None):
        self.input = None
        self.output = None
        self.linearValue = None
        self.weights = None
        self.bias = bias
        
    def linearInput(self):
        t = 0
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
    def __init__(self, numHidden, bias=None):
        self.neurons = self.initlayer(numHidden)
        self.bias = bias
    
    def initlayer(self, numHidden):
        neurons = []
        for i in range(numHidden):
            neurons.append(Neuron(bias))
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
            t += self.neurons.totalError(targets[i])
        return t

class NeuronNetwork:
    def __init__(self, numInputs, numHidden, numOutput, learnRate, precision, targets, hiddenLayerBias=None, outputLayerBias=None):
        self.numInputs = numInputs
        self.learnRate = learnRate
        self.precision = precision
        self.targets = targets
        self.hiddenLayer = NeuronLayer(numHidden, hiddenLayerBias)
        self.outputLayer = NeuronLayer(numOutput, outputLayerBias)
        # initialize hidden layer weights
        self.initHiddenLayerWeights()
        # initialize output layer weights
        self.initOutputLayerWeights()
    
    def initHiddenLayerWeights(self):
        for i, neuron in enumerate(self.hiddenLayer.neurons):
            for j in range(self.numInputs):
                self.hiddenLayer[i].weights[j] = random.random()
        
    def initOutputLayerWeights(self):
        for i in range(len(self.outputLayer.neurons):           
            for j in range(len(self.hiddenLayer.neurons):
                self.outputLayer.neurons[i].weights[j] = random.random()
        
    def feedForward(self, inputs):
        inputForOutput = self.hiddenLayer.feedForward(inputs)
        return self.outputLayer.feedForward(inputForOutput)
        
    def backPropagation(self, inputs):
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
                self.hiddenLayer.neurons[i].weights[j] = self.hiddenLayer.neurons[i].weights[j] - self.learnRate * inputs[i] * self.hiddenLayer.neurons[i].partitialDerivationOfSigmoid()
                        
    def train(self, inputs):
        self.feedForward(inputs)
        while (True):
            if (self.outputLayer.totalError(self.targets) <= self.precision):
                break
            self.backPropagation(inputs)
            self.feedForward(inputs)       
        
class TestNeuron(unittest.TestCase):
    def setUp(self):
        self.neuron = Neuron(0)
        
    @unittest.expectedFailure
    def testSigmoid(self):
        self.assertEquals(0.78, self.neuron.sigmoid([1,4]))
        
    @unittest.skipIf(self.neuron.input==None, "please assign input array to Neuron")
    def testIndexInput(self):
        self.neuron.sigmoid([1,4])
        self.assertEquals(4, self.neuron.indexInput(0))
        
    def tearDown(self):
        self.neuron = None

class TestNeuronLayer(unittest.TestCase):
    def setUp(self):
        self.layer = NeuronLayer(2)
        
    def tearDown(self):
        self.layer = None
class TestNeuronNetwork(unittest.TestCase):
    def setUp(self):
        self.network = NeuronNetwork()
        
    def tearDown(self):
        self.network = None
        
if __name__ == '__main__':
    neuronSuite = unittest.TestLoader().loadTestsFromTestCase(TestNeuron)
    layerSuite = unittest.TestLoader().loadTestsFromTestCase(TestNeuronLayer)
    networkSuite = unittest.TestLoader().loadTestsFromTestCase(TestNeuronNetwork)
    unittest.TextTestRunner(verbosity=2).run(neuronSuite)