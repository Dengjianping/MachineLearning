import math
import unittest

class Neuron:
    def __init__(self, bias=None):
        self.input = None
        self.output = None
        self.linearValue = None
        self.weights = [0.1,0.6]
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
    def __init__(self, bias=None, neurons=[]):
        self.neurons = neurons
        self.bias = bias
        
    def forward(self, intput):
        layerOutput = []
        for i in range(len(self.neurons):
            layerOutput.append(self.neurons[i].sigmoid(input))
        return layerOutput
        
    
        
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
        
if __name__ == '__main__':
    neuronSuite = unittest.TestLoader().loadTestsFromTestCase(TestNeuron)
    unittest.TextTestRunner(verbosity=2).run(neuronSuite)