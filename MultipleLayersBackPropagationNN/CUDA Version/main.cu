#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

class Neuron
{
public:
    double value;
    double derivationValue;
    vector<double> weights;
    void initWeghts(const int weightsNum)
    {
        for (size_t i = 0; i < weightsNum; i++)
        {
            double t = rand() / (double)RAND_MAX;
            //cout << t << endl;
            weights.push_back(t);
        }
    }
    //public:
    Neuron(const int N);
    double valueOfNeuron() const { return value; };
    double getDerivationValue() const { return derivationValue; };
    void changeDerivationValue(double newValue);
    void updateValue(double newValue) { value = newValue; };
    void updateWeight(int index, double newValue) { weights[index] = newValue; };
    vector<double> weightsOfNeuron() { return this->weights; };
    double sqrtError(double target);
    double derivationOfError(double target);
    double sigmoid(double input);
    double derivationOfSigmoid();
    double derivationOfTotalError(double target);
    ~Neuron();
};

Neuron::Neuron(const int weightsNum)
{
    value = 0.0;
    derivationValue = 0.0;
    //weightsNum = N;
    this->initWeghts(weightsNum);
}

void Neuron::changeDerivationValue(double newValue)
{
    derivationValue = newValue;
}

double Neuron::sqrtError(double target)
{
    return pow(target - value, 2);
}

double Neuron::derivationOfError(double target)
{
    return value - target;
}

double Neuron::sigmoid(double input)
{
    value = 1 / (1 + exp(-input));
    //cout << "in class: " << value << endl;
    return value;
}

double Neuron::derivationOfSigmoid()
{
    return value * (1 - value);
}

double Neuron::derivationOfTotalError(double target)
{
    return derivationOfError(target) * derivationOfSigmoid();
}

Neuron::~Neuron()
{}

class NeuronLayer
{
public:
    int nodes;
    vector<Neuron> layer;
    NeuronLayer(const int nodesNumber, const int weightNumber);
    int nodesOfLayer() const { return nodes; };
    vector<Neuron> layerOfNeuron() { return this->layer; };
    ~NeuronLayer();
};

NeuronLayer::NeuronLayer(const int nodesNumber, const int weightNumber)
{
    nodes = nodesNumber;
    for (size_t i = 0; i < nodes; i++)
    {
        Neuron n(weightNumber);
        layer.push_back(n);
    }
}

NeuronLayer::~NeuronLayer()
{}

class NeuronNetwork
{
public:
    int layersNumber;
    vector<NeuronLayer> layers;
    vector<double> inputs;
    vector<double> targets;
    static double learningRate;
    static double errorPrecision;
    //public:
    NeuronNetwork(vector<double> & inputs, vector<int> & weightSet, vector<int> & nodesNum, const int layersNum, vector<double> & targets);
    vector<NeuronLayer> getLayers() { return layers; };
    void forward();
    void backward();
    void changeANeuronVaue(int layerIndex, int neuronIndex, double newValue);
    bool isConvergent();
    void updateDerivationOfNode();
    void updateWeights();
    void train();
    void showWeights() const
    ~NeuronNetwork();
};

double NeuronNetwork::learningRate = 0.3;
double NeuronNetwork::errorPrecision = 1e-6;

NeuronNetwork::NeuronNetwork(vector<double> & inputSet, vector<int> & weightSet, vector<int> & nodesNum, const int layersNum, vector<double> & expected)
{
    inputs = inputSet; targets = expected; layersNumber = layersNum;
    for (size_t i = 0; i < layersNumber; i++)
    {
        NeuronLayer layer(nodesNum[i], weightSet[i]);
        layers.push_back(layer);
    }
    for (size_t i = 0; i < inputs.size(); i++)
    {
        layers[0].layerOfNeuron()[i].updateValue(inputs[i]);
    }
}

void NeuronNetwork::forward()
{
    for (size_t i = 1; i < layersNumber; i++)
    {
        for (size_t j = 0; j < layers[i].nodesOfLayer(); j++)
        {
            double t = 0.0;
            for (size_t k = 0; k < layers[i].layer[j].weights.size(); k++)
            {
                if (i == 1)
                {
                    t += layers[i].layer[j].weights[k] * inputs[k];
                }
                else
                {
                    t += layers[i].layer[j].weights[k] * layers[i - 1].layer[k].value;
                }
            }
            layers[i].layer[j].sigmoid(t);
        }
    }
}

void NeuronNetwork::changeANeuronVaue(int layerIndex, int neuronIndex, double newValue)
{
    layers[layerIndex].layer[neuronIndex].updateValue(newValue);
}

bool NeuronNetwork::isConvergent()
{
    double t = 0.0;
    for (size_t i = 0; i < layers[layersNumber - 1].nodesOfLayer(); i++)
    {
        t += layers[layersNumber - 1].layerOfNeuron()[i].sqrtError(targets[i]);
    }
    if (t <= errorPrecision)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void NeuronNetwork::updateDerivationOfNode()
{
    // update output layer first
    for (size_t i = 0; i < layers[layersNumber - 1].nodesOfLayer(); i++)
    {
        double t = 0.0;
        t = (layers[layersNumber - 1].layerOfNeuron()[i].derivationOfTotalError(targets[i]));
        layers[layersNumber - 1].layerOfNeuron()[i].changeDerivationValue(t);
    }
    // update hidden layers
    for (size_t i = layersNumber - 2; i > 0; i--)
    {
        for (size_t j = 0; j < layers[i].nodesOfLayer(); j++)
        {
            //layers[i].layerOfNeuron()[j]
            double t = 0.0;
            for (size_t k = 0; k < layers[i + 1].layerOfNeuron().size(); k++)
            {
                t += layers[i + 1].layerOfNeuron()[k].getDerivationValue() * layers[i + 1].layerOfNeuron()[k].weightsOfNeuron()[j];
            }
            layers[i].layerOfNeuron()[j].changeDerivationValue(t);
        }
    }
}

void NeuronNetwork::updateWeights()
{
    // update output layer firstly
    for (size_t i = 0; i < layers[layersNumber - 1].nodesOfLayer(); i++)
    {
        for (size_t j = 0; j < layers[layersNumber - 1].layerOfNeuron()[i].weightsOfNeuron().size(); j++)
        {
            double t = layers[layersNumber - 1].layerOfNeuron()[i].derivationOfTotalError(targets[i])*layers[layersNumber - 2].layer[j].value;
            t = layers[layersNumber - 1].layerOfNeuron()[i].weightsOfNeuron()[j] - learningRate*t;
            layers[layersNumber - 1].layer[i].updateWeight(j, t);
        }
    }
    // update hidden layers weights
    for (size_t i = layersNumber - 2; i > 0; i--)
    {
        if (i > 0)
        {
            for (size_t j = 0; j < layers[i].nodesOfLayer(); j++)
            {
                for (size_t k = 0; k < layers[i].layerOfNeuron()[j].weightsOfNeuron().size(); k++)
                {
                    double t = layers[i].layerOfNeuron()[j].getDerivationValue()*layers[i].layerOfNeuron()[j].value*layers[i - 1].layerOfNeuron()[k].valueOfNeuron();
                    t = layers[i].layerOfNeuron()[j].weightsOfNeuron()[k] - learningRate*t;
                    layers[i].layer[j].updateWeight(k, t);
                }
            }
        }
        // the first hidden layer
        else
        {
            for (size_t j = 0; j < layers[0].nodesOfLayer(); j++)
            {
                for (size_t k = 0; k < layers[0].layerOfNeuron()[j].weightsOfNeuron().size(); k++)
                {
                    double t = layers[0].layerOfNeuron()[j].getDerivationValue()*layers[0].layerOfNeuron()[j].value*inputs[k];
                    t = layers[i].layerOfNeuron()[j].weightsOfNeuron()[k] - learningRate*t;
                    layers[i].layer[j].updateWeight(k, t);
                }
            }
        }
    }
}

void NeuronNetwork::train()
{
    int i = 0;
    forward();
    cout << "iteration: " << i << ", " << layers[layersNumber - 1].layerOfNeuron()[0].valueOfNeuron() << ", " << layers[layersNumber - 1].layerOfNeuron()[1].valueOfNeuron() << endl;
    while (true)
    {
        i++;
        forward();
        if (isConvergent())
        {
            break;
        }
        else
        {
            updateDerivationOfNode();
            updateWeights();
            cout << "iteration: " << i << ", " << layers[layersNumber - 1].layerOfNeuron()[0].valueOfNeuron() << ", " << layers[layersNumber - 1].layerOfNeuron()[1].valueOfNeuron() << endl;
        }
    }
}

void NeuronNetwork::showWeights() const
{
    for (size_t i = 0; i < layersNumber; i++)
    {
        for (size_t j = 0; j < layers[i].nodes; j++)
        {
            cout << "node[" << i << "]" << "[" << j << "]" << ": " << layers[i].layer[j].value << endl;
            for (size_t k = 0; k < layers[i].layer[j].weights.size(); k++)
            {
                cout << layers[i].layer[j].weights[k] << endl;
            }
        }
    }
}

NeuronNetwork::~NeuronNetwork()
{}

int main()
{
    const int N = 3;
    Neuron n(N);
    srand((unsigned)time(NULL));

    vector<double> inputSet = { 3,5,8 };
    int weightsNumber = 3;
    vector<int> weightsSet = { 0,3,4,3 };
    vector<int> nodesNumber = { 3,4,3,2 };
    int layersNum = 4;
    vector<double> expected = { 0.3,0.8 };

    NeuronNetwork nn(inputSet, weightsSet, nodesNumber, layersNum, expected);
    nn.train();

    system("pause");
    return 0;
}