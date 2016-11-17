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
private:
    double value;
    //int weightsNum;
    double derivationValue;
    vector<double> weights;
    void initWeghts(const int weightsNum)
    {
        for (size_t i = 0; i < weightsNum; i++)
        {
            double t = rand() / (double)RAND_MAX;
            weights.push_back(t);
        }
    }
public:
    Neuron(const int N);
    double valueOfNeuron() const { return value; };
    double getDerivationValue() const { return derivationValue; };
    void changeDerivationValue(double newValue);
    void updateValue(double newValue) { value = newValue; };
    void updateWeight(int index, int newValue) { weights[index] = newValue; };
    vector<double> weightsOfNeuron() const { return this->weights; };
    double sqrtError(double target);
    double derivationOfError(double target);
    double sigmoid();
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

double Neuron::sigmoid()
{
    return 1 / (1 + exp(-this->value));
}

double Neuron::derivationOfSigmoid()
{
    double t = this->sigmoid();
    return t*(1 - t);
}

double Neuron::derivationOfTotalError(double target)
{
    double t = derivationOfError(target) * derivationOfSigmoid();
    return t;
}

Neuron::~Neuron()
{}

class NeuronLayer
{
private:
    int nodes;
    vector<Neuron> layer;
public:
    NeuronLayer(const int nodesNumber, const int weightNumber);
    int nodesOfLayer() const { return nodes; };
    vector<Neuron> layerOfNeuron() const { return this->layer; };
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
private:
    int layersNumber;
    vector<NeuronLayer> layers;
    vector<double> inputs;
    vector<double> targets;
    static double learningRate;
    static double errorPrecision;
public:
    NeuronNetwork(vector<double> & inputs, vector<int> & weightSet, vector<int> & nodesNum, const int layersNum, vector<double> & targets);
    void forward();
    void backward();
    bool isConvergent();
    void updateDerivationOfNode();
    void updateWeights();
    void train();
    ~NeuronNetwork();
};

double NeuronNetwork::learningRate = 0.3;
double NeuronNetwork::errorPrecision = 1e-2;

NeuronNetwork::NeuronNetwork(vector<double> & inputs, vector<int> & weightSet, vector<int> & nodesNum, const int layersNum, vector<double> & expected)
{
    inputs = inputs; targets = expected; layersNumber = layersNum;
    for (size_t i = 0; i < layersNumber; i++)
    {
        NeuronLayer layer(nodesNum[i], weightSet[i]);
        layers.push_back(layer);
    }
}

void NeuronNetwork::forward()
{
    vector<double> lastNodeValue;
    // update first layer
    for (size_t i = 0; i < layers[0].nodesOfLayer(); i++)
    {
        double t = 0.0;
        for (size_t j = 0; j < inputs.size(); j++)
        {
            t += inputs[j] * layers[0].layerOfNeuron()[i].weightsOfNeuron()[j];
        }
        layers[0].layerOfNeuron()[i].updateValue(t);
    }

    for (size_t i = 1; i < layersNumber; i++)
    {
        for (size_t j = 0; j < layers[i].nodesOfLayer(); j++)
        {
            double t = 0.0;
            for (size_t k = 0; k < layers[i].layerOfNeuron()[j].weightsOfNeuron().size(); k++)
            {
                t += layers[i].layerOfNeuron()[j].weightsOfNeuron()[k] * layers[i - 1].layerOfNeuron()[k].valueOfNeuron();
            }
            layers[i].layerOfNeuron()[j].updateValue(t);
        }
    }
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
            double t = layers[layersNumber - 1].layerOfNeuron()[i].derivationOfTotalError(targets[i])*layers[layersNumber - 2].layerOfNeuron()[j].valueOfNeuron();
            t = layers[layersNumber - 1].layerOfNeuron()[i].weightsOfNeuron()[j] - learningRate*t;
            layers[layersNumber - 1].layerOfNeuron()[i].updateWeight(j, t);
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
                    double t = layers[i].layerOfNeuron()[j].getDerivationValue()*layers[i].layerOfNeuron()[j].sigmoid()*layers[i - 1].layerOfNeuron()[k].valueOfNeuron();
                    t = layers[i].layerOfNeuron()[j].weightsOfNeuron()[k] - learningRate*t;
                    layers[i].layerOfNeuron()[j].updateWeight(k, t);
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
                    double t = layers[0].layerOfNeuron()[j].getDerivationValue()*layers[0].layerOfNeuron()[j].sigmoid()*inputs[k];
                    t = layers[i].layerOfNeuron()[j].weightsOfNeuron()[k] - learningRate*t;
                    layers[i].layerOfNeuron()[j].updateWeight(k, t);
                }
            }
        }
    }
}

void NeuronNetwork::train()
{
    int i = 0;
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
            cout << "iteration: " << i << layers[layersNumber - 1].layerOfNeuron()[0].valueOfNeuron() << ", " << layers[layersNumber - 1].layerOfNeuron()[1].valueOfNeuron() << endl;
        }
    }
}

NeuronNetwork::~NeuronNetwork()
{}

int main()
{
    const int N = 3;
    Neuron n(N);
    cout << n.valueOfNeuron() << endl;

    vector<double> inputs = { 1.0,2.0,3.0 };
    int weightsNumber = 3;
    vector<int> weightsSet = { 0,3,4,3 };
    vector<int> nodesNumber = { 3,4,3,2 };
    int layersNum = 4;
    vector<double> expected = { 0.5,0.5 };

    NeuronNetwork nn(inputs, weightsSet, nodesNumber, layersNum, expected);
    nn.train();

    for (size_t i = 0; i < N; i++)
    {
        cout << n.weightsOfNeuron()[i] << endl;
    }
    system("pause");
    return 0;
}