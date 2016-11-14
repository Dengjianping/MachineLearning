#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>

using namespace std;

class Neuron
{
private:
    double value;
    int weightsNum;
    double derivationValue;
    vector<double> weights;
    void initWeghts()
    {
        for (size_t i = 0; i < weightsNum; i++)
        {
            // double t = rand() / RAND_MAX;
            double t = rand() % 10;
            weights.push_back(t);
        }
    }
public:
    Neuron(const int N);
    double valueOfNeuron() const { return value; };
    double getDerivationValue() const { return derivationValue; };
    void changeDerivationValue(double target);
    void updateValue(double newValue) { value = newValue; };
    vector<double> weightsOfNeuron() const { return this->weights; };
    double sqrtError(double target);
    double derivationOfError(double target);
    double sigmoid();
    double derivationOfSigmoid();
    double derivationOfTotalError(double target);
    ~Neuron();
};

Neuron::Neuron(const int N)
{
    value = 0.0;
    derivationValue = 0.0;
    weightsNum = N;
    this->initWeghts();
}

void Neuron::changeDerivationValue(double target)
{
    derivationValue = this->derivationOfTotalError(target);
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
    double t = this->derivationOfError * this->derivationOfSigmoid();
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
    NeuronLayer(const int M, const int N);
    int nodesOfLayer() const { return nodes; };
    vector<Neuron> layerOfNeuron() const { return this->layer; };
    ~NeuronLayer();
};

NeuronLayer::NeuronLayer(const int M, const int N)
{
    nodes = M;
    for (size_t i = 0; i < nodes; i++)
    {
        Neuron n(N);
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
    NeuronNetwork(vector<double> & inputs, const int weightsNum, vector<int> & nodesNum, const int layersNum, vector<double> & targets);
    void forward();
    void backward();
    void updateDerivationOfNode();
    double updateWeight();
    void train();
    ~NeuronNetwork();
};

double NeuronNetwork::learningRate = 0.3;
double NeuronNetwork::errorPrecision = 1e-4;

NeuronNetwork::NeuronNetwork(vector<double> & inputs, const int weightsNum, vector<int> & nodesNum, const int layersNum, vector<double> & targets)
{
    inputs = inputs; targets = targets; layersNumber = layersNum;
    for (size_t i = 0; i < layersNumber; i++)
    {
        NeuronLayer layer(nodesNum[i], weightsNum);
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

void NeuronNetwork::updateDerivationOfNode()
{
    // update output layer first
    for (size_t i = 0; i < layers[layersNumber - 1].nodesOfLayer(); i++)
    {
        double t = 0.0;
        t = (layers[layersNumber - 1].layerOfNeuron()[i].derivationOfTotalError(targets[i])) * layers[layersNumber - 1].layerOfNeuron()[i].derivationOfSigmoid();
        layers[layersNumber - 1].layerOfNeuron()[i].changeDerivationValue(t);
    }
    // update hidden layers
    for (size_t i = layersNumber - 2; i >= 0; i--)
    {
        for (size_t j = 0; j < layers[i].nodesOfLayer(); i++)
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

double NeuronNetwork::updateWeight()
{
    // update output layer firstly
    for (size_t i = 0; i < layers[layersNumber-1].nodesOfLayer(); i++)
    {
        for (size_t j = 0; j < layers[layersNumber - 1].layerOfNeuron()[i].weightsOfNeuron().size(); j++)
        {
            double t = layers[layersNumber - 1].layerOfNeuron()[i].derivationOfTotalError(targets[i])*layers[layersNumber - 2].layerOfNeuron()[j].valueOfNeuron();
            t = layers[layersNumber - 1].layerOfNeuron()[i].weightsOfNeuron()[j] - learningRate*t;
        }
    }
    // update hidden layers weights
    for (size_t i = layersNumber - 2; i >= 0; i--)
    {

    }
}

NeuronNetwork::~NeuronNetwork()
{}

int main()
{
    const int N = 3;
    Neuron n(N);
    cout << n.valueOfNeuron() << endl;
    for (size_t i = 0; i < N; i++)
    {
        cout << n.weightsOfNeuron()[i] << endl;
    }
    system("pause");
    return 0;
}