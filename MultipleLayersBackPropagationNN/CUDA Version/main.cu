#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace std;


__constant__ double learningRate = 0.3;
__constant__ double precision = 1e-4;

template <class T>
class Neuron
{
public:
    // member data
    T value;
    T differentialTotalErrorValue;
    device_vector<T> weights;
    void initialWeights(int weightsNumber)
    {
        for (size_t i = 0; i < weightsNumber; i++)
        {
            T t = rand() / (T)RAND_MAX;
            weights.push_back(t);
        }
    }
    // member functions
    Neuron();
    Neuron(int weightsNumber);
    __host__ __device__ T sigmoid(T input);
    __host__ __device__ T differentialSigmoid();
    __host__ __device__ T totalError(T target);
    __host__ __device__ T differentialTotalError(T target) { return value - target; };
    __host__ __device__ void changeValue(T newValue);
    __host__ __device__ void changeSingleWeight(int index, T newValue);
    __host__ __device__ int weightsCount() const { return weights.size(); };
    __host__ __device__ T valueOfNeuron() const { return value; };
    ~Neuron() {};
};

template <class T>
Neuron<T>::Neuron()
{
    value = differentialTotalErrorValue = 0;
    initialWeights(0);
}

template <class T>
Neuron<T>::Neuron(int weightsNumber)
{
    value = differentialTotalErrorValue = 0;
    initialWeights(weightsNumber);
}

template <class T>
__host__ __device__ T Neuron<T>::sigmoid(T input)
{
    value = 1 / (1 + exp(-input))
        return value;
}

template <class T>
__host__ __device__ T Neuron<T>::differentialSigmoid()
{
    return value * (1 - value);
}

template <class T>
__host__ __device__ T Neuron<T>::totalError(T target)
{
    return 0.5 * powf(value - target, 2);
}

template <class T>
__host__ __device__ void Neuron<T>::changeValue(T newValue)
{
    if (newValue != value)
    {
        value = newValue;
    }
}

template <class T>
__host__ __device__ void Neuron<T>::changeSingleWeight(int index, T newValue)
{
    if (newValue != value)
    {
        weights[index] = newValue;
    }
}

template <class T>
class NeuronLayer
{
public:
    // member data
    device_vector<Neuron<T> > layer;
    // member functions
    NeuronLayer(host_vector<T> & inputs); // for input layer
    NeuronLayer(int neuronNumber, int eachNeuronOfWeightsNumber); // for hidden layer and output layer
    T valueOfSingleNeuron(int index) const { return layer[index].valueOfNeuron(); };
    __host__ __device__ void changeSingleNeuronValue(int index, T newValue);
    ~NeuronLayer() {};
};

template <class T>
NeuronLayer<T>::NeuronLayer(host_vector<T> & inputs)
{
    for (size_t i = 0; i < inputs.size(); i++)
    {
        Neuron<T> neuron();
        neuron.changeValue(inputs[i]);
        layer.push_back(neuron);
    }
}

template <class T>
NeuronLayer<T>::NeuronLayer(int neuronNumber, int eachNeuronOfWeightsNumber)
{
    for (size_t i = 0; i < neuronNumber; i++)
    {
        Neuron<T> neuron(eachNeuronOfWeightsNumber);
        layer.push_back(neuron);
    }
}

template <class T>
__host__ __device__ void NeuronLayer<T>::changeSingleNeuronValue(int index, T newValue)
{
    if (layer[index].valueOfNeuron() != newValue)
    {
        layer[index].changeValue(newValue);
    }
}

template <class T>
class NeuronNetwork
{
private:
    NeuronLayer<T> inputLayer;
    device_vector<NeuronLayer<T> > hiddenLayers;
    NeuronLayer<T> outputLayer;
    device_vector<T> targets;
public:
    NeuronNetwork(host_vector<T> inputs, host_vector<T> hiddenLayerWeightsSet, host_vector<T> hiddenLayerSet, int outputNeuronNumber, int countOfWeights, host_vector<T> expected);
    __host__ __device__ void forward();
    __host__ __device__ void backward();
    __host__ __device__ bool isConvergenced();
    __host__ __device__ void showWeights() const;
    ~NeuronNetwork() {};
};

template <class T>
NeuronNetwork<T>::NeuronNetwork(host_vector<T> inputs, host_vector<T> hiddenLayerWeightsSet, host_vector<T> hiddenLayerSet, int outputNeuronNumber, int countOfWeights, host_vector<T> expected)
{
    // construct input layer
    inputLayer = NeuronLayer<T>(inputs);

    // construct hidden layers and output layer
    for (size_t i = 0; i < hiddenLayerWeightsSet.size(); i++)
    {
        NeuronLayer hiddenLayer(hiddenLayerSet[i], hiddenLayerWeightsSet[i]);
        network.push_back(hiddenLayer);
    }

    // construct output layer
    outputLayer = NeuronLayer<T>(outputNeuronNumber, countOfWeights);

    targets = expected;
}



template <class T>
__host__ __device__ void NeuronNetwork<T>::forward()
{
    // handle hidden layers
    for (size_t i = 0; i < hiddenLayers.size(); i++)
    {
        for (size_t j = 0; j < hiddenLayers[i].layer.size()(); j++)
        {
            T t = (T)0;
            for (size_t k = 0; k < hiddenLayers[i].layer[j].weights.size(); k++)
            {
                if (i == 0)
                {
                    t += hiddenLayers[i].layer[j].weights[k] * inputLayer[k].value;
                }
                else
                {
                    t += hiddenLayers[i].layer[j].weights[k] * hiddenLayers[i - 1].layer[k].value;
                }
            }
            hiddenLayers[i].layer[j].sigmoid(t);
        }
    }

    // handle output layer
    for (size_t i = 0; i < outputLayer.size(); i++)
    {
        T t = (T)0;
        for (size_t j = 0; j < outputLayer.weights.size(); j++)
        {
            t += outputLayer[i].weights[j] * hiddenLayers.back()[j].value;
        }
        outputLayer[i].sigmoid(t);
    }
}

template <class T>
__host__ __device__ void NeuronNetwork<T>::backward()
{
    // update output layer weights first
    for (size_t i = 0; i < outputLayer.size(); i++)
    {
        outputLayer[i].differentialTotalErrorValue = outputLayer[i].differentialSigmoid() * outputLayer[i].differentialTotalError(targets[i]);
        for (size_t j = 0; j < outputLayer.weights.size(); j++)
        {
            T t = outputLayer[i].differentialTotalErrorValue * hiddenLayers.back()[j].value;
            outputLayer[i].weights[j] = outputLayer[i].weights[j] - learningRate * t;
        }
    }

    // update hidden layers weights
    for (size_t i = hiddenLayers.size() - 1; i >= 0; i--)
    {
        for (size_t j = 0; j < hiddenLayers[i].size(); j++)
        {
            if (i == hiddenLayers.size() - 1)
            {
                T t = (T)0;
                for (size_t h = 0; h < outputLayer.size(); h++)
                {
                    t += outputLayer[h].differentialTotalErrorValue * outputLayer[h].weight[j];
                }
                hiddenLayers.back()[j].differentialTotalErrorValue = t;
                // update weight
                for (size_t m = 0; m < hiddenLayers[i].layer[j].weights.size(); m++)
                {
                    T t = hiddenLayers[i].layer[j].differentialTotalErrorValue * hiddenLayers[i].layer[j].differentialSigmoid() * hiddenLayers[i - 1].layer[m].value;
                    hiddenLayers[i].layer[j].weights[m] = hiddenLayers[i].layer[j].weights[m] - learningRate * t;
                }
            }
            else
            {
                T t = (T)0;
                for (size_t h = 0; h < hiddenLayers[i + 1].size(); h++)
                {
                    t += hiddenLayers[i + 1].layer[h].differentialTotalErrorValue * hiddenLayers[i + 1].layer[h].weight[j];
                }
                hiddenLayers.back()[j].differentialTotalErrorValue = t;
                // 
                for (size_t m = 0; m < hiddenLayers[i].layer[j].weights.size(); m++)
                {
                    T t = hiddenLayers[i].layer[j].differentialTotalErrorValue * hiddenLayers[i].layer[j].differentialSigmoid() * hiddenLayers[i - 1].layer[m].value;
                    hiddenLayers[i].layer[j].weights[m] = hiddenLayers[i].layer[j].weights[m] - learningRate * t;
                }
            }
        }
    }
}

template <class T>
__host__ __device__ bool NeuronNetwork<T>::isConvergenced()
{

}

int main()
{
    system("pause");
    return 0;
}