#include <cuda_runtime.h>
#include <device_lanuch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace std;

template <class T>
class Neuron
{
private:
    __managed__ T value;
    __managed__ T differentialTotalError;
    device_vector<T> weights;
    void initialWeights(int weightsNumber)
    {
        for (size_t i = 0; i < weightsNumber; i++)
        {
            T t = rand() / (T)RAND_MAX;
            weights.push_back(t);
        }
    }
public:
    Neuron();
    Neuron(const host_vector<T> & w);
    __host__ __device__ T sigmoid();
    __host__ __device__ T differentialSigmoid();
    __host__ __device__ T totalError(T target);
    __host__ __device__ void changeValue(T newValue);
    __host__ __device__ void changeSingleWeight(int index, T newValue);
    ~Neuron();
};

Neuron：：Neuron()
{
    value = totalError = 0;
    initialWeights(0);
}

Neuron::Neuron(int weightsNumber)
{
    value = totalError = 0;
    initialWeights(weightsNumber);
}

__host__ __device__ T Neuron<T>::sigmoid()
{
    value = 1 / (1 + exp(-value))
    return value;
}

__host__ __device__ T Neuron<T>::differentialSigmoid()
{
    return value * (1-value);
}

__host__ __device__ T Neuron<T>::totalError(T target)
{
    return 0.5 * powf(value - target, 2);
}

__host__ __device__ void Neuron<T>::changeValue(T newValue)
{
    if (newValue != value)
    {
        value = newValue;
    }
}

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
private:
    device_vector<Neuron<T> > layer;
public:
    NeuronLayer(host_vector<T> & inputs); // for input layer
    NeuronLayer(int neuronNumber, int eachNeuronOfWeightsNumber); // for hidden layer and output layer
    ~NeuronLayer();
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
class NeuronNetwork
{
private:
    device_vector<NeuronLayer> network;
public:
    NeuronNetwork();
    ~NeuronNetwork();
}