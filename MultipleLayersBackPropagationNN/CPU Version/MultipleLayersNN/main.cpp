#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>
#include <chrono>

using namespace std;

//#define vector<vector<double> > matrix;

/*
    input layer: 3 nodes
    hidden layer: 1st layer has 4 nodes, 2nd layer has 3 nodes
    output layer: just 1 noe
    bias node: none
*/

const int numberOfInput = 3;
const int numberOfFirstHiddenLayer = 4;
const int numberOfSecondHiddenLayer = 3;
const int numberOfOutput = 1;

struct Node
{
    double input;
    vector<double> weights;
};

double nodeValue()
{
    srand(time(NULL));
    return rand() % 10;
}

vector<double> weightsGeneration(int number)
{
    vector<double> weights;
    srand(time(NULL));
    for (size_t i = 0; i < number; i++)
    {
        weights.push_back(rand() / double(RAND_MAX));
    }

    return weights;
}

vector<Node> initLayers(int nodesOfNumber, int adjacentNodesOfNumber)
{
    vector<Node> nodes;
    for (size_t i = 0; i < nodesOfNumber; i++)
    {
        Node n = { nodeValue(), weightsGeneration(adjacentNodesOfNumber) };
        nodes.push_back(n);
    }

    return nodes;
}

void organizedNodes()
{
    vector<vector<Node> > allNodes;

    // input layer : 3 nodes
    vector<Node> inputs = initLayers(numberOfInput, numberOfFirstHiddenLayer);
    allNodes.push_back(inputs);

    // first hidden layer: 4 nodes
    vector<Node> firstHiddenLayer = initLayers(numberOfFirstHiddenLayer, numberOfSecondHiddenLayer);
    allNodes.push_back(firstHiddenLayer);

    // first hidden layer: 3 nodes
    vector<Node> secondHiddenLayer = initLayers(numberOfSecondHiddenLayer, numberOfOutput);
    allNodes.push_back(secondHiddenLayer);
}

double linearInput(vector<double> & weights, vector<double> & inputs)
{
    double linearValue = 0.0;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        linearValue += weights[i] * inputs[i];
    }

    return linearValue;
}

double sigmoid(vector<double> & weights, vector<double> & inputs)
{
    double linearValue = 0.0;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        linearValue += weights[i] * inputs[i];
    }

    return 1 / (1 + exp(-linearValue));
}

double partialDerivationOfSigmoid(vector<double> & weights, vector<double> & inputs)
{
    double linearValue = 0.0;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        linearValue += weights[i] * inputs[i];
    }

    double sigmoidValue = 1 / (1 + exp(-linearValue));
    return sigmoidValue*(1 - sigmoidValue);
}

double partialDerivationOfLinearInput(int index, vector<double> & inputs)
{
    return inputs[index];
}

double partialDerivationOfError(double target, double currentValue)
{
    return currentValue - target;
}

vector<vector<vector<double> > > updateWeights(double alpha, double target, vector<vector<vector<double> > > & weights, vector<vector<double> > & inputs)
{
    vector<vector<vector<double> > > updatedW;
    for (size_t i = 0; i < weights.size(); i++)
    {
        vector<vector<double> > layer;
        for (size_t j = 0; j < weights[i].size(); j++)
        {
            vector<double> tmp;
            for (size_t k = 0; k < weights[i][j].size(); k++)
            {
                double t = weights[i][j][k] - alpha;
                tmp.push_back(t);
            }
            layer.push_back(tmp);
        }
        updatedW.push_back(layer);
    }

    return updatedW;
}

double costFuntion()
{

}

void misc()
{
    // node value
    vector<vector<double> > nodes = { {1,2,3},{1,2,3,4},{1,2,3} };

    // all weights matrix
    vector<vector<vector<double> > > weights;
    vector<vector<double> > inputs = {
        { 1,1,1,1 },
        { 1,1,1,1 },
        { 1,1,1,1 }
    };
    weights.push_back(inputs);
    vector<vector<double> > firstHiddenLayer = {
        { 1,1,1 },
        { 1,1,1 },
        { 1,1,1 },
        { 1,1,1 }
    };
    weights.push_back(firstHiddenLayer);
    vector<vector<double> > secondhiddenLayer = {
        { 1 },
        { 1 },
        { 1 }
    };
    weights.push_back(secondhiddenLayer);
}

void updateNodes(double alpha, vector<double> & target, vector<vector<Node> > & nodes)
{
    for (size_t i = 0; i < nodes.size(); i++)
    {
        if (i == nodes.size - 1)break;
        for (size_t j = 0; j < nodes[i].size(); j++)
        {
            for (size_t m = 0; m < nodes[i][j].weights.size(); m++)
            {
                double sum = 0;
                // sum += nodes[i][j].input*nodes[i + 1];
                for (size_t n = i+1; n < nodes.size(); n++)
                {
                    for (size_t p = 0; p < nodes[n].size(); p++)
                    {
                        for (size_t q = 0; q < nodes[n][p].weights.size(); q++)
                        {
                            nodes[i][j].input*nodes[n][p].input*(1- nodes[n][p].input)*nodes[n][p].weights[q]*
                        }
                    }
                }
                for (size_t n = 0; n < nodes[i+1].size(); n++)
                {
                    for (size_t p = 0; p < nodes[i+1][n].weights.size(); p++)
                    {
                        sum += nodes[i][j].input*nodes[i+1][n].input*(1- nodes[i + 1][n].input)*nodes[i + 1][n].weights[index]*
                    }
                }
            }
        }
    }
}

void updateWeights(double alpha, vector<double> & target, vector<vector<Node> > & nodes)
{
    for (int i = 0; i < nodes.size(); i++)
    {
        if (i == nodes.size() - 1)break;
        for (int j = 0; j < nodes[i].size(); j++)
        {
            for (int m = 0; m < nodes[i][j].weights.size(); m++) // index of weights in single nodes
            {
                //nodes[i][j].value * 
                static vector<double> tmp;
                for (int n = i + 1; n < nodes.size(); n++) // 
                {
                    if (tmp.empty())
                    {
                        double t = nodes[i][j].value * nodes.[n][m].value * (1 - nodes[n][m].value);
                        for (int p = 0; p < nodes[n][m].weights.size(); p++)
                        {
                            tmp.push_back(nodes[n][m].weights[p]*t);
                        }
                    }
                    else
                    {
                        for (int q = 0; q < nodes[n].size(); q++)
                        {
                            tmp[q] * nodes[n][q].value * (1 - nodes[n][q]);
                        }
                    }
                }
            }
        }
    }
}