#include <iostream>
#include <vector>
#include <cmath>

#define ROW 3
#define COL 2
using namespace std;

class LinearRegression
{
private:
    int row, col;
	float** trainX;
	float* trainY;
	float* theta;
	static float learnRate;
	static float precision;
public:
	LinearRegression(const vector<vector<float> > & featureX, const vector<float> & featureY);
	float hypothesis(float* x);
	float costFunction();
	float partialDerivation(int index);
	void updateTheta();
	void train();
	void showResult()const;
	~LinearRegression();
};

float LinearRegression::learnRate = 0.3;
float LinearRegression::precision = 1e-6;

LinearRegression::LinearRegression(const vector<vector<float> > & featureX, const vector<float> & featureY)
{
    row = featureX.size(); col = featureX[0].size();
    // initialize trainX
    trainX = new float* [row];
    for (size_t i = 0; i < row; i++)
    {
        trainX[i] = new float[col];
    }
	for (size_t i = 0; i < row; i++)
	{
		for (size_t j = 0; j < col; j++)
		{
			trainX[i][j] = featureX[i][j];
		}
	}
    
    // initialize trainY
    trainY = new float [row];
	for (size_t i = 0; i < row; i++)
	{
		trainY[i] = featureY[i];
	}
    
    // initialize theta
    theta = new float[col];
	for (size_t i = 0; i < col; i++)
	{
		theta[i] = 0.0;
	}
}

float LinearRegression::hypothesis(float* x)
{
	float hypo = 0.0;
	for (size_t i = 0; i < COL; i++)
	{
		hypo += theta[i] * x[i];
	}
	return hypo;
}

float LinearRegression::costFunction()
{
	float cost = 0.0;
	for (size_t i = 0; i < ROW; i++)
	{
		float t = hypothesis(trainX[i]) - trainY[i];
		cost += pow(t, 2);
	}
	return 0.5 * cost / (float)ROW;
}

float LinearRegression::partialDerivation(int index)
{
	float partial = 0.0;
	for (size_t i = 0; i < ROW; i++)
	{
		partial += (hypothesis(trainX[i]) - trainY[i]) * trainX[i][index];
	}
	return partial;
}

void LinearRegression::updateTheta()
{
	// update theta
	for (size_t i = 0; i < COL; i++)
	{
		theta[i] -= learnRate * partialDerivation(i);
	}
}

void LinearRegression::train()
{
	int interation = 0;
	float lastCost = costFunction();
	for (;;)
	{
		interation += 1;
		cout << "interation: " << interation << ", cost: " << lastCost << endl;
		updateTheta();
		if (fabs(costFunction() - lastCost) <= precision)
		{
			cout << interation << endl;
			break;
		}
		else
		{
			lastCost = costFunction();
		}
	}
}

void LinearRegression::showResult() const
{
	for (size_t i = 0; i < COL; i++)
	{
		cout << theta[i] << endl;
	}
}

LinearRegression::~LinearRegression()
{
    for (size_t i = 0; i < row; i++)
    {
        delete[] trainX[i];
    }
    delete[] trainX;
    delete[] trainY;
    delete[] theta;
}

int main()
{
	// suppose this euqation is  y = 4*x + 2
	vector<vector<float> > x = { { 1.0,1.0 },{ 1.0,2.0 },{ 1.0,3.0 } };
	vector<float> y = { 6,10,14 };
	LinearRegression ln(x, y);
	ln.train();
	ln.showResult();

	system("pause");
	return 0;
}