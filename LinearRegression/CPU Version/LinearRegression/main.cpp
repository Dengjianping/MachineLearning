#include <iostream>
#include <vector>
#include <cmath>

#define ROW 3
#define COL 2
using namespace std;

class LinearRegression
{
private:
	float trainX[ROW][COL];
	float trainY[ROW];
	float theta[COL];
	static float learnRate;
	static float precision;
public:
	LinearRegression(float featureX[][COL], float featureY[ROW]);
	float hypothesis(float x[COL]);
	float costFunction();
	float partialDerivation(int index);
	void train();
	void showResult()const;
	~LinearRegression() {};
};

float LinearRegression::learnRate = 0.3;
float LinearRegression::precision = 1e-4;

LinearRegression::LinearRegression(float featureX[][COL], float featureY[ROW])
{
	for (size_t i = 0; i < ROW; i++)
	{
		for (size_t j = 0; j < COL; j++)
		{
			trainX[i][j] = featureX[i][j];
		}
	}

	for (size_t i = 0; i < ROW; i++)
	{
		trainY[i] = featureY[i];
	}

	for (size_t i = 0; i < COL; i++)
	{
		theta[i] = 0.0;
	}
}

float LinearRegression::hypothesis(float x[COL])
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

void LinearRegression::train()
{
	int interation = 0;
	float lastCost = costFunction();
	for (;;)
	{
		interation += 1;
		// update theta
		for (size_t i = 0; i < COL; i++)
		{
			theta[i] -= learnRate * partialDerivation(i);
		}
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

int main()
{
	// suppose this euqation is  y = 4*x + 2
	float x[ROW][COL] = { { 1.0,1.0 },{ 1.0,2.0 },{ 1.0,3.0 } };
	float y[ROW] = { 6,10,14 };
	LinearRegression ln(x, y);
	ln.train();
	ln.showResult();

	system("pause");
	return 0;
}