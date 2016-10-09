#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

double hypothesis(vector<double> & theta, vector<double> & x)
{
	double sum = 0.0;
	for (size_t i = 0; i < x.size(); i++)
	{
		sum += theta[i] * x[i];
	}

	return sum;
}

double costFunction(vector<double> & theta, vector<vector<double> > & x, vector<double> & y)
{
	double sum = 0.0;
	for (size_t i = 0; i < y.size(); i++)
	{
		double h = hypothesis(theta, x[i]);
		sum += pow((h - y[i]), 2);
	}

	return 0.5*sum / y.size();
}

double partialDerivative(int thetaIndex, vector<double> & theta, vector<vector<double> > & x, vector<double> & y)
{
	double m = y.size();
	double sum = 0.0;
	for (size_t i = 0; i < y.size(); i++)
	{
		double h = hypothesis(theta, x[i]);
		sum += (h - y[i])*x[i][thetaIndex];
	}
	
	return sum / m;
}

vector<double> gradientDescent(double alpha, vector<double> & theta, vector<vector<double> > & x, vector<double> & y)
{
	vector<double> updatedTheta;
	for (size_t i = 0; i < theta.size(); i++)
	{
		double u = theta[i] - alpha*partialDerivative(i, theta, x, y);
		updatedTheta.push_back(u);
	}

	return updatedTheta;
}

int main()
{
	double alpha = 0.3;
	double error = 0.0;

	vector<double> theta = { 0.0,0.0 };
	vector<vector<double> > x = { {1.0,1.0},{1.0,2.0},{1.0,3.0} };
	vector<double> y = { 3.0,5.0,7.0 };

	/*vector<double> theta = { 0.0,0.0 ,0.0,0.0,0.0 };
	vector<vector<double> > x = { { 1.0,2104.0,5.0,1.0,45.0 },{ 1.0,1416.0,3.0,2.0,40.0 },{ 1.0,1534.0,3.0,2.0,30.0 },{1.0,852.0,2.0,1.0,36.0} };
	vector<double> y = { 460.0,232.0,315.0,178.0 };*/

	double lastValue = costFunction(theta, x, y);

	int i = 0;
	while (true)
	{
		theta = gradientDescent(alpha, theta, x, y);
		if (abs(costFunction(theta, x, y) - lastValue) <= error)
		{
			cout << "iteration: " << i << endl;
			break;
		}
		else
		{
			lastValue = costFunction(theta, x, y);
			cout << i << endl;
			i++;
		}
	}
	cout << "theta: " << theta[0] << ", " << theta[1] << endl;

	system("pause");

	return 0;
}