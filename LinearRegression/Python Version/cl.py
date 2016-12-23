import math

class LinearRegression:
    def __init__(self, trainX, trainY, learnReate, precision):
        self.trainX = trainX
        self.trainY = trainY
        self.learnReate = learnReate
        self.precision = precision
        self.theta = [0 for i in range(len(self.trainX[0]))]
        
    @property
    def theta(self):
        self.theta
        
    def hypothesis(self, x=[]):
        hypo = 0
        for index, value in enumerate(self.theta):
            hypo += value * x[index]
        return hypo
        
    def costFuntion(self):
        cost = 0
        for index, y in enumerate(self.trainY):
            t = self.hypothesis(self.trainX[index]) - self.trainY[index]
            cost += math.pow(t, 2)
        return 0.5 * cost / len(self.trainY)
        
    def partialDerivation(self, subIndex):
        length = len(self.trainY)
        partial = 0
        for index, y in enumerate(self.trainY):
            t = self.hypothesis(self.trainX[index]) - y
            partial += t * self.trainX[index][subIndex]
        return partial / length
        
    def train(self):
        # update theta with batch gradient descent method
        lastCost = self.costFuntion()
        print lastCost
        epoch = 0
        while (True):
            epoch += 1
            for index, value in enumerate(self.theta):
                self.theta[index] -= self.partialDerivation(index) * self.precision
            if (math.fabs(self.costFuntion() - lastCost) <= self.precision):
                print epoch
                break
            lastCost = self.costFuntion()
        print self.costFuntion()

if __name__ == '__main__':
    x = [[1, 1], [1, 2], [1, 3]]
    y = [6, 10, 14]
    precision = 1e-5
    learnReate = 0.3
    lg = LinearRegression(x, y, learnReate, precision)
    lg.train()
    print lg.theta
