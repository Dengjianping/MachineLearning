import math

def normlize(theta=[], x=[]):
    sum = 0
    for index, value in enumerate(theta):
        sum += theta[index]*x[index]
    return sum
    
def costFunction(number, theta=[], x=[[]], y=[]):
    sum = 0
    for index, value in enumerate(y):
        t = normlize(theta=theta, x=x[index])
        sum += (t-y[index])*(t-y[index])
        
    return 0.5*sum/number
    
def partialDifferential(number, subindex=0, y=[], theta=[], x=[[]]):
    sum = 0
    for index, value in enumerate(y):
        t = normlize(theta=theta, x=x[index])
        sum += x[index][subindex]*(t-y[index])
    return sum/number
    
def learningRate(number, alpha=0.3, theta=[], x=[[]], y=[]):
    lastTheta = []
    for index, value in enumerate(theta):
        lr = partialDifferential(number, subindex=index, y=y, theta=theta, x=x)
        t = theta[index] - alpha*lr
        lastTheta.append(t)
    return lastTheta
    
if __name__ == '__main__':
    rate = 0.3
    # suppose this linear equation is y = 1 + 2*x
    x = [[1, 1], [1, 2], [1, 3]]
    y = [3, 5, 7]
    theta = [0,0]
    number = len(y)
    error = 0.0001
    lastValue = costFunction(number, theta=theta, x=x, y=y)
    
    for i in range(10000):
        theta = learningRate(number, theta=theta, x=x, y=y)
        if (math.fabs(costFunction(number, theta=theta, x=x,y=y)-lastValue) <= error):
            print i
            break
        else:
            lastValue = costFunction(number, theta=theta, x=x,y=y)
            
    print lastValue
    print theta