import math
import numpy as np

def norm_data(data):
    data.dtype = "float32"
    row, col = data.shape
    for j in range(col):
        min_value = data[:,j].min()
        max_value = data[:,j].max()
        for i in range(row):
            data[i][j] = (data[i][j] - min_value) / ((max_value - min_value))
    return data

def calculate_distance(x, y):
    dis = 0
    for index, value in enumerate(x):
        dis += (value - y[index])**2
    return math.sqrt(dis)

def calculate_distance2(data, kmean, label):
    row, col = data.shape

    for i in range(row):
        min_value = 0
        min_index = 0
        for k in kmean:
            t = calculate_distance(data[i], data[k])
            if (i != k):
                if (min_value == 0):
                    min_value = t
                else:
                    if (min_value > t):
                        min_value = t
                        min_value = k
        label[label.keys()[min_value]].append(min_value)
    return label

if __name__ == '__main__':
    x0 = [50, 50, 9]
    x1 = [28, 9, 4]
    x2 = [17, 15, 3]
    x3 = [25, 40, 5]
    x4 = [28, 40, 2]
    x5 = [50, 50, 1]
    x6 = [50, 40, 9]
    x7 = [50, 40, 9]
    x8 = [40, 40, 5]
    x9 = [50, 50, 9]
    x10 = [50, 50, 5]
    x11 = [50, 50, 9]
    x12 = [40, 40, 9]
    x13 = [40, 32, 17]
    x14 = [50, 50, 9]
    # conutry = ["china", "japan", "krea", "iran", "sadi", "iraq", "qatar", "uae", "uzb", "thailand", "vietnam", "amen", "bahrain", "north krea", "india"]

    data = np.array([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14])
    data = norm_data(data)
    label = {"A": [], "B": [], "C": []}
    kmean = [1, 9, 12]
    print(list(label.keys()))
    # l = calculate_distance2(data, kmean, label)
    # print(l)