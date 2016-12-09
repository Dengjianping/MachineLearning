import os
import random
import unittest

class Matrix:
    def __init__(self, row, col, zero=False):
        self.row = row
        self.col = col
        self.data = [[0 for i in range(self.col)] for j in range(self.row)] if zero else [[random.randint(1, 10) for i in range(self.col)] for j in range(self.row)]
        
    @property
    def data(self):
        return self.data
        
    @property
    def row(self):
        return self.row
        
    def __len__(self):
        # return "{0.row} * {0.col}".format(self)
        return self.row * self.col
        
    @property
    def col(self):
        return self.col
        
    @data.setter
    def data(self, row, col, value):
        if not (0<=row<=self.row and 0<=col<=self.col and isinstance(value,int)):
            raise TypeError("0 <= row <= {0.row} and 0 <= col <= {0.col}, value is int type".format(self))
        self.data[row][col] = value

def convArray(kernel, target):
    result = [0 for i in range(len(kernel) + len(target) - 1)]
    for i, t in enumerate(target):
        for j, k in enumerate(kernel):
            result[i+j] += k * t
    return result
    
def convMatrix(kernel, target):
    result = Matrix(kernel.row+target.row - 1, target.col+kernel.col-1, True)
    for i in range(target.row):
        for j in range(target.col):
            for m in range(kernel.row):
                for n in range(kernel.col):
                    result.data[i+m][j+n] += kernel.data[m][n] * target.data[i][j]
    return result

class TestConvolution(unittest.TestCase):
    def testConvArray(self):
        kernel = [1,2]
        target = [4,7]
        self.assertEquals([4,15,14], convArray(kernel, target))
        
    def testConvMatrix(self):
        kernel = Matrix(2,2)
        kernel.data = [[1,2],[3,4]]
        target = Matrix(3,3)
        target.data = [[1,2,3],[3,4,5],[5,6,7]]
        result = [[1,4,7,6],[6,20,30,22],[14,40,50,34],[15,38,45,28]]
        self.assertEquals(result, convMatrix(kernel, target).data)
    
if __name__ == '__main__':
    kernel = [1,2,3]
    target = [3,4,5,6]
    print convArray(kernel, target)
    
    m = Matrix(3,4)
    n = Matrix(5,7)
    
    result = convMatrix(m, n)
    # print result.data
    print len(result)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConvolution)
    unittest.TextTestRunner(verbosity=2).run(suite)