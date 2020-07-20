import csv
import numpy as np
import random
class LR():
    data=[]
    size=0
    x=[]
    y=[]
    test_x=[]
    test_y=[]
    def __init__(self,fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            LR.data = list(self.raw_data)
            del io
            del self.raw_data
        LR.data=[list(map(float, i)) for i in LR.data]
        LR.size=len(LR.data)

    def formulate(self,n):
        if np.size(LR.data[0])<=n:
            raise ValueError('Invalid index')
        self.y_index = n
        for i in LR.data:
            LR.y.append(i[n])
            LR.x.append([1]+i[0:n]+i[n+1:])

    def cal_beta(self):
        if not LR.x or not LR.y:
            raise ValueError('Empty list x, y')
        self.XM = np.matrix(LR.x)
        self.YM = np.matrix(LR.y).transpose()
        self.beta = np.linalg.inv(self.XM.transpose()*self.XM)*self.XM.transpose()*self.YM
        return self.beta
        
    def test(self, fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            LR.t_data = list(self.raw_data)
            del io
            del self.raw_data
        LR.t_data=[list(map(float, i)) for i in LR.t_data]
        LR.test_x=[]
        LR.test_y=[]
        for i in LR.t_data:
            LR.test_y.append(i[self.y_index])
            LR.test_x.append([1]+i[0:self.y_index]+i[self.y_index+1:])
        self.tX = np.matrix(LR.test_x)
        self.tY = np.matrix(LR.test_y).transpose()
        est = float(sum(np.square(self.tX * self.beta-self.tY))/len(LR.t_data))
        return est
