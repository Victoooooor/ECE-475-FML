import csv
import numpy as np
import random
class RR():
    data=[]
    size=0
    x=[]
    y=[]
    test_x=[]
    test_y=[]
    def __init__(self,fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            RR.data = list(self.raw_data)
            del io
            del self.raw_data
        RR.data=[list(map(float, i)) for i in RR.data]
        RR.size=len(RR.data)

    def formulate(self,n):
        if np.size(RR.data[0])<=n:
            raise ValueError('Invalid index')
        self.y_index = n
        for i in RR.data:
            RR.y.append(i[n])
            RR.x.append(i[0:n]+i[n+1:])

    def normalize(self):
        self.standard=[]
        for i in range(0,len(RR.x[0])):
            self.standard.append(max(RR.x, key=lambda k: k[i])[i])
        RR.x = np.array(RR.x)/np.array(self.standard)
        RR.x = RR.x-self.x.mean(axis=0)
        RR.x = RR.x*10
        
    def cal_beta(self,lamda):
        self.XM = np.matrix(RR.x)
        self.YM = np.matrix(RR.y).transpose()
        self.beta = np.linalg.inv((self.XM.transpose()*self.XM)-lamda*np.identity(len(self.x[0])))*self.XM.transpose()*self.YM
        self.beta = np.insert(self.beta,0,self.YM.mean()).transpose()
        return self.beta

    def test(self, fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            RR.t_data = list(self.raw_data)
            del io
            del self.raw_data
        RR.t_data=[list(map(float, i)) for i in RR.t_data]
        RR.test_x=[]
        RR.test_y=[]
        for i in RR.t_data:
            RR.test_y.append(i[self.y_index])
            RR.test_x.append(i[0:self.y_index]+i[self.y_index+1:])
        RR.test_x = np.column_stack((np.ones(len(RR.test_x)).transpose(),RR.test_x))
        self.tX = np.matrix(RR.test_x)
        self.tY = np.matrix(RR.test_y).transpose()
        self.est = float(sum(np.square(self.tX * self.beta-self.tY))/len(RR.t_data))
        return self.est
