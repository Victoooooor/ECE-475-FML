import csv
from sklearn.linear_model import Lasso
from sklearn.linear_model import lasso_path
import numpy as np
from sklearn.metrics import mean_squared_error
class LoR():
    data=[]
    size=0
    vali_x=[]
    vali_y=[]
    def __init__(self,fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            LoR.data = list(self.raw_data)
            del io
            del self.raw_data
        LoR.data=[list(map(float, i)) for i in LoR.data]
        LoR.size=len(LoR.data)

    def h(self, row_x, theta):
        return 1/(1+np.exp(-np.dot(np.array(theta).transpose(),np.array(row_x))))
    
    def coef_SDG(self, y_ind, epoch, step_size):
        self.y_index = y_ind
        self.Llhood=[]
        LoR.x = []
        LoR.y = []
        for i in LoR.data:
            LoR.y.append(i[self.y_index])
            LoR.x.append(i[0:self.y_index]+i[self.y_index+1:])
        self.theta = [0.0 for i in range(len(LoR.x[0]))]
        for i in range(epoch):
            for j,q in enumerate(LoR.x):
                for k in range(len(LoR.x[0])):
                    self.theta[k] = self.theta[k] + step_size * (LoR.y[j]-self.h(q,self.theta))*q[k] 
            Likeli = 1
            for j,q in enumerate(LoR.x):
                ht = self.h(q,self.theta)
                Likeli = Likeli*(ht**self.y[j]*(np.ones(np.shape(ht))-ht)**(1-self.y[j]))
            self.Llhood.append(Likeli)
        
    def test(self, fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            LoR.t_data = list(self.raw_data)
            del io
            del self.raw_data
        LoR.t_data=[list(map(float, i)) for i in LoR.t_data]
        self.test_x = []
        self.test_y = []
        for i in LoR.t_data:
            self.test_y.append(i[self.y_index])
            self.test_x.append(i[0:self.y_index]+i[self.y_index+1:])
        self.test_yhat = []
        self.error_count = 0
        for j,q in enumerate(self.test_x):
            temp = 0 if self.h(q,self.theta) < 0.5 else 1
            self.test_yhat.append(temp)
            if not temp == self.test_y[j]:
                self.error_count += 1
        self.err_rate = self.error_count/len(self.test_x)
        return self.err_rate