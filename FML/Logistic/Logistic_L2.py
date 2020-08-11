import csv
from sklearn.linear_model import Lasso
from sklearn.linear_model import lasso_path
import numpy as np
from sklearn.metrics import mean_squared_error
class LoR_L2():
    data=[]
    size=0
    def __init__(self,fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            LoR_L2.data = list(self.raw_data)
            del io
            del self.raw_data
        LoR_L2.data=[list(map(float, i)) for i in LoR_L2.data]
        LoR_L2.size=len(LoR_L2.data)

    def h(self, row_x, theta):
        return 1/(1+np.exp(-np.dot(np.array(theta).transpose(),np.array(row_x))))
    
    def coef_SDG(self, y_ind, epoch, step_size, lamda):
        self.y_index = y_ind
        self.Llhood=[]
        self.theta = []
        LoR_L2.x = []
        LoR_L2.y = []
        for i in LoR_L2.data:
            LoR_L2.y.append(i[self.y_index])
            LoR_L2.x.append(i[0:self.y_index]+i[self.y_index+1:])
        
        for ind, lam in enumerate(lamda):
            self.lamda = lamda
            self.Llhood.append([])
            temp_theta = [0.0 for i in range(len(LoR_L2.x[0]))]
            for i in range(epoch):
                for j,q in enumerate(LoR_L2.x):
                   for k in range(len(LoR_L2.x[0])):
                        temp_theta[k] = temp_theta[k] + step_size * (LoR_L2.y[j]-self.h(q,temp_theta))*q[k] - 2 * lam * temp_theta[k]
                Likeli = 1
                for j,q in enumerate(LoR_L2.x):
                    ht = self.h(q,temp_theta)
                    Likeli = Likeli*(ht**self.y[j]*(np.ones(np.shape(ht))-ht)**(1-self.y[j]))
                self.Llhood[ind].append(Likeli)
            self.theta.append(temp_theta)

    def validate(self, fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            LoR_L2.t_data = list(self.raw_data)
            del io
            del self.raw_data
        LoR_L2.t_data=[list(map(float, i)) for i in LoR_L2.t_data]
        self.vali_x = []
        self.vali_y = []
        for i in LoR_L2.t_data:
            self.vali_y.append(i[self.y_index])
            self.vali_x.append(i[0:self.y_index]+i[self.y_index+1:])
        self.vali_yhat = []
        self.error = np.zeros(len(self.theta))
        for j,q in enumerate(self.theta):
            self.vali_yhat.append([])
            for i,p in enumerate(self.vali_x):
                self.error[j] += np.abs(self.h(p,q)-self.vali_y[i])
                self.vali_yhat[j].append(self.h(p,q))
        self.error = self.error.tolist()
        self.best_lambda = self.error.index(min(self.error))
        return self.best_lambda

    def test(self, fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            LoR_L2.t_data = list(self.raw_data)
            del io
            del self.raw_data
        LoR_L2.t_data=[list(map(float, i)) for i in LoR_L2.t_data]
        self.test_x = []
        self.test_y = []
        for i in LoR_L2.t_data:
            self.test_y.append(i[self.y_index])
            self.test_x.append(i[0:self.y_index]+i[self.y_index+1:])
        self.test_yhat = []
        self.error_count = 0
        for j,q in enumerate(self.test_x):
            temp = 0 if self.h(q,self.theta[self.best_lambda]) < 0.5 else 1
            self.test_yhat.append(temp)
            if not temp == self.test_y[j]:
                self.error_count += 1
        self.err_rate = self.error_count/len(self.test_x)
        return self.err_rate