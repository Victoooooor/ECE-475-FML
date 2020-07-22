import csv
from sklearn.linear_model import Lasso
from sklearn.linear_model import lasso_path
import numpy as np
from sklearn.metrics import mean_squared_error
class LaR():
    data=[]
    size=0
    x=[]
    y=[]
    vali_x=[]
    vali_y=[]
    def __init__(self,fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            LaR.data = list(self.raw_data)
            del io
            del self.raw_data
        LaR.data=[list(map(float, i)) for i in LaR.data]
        LaR.size=len(LaR.data)

    def formulate(self,n):
        if np.size(LaR.data[0])<=n:
            raise ValueELaRor('Invalid index')
        self.y_index = n
        for i in LaR.data:
            LaR.y.append(i[n])
            LaR.x.append(i[0:n]+i[n+1:])
    
    def lasso_train(self, fn, array = None):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            LaR.t_data = list(self.raw_data)
            del io
            del self.raw_data
        LaR.t_data=[list(map(float, i)) for i in LaR.t_data]
        LaR.vali_x = []
        LaR.vali_y = []
        for i in LaR.t_data:
            LaR.vali_y.append(i[self.y_index])
            LaR.vali_x.append(i[0:self.y_index]+i[self.y_index+1:])
        self.beta = []
        self.alpha = list(array)
        self.MSE = 100
        self.best = None
        self.LassR = Lasso()
        for i in array:
            self.LassR.set_params(alpha = i)
            self.LassR.fit(LaR.x,LaR.y)
            self.beta.append(self.LassR.coef_)
            self.predict = self.LassR.predict(LaR.vali_x)
            temp = mean_squared_error(LaR.vali_y,self.predict)
            if temp < self.MSE:
                self.MSE = temp
                self.best = i
        #self.alpha, self.beta,_  = lasso_path(LaR.x, LaR.y, eps=0.0001, positive=False, fit_intercept=True)
        #self.alpha, self.beta,_  = lasso_path(LaR.x, LaR.y, alphas=array, positive=False, fit_intercept=True)
        
    def test(self, fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io)
            LaR.t_data = list(self.raw_data)
            del io
            del self.raw_data
        LaR.t_data=[list(map(float, i)) for i in LaR.t_data]
        self.test_x = []
        self.test_y = []
        for i in LaR.t_data:
            self.test_y.append(i[self.y_index])
            self.test_x.append(i[0:self.y_index]+i[self.y_index+1:])
        self.LassR.set_params(alpha = self.best)
        self.LassR.fit(LaR.x,LaR.y)
        self.predict = self.LassR.predict(self.test_x)
        return mean_squared_error(self.test_y,self.predict)
