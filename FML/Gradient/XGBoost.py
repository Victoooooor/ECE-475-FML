import numpy as np
import xgboost
import pandas as pd
from sklearn.metrics import mean_squared_error
class XGB():
    test_data=[]
    test_x=[]
    test_y=[]
    train_data=[]
    train_x=[]
    train_y=[]
    valid_data=[]
    valid_x=[]
    valid_y=[]
    def __init__(self,trainf, testf, validf, y_index):
        self.y_index = y_index
        XGB.train_data = pd.read_csv(trainf,header=None)
        XGB.test_data = pd.read_csv(testf,header=None)
        XGB.valid_data = pd.read_csv(validf,header=None)
        XGB.train_y=XGB.train_data[y_index]
        XGB.train_x=XGB.train_data.loc[:, XGB.train_data.columns != y_index]
        self.train_Dmatrix = xgboost.DMatrix(data=self.train_x,label=self.train_y)

        XGB.test_y=XGB.test_data[y_index]
        XGB.test_x=XGB.test_data.loc[:, XGB.test_data.columns != y_index]
        self.test_Dmatrix = xgboost.DMatrix(data=self.test_x,label=self.test_y)

        XGB.valid_y=XGB.valid_data[y_index]
        XGB.valid_x=XGB.valid_data.loc[:, XGB.valid_data.columns != y_index]
        self.valid_Dmatrix = xgboost.DMatrix(data=self.valid_x,label=self.valid_y)



    def train_lambda(self, lamda_list, param,round):
        self.rate = []
        for i in lamda_list:
            param['lambda'] = i
            temp = xgboost.train(param, self.train_Dmatrix, round)
            yhat = temp.predict(self.valid_Dmatrix)
            for i,j in enumerate(yhat):
                if j<0.67:
                    yhat[i] = 0
                elif j<1.33:
                    yhat[i] = 1
                else:
                    yhat[i] = 2
            self.rate.append(sum(self.valid_y!=yhat)/len(yhat))

    def base(self, lamda, param,round):
        param['lambda'] = lamda
        temp = xgboost.train(param, self.train_Dmatrix, round)
        self.yhat = temp.predict(self.train_Dmatrix)
        for i,j in enumerate(self.yhat):
            if j<0.67:
                self.yhat[i] = 0
            elif j<1.33:
                self.yhat[i] = 1
            else:
                self.yhat[i] = 2
        
        return sum(self.train_y!=self.yhat)/len(self.yhat)

    def test(self, lamda, param,round):
        param['lambda'] = lamda
        temp = xgboost.train(param, self.train_Dmatrix, round)
        self.yhat = temp.predict(self.test_Dmatrix)
        self.importance = temp.get_score(importance_type='gain')
        for i,j in enumerate(self.yhat):
            if j<0.67:
                self.yhat[i] = 0
            elif j<1.33:
                self.yhat[i] = 1
            else:
                self.yhat[i] = 2
        
        return sum(self.test_y!=self.yhat)/len(self.yhat)