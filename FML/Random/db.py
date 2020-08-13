import numpy as np
import pandas as pd

class db():
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
        header_name=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
        self.y_index = y_index
        db.train_data = pd.read_csv(trainf,header=None,names = header_name)
        db.test_data = pd.read_csv(testf,header=None,names = header_name)
        db.valid_data = pd.read_csv(validf,header=None,names = header_name)

        db.train_y=db.train_data[y_index]
        db.train_x=db.train_data.loc[:, db.train_data.columns != y_index]

        db.test_y=db.test_data[y_index]
        db.test_x=db.test_data.loc[:, db.test_data.columns != y_index]

        db.valid_y=db.valid_data[y_index]
        db.valid_x=db.valid_data.loc[:, db.valid_data.columns != y_index]

    