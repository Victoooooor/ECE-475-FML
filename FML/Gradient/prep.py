import pandas as pd
import numpy as np
import random
class FL():
    data=[]
    size=0
    x=[]
    y=[]
    def __init__(self,fn):
        FL.data = pd.read_csv(fn,sep=',',header=None)
    def prep(self,yname=[], train_name = 'train.csv', test_name = 'test.csv', validation_name = 'validation.csv'):
        FL.data = FL.data.sample(frac=1)
        if yname:
            inputx = list(set(FL.data.columns)-set([yname]))
            FL.data[inputx] = FL.data[inputx]/FL.data[inputx].max()
        FL.size = len(FL.data.index)
        diction = {'M': 1, 'I': 0, 'F': 2}
        FL.data[yname].replace(diction, inplace=True)
        p1=int(FL.size*0.8)
        p2=int(FL.size*0.9)
        FL.data[:p1].to_csv(train_name,index=False,header=False)
        FL.data[p1:p2].to_csv(test_name,index=False,header=False)
        FL.data[p2:].to_csv(validation_name,index=False,header=False)