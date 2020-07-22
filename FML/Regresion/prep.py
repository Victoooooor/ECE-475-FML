import pandas as pd
import numpy as np
import random
class FL():
    data=[]
    size=0
    x=[]
    y=[]
    def __init__(self,fn):
        FL.data = pd.read_csv('1.csv',sep=';')
    def prep(self,yname=[]):
        FL.data = FL.data.sample(frac=1)
        if yname:
            inputx = list(set(FL.data.columns)-set([yname]))
            FL.data[inputx] = FL.data[inputx]/FL.data[inputx].max()
        FL.size = len(FL.data.index)
        p1=int(FL.size*0.8)
        p2=int(FL.size*0.9)
        FL.data[:p1].to_csv('train.csv',index=False,header=False)
        FL.data[p1:p2].to_csv('test.csv',index=False,header=False)
        FL.data[p2:].to_csv('validation.csv',index=False,header=False)