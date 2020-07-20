import csv
import numpy as np
import random
class FL():
    data=[]
    size=0
    x=[]
    y=[]
    def __init__(self,fn):
        with open(fn, mode='r') as io:
            self.raw_data = csv.reader(io,delimiter =';')
            FL.data = list(self.raw_data)
            del io
            del self.raw_data
        FL.data.pop(0)
        FL.data=[list(map(float, i)) for i in FL.data]
        FL.size=len(FL.data)
    def prep(self):
        random.shuffle(FL.data)
        p1=int(FL.size*0.8)
        p2=int(FL.size*0.9)
        with open('train.csv','w',newline='') as train:
            write = csv.writer(train)
            write.writerows(FL.data[:p1])
        with open('test.csv','w',newline='') as test:
            write = csv.writer(test)
            write.writerows(FL.data[p1:p2])
        with open('validation.csv','w',newline='') as validate:
            write = csv.writer(validate)
            write.writerows(FL.data[p2:])