import numpy as np
import csv

def getdata(path):
    l= []
    with open(path,'r') as myFile:
        lines=csv.reader(myFile)
        for line in lines:
            l.append(line)
    data = np.array(l).T
    return data

def scalar2vector(input):
    vector = [0]*17
    for i in input:
        vector[i-1] = 1
    return vector

def label2vector(label_in):
    for i in range(len(label_in)):
        label_in[i] = scalar2vector(label_in[i])
    return label_in

#对feature进行类型转化，转化为[[int，int],[],....]的形式
def feature_trans(feature):
    for n in range(len(feature)):
        feature[n] = feature[n].lstrip('|').rstrip('|').split()
        for i in range(len(feature[n])):
            feature[n][i] = int(feature[n][i])
    return feature

def padding(feature):
    for _ in range(len(feature)):
        feature[_] += [858]*(104-len(feature[_]))
    return feature

#对label进行类型转化，转化为[[int，int],[],....]的形式
def label_trans(label):
    for n in range(len(label)):
        label[n] = label[n].lstrip('|').split()
        for i in range(len(label[n])):
            label[n][i] = int(label[n][i])
    label = label2vector(label)
    return label

def rerestrip(input):
    out = '|'
    for i in range(len(input)):
        out += str(input[i])
        out += ' '
    return out