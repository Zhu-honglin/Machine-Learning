import torch
import csv
import torch.nn as nn
from FNet import Net
import func

data = func.getdata('track1_round1_testA_20210222.csv')

# 此时feature格式为['|1 2 3|','|1 2 3|',...]
feature = data[1].tolist()
#对feature进行类型转化，转化为[[1,2,3],[1,2,3],....]的形式
feature = func.feature_trans(feature)
#padding,将长度填充为104
feature = func.padding(feature)

#获取网络模型
model = Net()
model.load_state_dict(torch.load('my_base_model.pth'))

#得到输出
output = model(torch.FloatTensor(feature))
print(output.shape)

#将结果保存需要的格式
with open('prediction.csv','w') as myFile:
    myWriter=csv.writer(myFile) #quoting=3,escapechar='$'
    for i in range(output.shape[0]):
        l = []
        l.append(str(i)+'|')
        l.append(func.rerestrip(output[i].tolist()))
        myWriter.writerow(l)
    print('Finished Writer')