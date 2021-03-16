#python3.8_path = /usr/local/bin/python,/usr/local/lib/python3.8/lib-dynload
import torch
import torch.nn as nn
from FNet import Net
import torch.optim as optim
import os
import func

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data = func.getdata('track1_round1_train_20210222.csv')
# 此时feature格式为['|1 2 3|','|1 2 3|',...]
feature = data[1].tolist()
label = data[2].tolist()

#对feature进行类型转化，转化为[[1,2,3],[1,2,3],....]的形式
feature = func.feature_trans(feature)
#padding,将长度填充为104
feature = func.padding(feature)
#对label进行类型转化，转化为[[int，int],[],....]的形式
label = func.label_trans(label)

net = Net()

#训练参数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.01)#, momentum=0.01
# optimizer = optim.Adam(net.parameters(), lr=0.001, momentum=0.01)
# optimizer = optim.RMSprop(net.parameters(), lr=0.0001, momentum=0.01)

loss_function = nn.BCELoss() #nn.BCEWithLogitsLoss()

#训练
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(0,9998,10):
        # get the inputs
        inputs = torch.FloatTensor(feature)[i:i+10]
        labels = torch.FloatTensor(label)[i:i+10]

        outputs = net(inputs)
        #print(outputs.shape)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

#torch.save(net.state_dict(), 'my_base_model.pth')