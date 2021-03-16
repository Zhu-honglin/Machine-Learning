import torch
import torch.nn as nn
from FNet import Net
import torch.optim as optim
import os
import func
import torch.nn.functional as F

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device_ids=range(torch.cuda.device_count())

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

x = torch.arange(0, 104, 1).reshape(1,104).to(device)#.to(device)

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
        return F.max_pool1d(x, kernel_size=x.shape[2]) # shape: (batch_size, channel, 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = nn.Embedding(859, 64)
        self.dropout = nn.Dropout(0.5)
        self.pool = GlobalMaxPool1d()
        self.cov1 = nn.Conv2d(1,2,(64,2))
        self.cov2 = nn.Conv2d(1,2,(64,3))
        self.cov3 = nn.Conv2d(1,2,(64,4))
        self.Linear = nn.Linear(2*3, 17)
    def forward(self, x):
        x = self.embed(x).reshape(1,1,64,104)
        c1 = self.pool(F.relu(self.cov1(x)).reshape(1,2,103))
        c2 = self.pool(F.relu(self.cov2(x)).reshape(1,2,102))
        c3 = self.pool(F.relu(self.cov3(x)).reshape(1,2,101))
        cat = torch.cat((c1,c2,c3), 1).reshape(1,6)
        d = self.dropout(cat)
        out = torch.sigmoid(self.Linear(d))
        return out

net = Net().to(device)

print(net(x).shape)

#训练参数
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0)#, momentum=0.01
# optimizer = nn.DataParallel(optimizer)
loss_function = nn.BCELoss() #nn.BCEWithLogitsLoss()

#训练
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(1,9998,1):
        # get the inputs
        inputs = torch.LongTensor(feature)[i:i+1].to(device)
        labels = torch.FloatTensor(label)[i:i+1].to(device)
        # print(inputs.shape)
        # print(inputs.dtype)
        outputs = net(inputs)
        # print('outputs.dtype:')
        # print(outputs.dtype)
        # print('labels.dtype:')
        # print(labels.dtype)

        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 99 == 0:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')