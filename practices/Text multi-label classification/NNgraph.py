import tensorwatch as tw
from FNet import Net

net = Net()
tw.draw_model(net, [1, 104])