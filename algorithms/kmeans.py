import numpy as np

# 距离函数
def distEclud(vecA,vecB):
    # vecA,vecB是数组形式，列表形式不行
    return sum((vecA-vecB)**2)**0.5

# 初次随机产生质心
def randCent(dataset,k):
    # dataset要聚类的数据集，k是聚类的个数
    n = np.shape(dataset)[1] # 列的个数
    centroids = np.mat(np.zeros([k,n])) # 质心的存储形式
    for i in range(n):
        maxi = max(dataset[:,i])
        mini = min(dataset[:,i])
        centroids[:,i] = mini + (maxi-mini)*np.random.random([k,1]) # 填充质心矩阵的第i列
    # n 次循环完毕，质心矩阵填充完成
    return centroids


def kmeans(dataset,k):
    m = np.shape(dataset)[0] # 样本的个数
    clusterAssment = np.mat(np.zeros((m,2))) # 保存每个样本的聚类情况，第一列表示该样本属于某一类，第二列是与聚类中心的距离
    centroids = randCent(dataset,k) # 调用函数产生随机质心
    clusterChanged = True # 控制聚类算法迭代停止的标志，当聚类不再改变时，就停止迭代
    while clusterChanged:  
        clusterChanged = False # 先进行本次迭代，如果聚类还是改变，最后把该标志改为True，从而继续下一次迭代
        for i in range(m): # 遍历每一个样本
            # 每个样本与每个质心计算距离
            # 采用一趟冒泡排序找出最小的距离，并找出对应的类
            # 计算与质心的距离时，刚开始需要比较，记为无穷大
            mindist = np.inf
            for j in range(k): # 遍历每一类
                distj = distEclud(dataset[i,:],centroids[j,:].A[0])
                if distj<mindist:
                    mindist = distj
                    minj = j
            # 遍历完k个类，本次样本已聚类
            if clusterAssment[i,0] !=minj:  # 判断本次聚类结果和上一次是否一致
                clusterChanged = True   # 只要有一个聚类结果改变，就重新迭代
            clusterAssment[i,:] = minj,mindist**2  # 类别，与距离
        # 外层循环结束，每一个样本都有了聚类结果
        
        # 更新质心
        for cent in range(k):
            # 找出属于相同一类的样本
            data_cent = dataset[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = np.mean(data_cent,axis=0)
    return centroids,clusterAssment

if __name__ == '__main__':
    dataset = np.random.randint(1,20,[20,5])
    k = 3
    centroids, clusterAssment = kmeans(dataset, k)
    print(clusterAssment)