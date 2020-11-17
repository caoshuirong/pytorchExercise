# Tensor,nn.module,F.relu……
import torch
import torch.nn as nn
import torch.nn.functional as F
# ETL
import torchvision
import torchvision.transforms as Transforms
from torch.utils.data import DataLoader
# 优化器
import torch.optim as optim
# 独热码
from sklearn.preprocessing import OneHotEncoder
# tensorboard
from torch.utils.tensorboard import SummaryWriter
# product
from itertools import product
from collections import OrderedDict,namedtuple
# image show
import numpy as np
from matplotlib import pyplot as plt
# display the network.named_parameters()
NEED_HISTOGRAM = True

# try the possible hyper-parameters
class RunBuilder():
    @staticmethod        # It can use the class name to call get_runs function
    def get_runs(params):
        Run = namedtuple('Run', params.keys())  # Define the class name and field name
        runs = []
        # the elements of params.values are [0.01 0.001] and [10,100,1000].The type of them is list.
        for v in product(*params.values()):     # use the elements of params.values to calculate the product
            runs.append(Run(*v))                # use the values of production to initial the subclass of tuple.
        return runs

# determine the value of hyper-parameters
params = OrderedDict(
    lr = [0.01, 0.001],
    batch_size = [10, 100, 1000],

)


#print(torch.__version__)

# 一、搭建网络
class Network(nn.Module):
    # 指定每一层连接权重的形状
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=12*4*4,out_features=120,bias=True)
        self.fc2 = nn.Linear(in_features=120, out_features=60, bias=True)
        self.out = nn.Linear(in_features=60, out_features=10, bias=True)

    def forward(self, t):
    # input layer
        t = t
    # conv1
        t = self.conv1(t)    # [B,C,24,24]
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size = 2,stride = 2) # [B,C,12,12]

    # conv2
        t = self.conv2(t)    # [B,C,8,8]
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # [B,C,4,4]
    # fc1
        t = t.reshape(-1,12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

    # fc2
        t = self.fc2(t)
        t = F.relu(t)

    # output layer
        t = self.out(t)
        # t = F.softmax(t,dim = 1)
        return t

# 每一批样本判断正确的个数
def get_num_correct(preds,labels):
    return torch.argmax(preds,dim=1).eq(labels).sum().item()

# 二、导入数据集
train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                              train=True,
                                            download=True,
                                              transform=Transforms.Compose([Transforms.ToTensor()]))

network = Network()
optimizer = optim.Adam(network.parameters(), lr= 0.01)
data_loader = DataLoader(train_set,batch_size=100)
images,labels = next(iter(data_loader))
grid = torchvision.utils.make_grid(images, nrow=10)
image_grid = np.transpose(grid,[1,2,0]) # [B C H W] -> [H W C]
# display image
plt.imshow(image_grid)
# keep the window
plt.show()
# tensorboard
tb = SummaryWriter()
# write images
tb.add_image('images',grid)
# write the nn structure
tb.add_graph(network,images)
# close the file
tb.close()

# 三、训练网络
for run in RunBuilder.get_runs(params):
    # 生成模型实例
    network = Network()
    # 指定优化器,learning rate
    optimizer = optim.Adam(network.parameters(), lr=run.lr)
    # batch_size
    data_loader = DataLoader(train_set, run.batch_size)
    # define the comment of tensorboard
    comment = f'-{run}'
    # create instance of tensorboard
    tb = SummaryWriter(comment=comment)

    # 循环训练
    for epoch in range(5):
        total_correct = 0   # 每一轮的总正确个数
        total_accuracy = 0  # 每一轮的总正确率
        total_loss = 0.0
        total_num = len(train_set)
        # 内循环
        for batch in data_loader:
            # 解包
            images,labels = batch
            # 对标签进行独热编码
            enc = OneHotEncoder(sparse=False)
            # 一个train_data含有多个特征，使用OneHotEncoder时，特征和标签都要按列存放, sklearn都要用二维矩阵的方式存放
            one_hot_labels = enc.fit_transform(
                labels.reshape(-1, 1))  # 如果不加 toarray() 的话，输出的是稀疏的存储格式，即索引加值的形式，也可以通过参数指定 sparse = False 来达到同样的效果

            # 预测
            preds = network(images)

            # 计算误差
            # loss = F.mse_loss(preds.type(torch.FloatTensor),torch.tensor(one_hot_labels,dtype = torch.float32),reduction="mean")

            # ****为什么梯度为None,因为我的标签是0~9的类别编号，并没有转换成独热码。分类问题拟合的是一个概率分布，就是在每一个位置的分布律。
            # 回归问题才是用具体的数值

            loss = F.cross_entropy(preds,labels) # 直接输入preds 因为crossEntropy内部封装了softMax，它需要把每一行的预测结果装换成对应的概率
            # 不同批次的遗留梯度清零
            optimizer.zero_grad()
            # 反向传播，计算导数
            loss.backward()
            # print(torch.max(network.conv1.weight.grad))

            # 更新权重
            optimizer.step()
            # accumulate the num of correct prediction
            total_correct += get_num_correct(preds,labels)
            # calculate each batch's total loss
            total_loss += loss.item()*images.shape[0]
        # calculate the accuracy of each epoch
        total_accuracy = total_correct / total_num
        # write to file
        tb.add_scalar('loss', total_loss, epoch)
        tb.add_scalar('num_of_correct',total_correct,epoch)
        tb.add_scalar('accuracy',total_accuracy,epoch)
        if NEED_HISTOGRAM:
            for name,weight in network.named_parameters():
                tb.add_histogram(name,weight,epoch)
                tb.add_histogram(f'{name}.grad',weight.grad,epoch)
            # show information
        string = 'epoch:%d loss = %f  accuracy = %f' % (epoch,loss.item(),total_accuracy)
        print(string)
    tb.close()

# 四、评价模型
from sklearn.metrics import confusion_matrix
from plotcm import plot_confusion_matrix
from matplotlib import pyplot as plt

def get_all_preds(model,data_loader):
    all_preds = torch.tensor([])
    with torch.no_grad():
        for batch in data_loader:
            images,labels = batch
            preds = model(images)
            all_preds = torch.cat((all_preds,preds),dim=0)
    return all_preds
train_preds = get_all_preds(network,data_loader)
# 生成混淆矩阵
cm = confusion_matrix(train_set.targets,train_preds.argmax(dim = 1))
# 绘制混淆矩阵
name = ('T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Akle boot')
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, name)