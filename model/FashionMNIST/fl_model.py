import utils.load_data as ld
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import logging


# 训练参数
lr = 0.005          # 学习率
momentum = 0.9      # 动量

# Cuda设置
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Generator(ld.Generator):
    """FashionMNIST数据集的生成器"""

    # 获取MINST训练集，测试集，标签
    def read(self, path):
        self.trainset = datasets.FashionMNIST(
            path, train=True, download=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.testset = datasets.FashionMNIST(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.labels = list(self.trainset.classes)


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        # 规定结构
        super(Net, self).__init__()
        # in_channels指输入通道，out_channels指输出通道，kernel_size指卷积核的大小，stride指步长
        # MINST数据集，图像大小为28*28，灰度图像的输入通道为1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=5, stride=1)
        # 两层全连接层
        self.fc1 = nn.Linear(4 * 4 * 32, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # 通过第一层卷积层，大小变为24*24，然后使用rulu激活
        x = F.relu(self.conv1(x))
        # 通过一个2*2的池化层，此时大小为12*12
        x = F.max_pool2d(x, 2, 2)
        # 通过第二层卷积层，大小变为8*8，然后使用rulu激活
        x = F.relu(self.conv2(x))
        # 通过一个2*2的池化层，此时大小为4*4
        x = F.max_pool2d(x, 2, 2)
        # 压扁，变成一维，才能连接全连接层
        x = x.view(-1, 4 * 4 * 32)
        # 通过两层全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 获取优化器，采用SGD
def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


# 训练
def train(model, train_loader, optimizer, epochs):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 获取输出，并计算loss
            output = model(image)
            loss = criterion(output, label)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()


# 测试
def test(model, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            # 将一批的损失相加
            test_loss += criterion(output, label).item()
            # 找到概率最大的下标
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    msg = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), test_acc)
    # 记录日志
    logging.info(msg)
    return test_acc, test_loss
