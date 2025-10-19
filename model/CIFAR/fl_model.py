import utils.load_data as ld
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import logging


# 训练参数
lr = 0.005       # 学习率
momentum = 0.9  # 动量

# Cuda设置
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Generator(ld.Generator):
    """CIFAR10数据集的生成器"""

    # 获取CIFAR10训练集，测试集，标签
    def read(self, path):
        self.trainset = datasets.CIFAR10(
            path, train=True, download=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        self.testset = datasets.CIFAR10(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        self.labels = list(self.trainset.classes)


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        # 规定结构
        super(Net, self).__init__()
        # in_channels指输入通道，out_channels指输出通道，kernel_size指卷积核的大小，stride指步长
        # CIFAR-10数据集，图像大小为32*32，图像的输入通道为3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # 三层全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 通过第一层卷积层，大小变为28*28，然后使用rulu激活
        x = F.relu(self.conv1(x))
        # 通过一个2*2的池化层，此时大小为14*14
        x = F.max_pool2d(x, 2, 2)
        # 通过第二层卷积层，大小变为10*10，然后使用rulu激活
        x = F.relu(self.conv2(x))
        # 通过一个2*2的池化层，此时大小为5*5
        x = F.max_pool2d(x, 2, 2)
        # 压扁，变成一维，才能连接全连接层
        x = x.view(-1, 16 * 5 * 5)
        # 通过三层全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 获取优化器，采用SGD
def get_optimizer(model, lr_rate=1):
    return optim.SGD(model.parameters(), lr * lr_rate, momentum=momentum)


# 训练
def train(model, train_loader, optimizer, epochs):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

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
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels).item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / total
    msg = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), test_acc)
    # 记录日志
    logging.info(msg)
    return test_acc, test_loss
