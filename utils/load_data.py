# import logging
import random
from . import distribution


class Generator(object):
    """生成联邦学习的训练集和测试集"""

    # 抽象函数，读取数据集
    def read(self, path):
        # 读取数据集
        raise NotImplementedError

    # 将数据按标签分组
    def group(self):
        # 创建标签的空字典
        grouped_data_idxs = {label: [] for label in self.labels}

        # 为每个图片添加一个序号idx
        idx = 0
        # 把idx根据标签分组放进字典
        for _, label in self.trainset:
            # 提取标签
            label = self.labels[label]
            # 将对应的数据放入对应标签的字典
            grouped_data_idxs[label].append(idx)
            idx += 1

        self.grouped_data_idxs = grouped_data_idxs

    # 产生数据，并返回训练集
    def generate(self, dataset_path, sample_generate_path=''):
        self.read(dataset_path)
        # 获取训练集的大小
        self.trainset_size = len(self.trainset)
        self.sample_generate_path = sample_generate_path
        # 如果未给定生成路径，则将数据分成10组
        if(sample_generate_path == ''):
            self.group()

        return self.trainset


class Loader(object):
    """载入IID数据分布"""

    def __init__(self, generator):
        # 从generator获取数据
        self.trainset = generator.trainset
        self.testset = generator.testset
        self.labels = generator.labels
        self.trainset_size = generator.trainset_size
        self.sample_generate_path = generator.sample_generate_path
        self.grouped_data_idxs = generator.grouped_data_idxs

        # 将使用过的数据分开存储
        self.used_idxs = {label: [] for label in self.labels}

        # 记录自己的类型
        self.type = 'basic'

    def extract(self, label, n):
        if len(self.grouped_data_idxs[label]) > n:
            # 获取对应样本的idx
            extracted = self.grouped_data_idxs[label][:n]

            self.used_idxs[label].extend(extracted)  # 将这些数据移到已经使用
            del self.grouped_data_idxs[label][:n]  # 将这些数据从训练集删除

            return extracted
        else:
            print('Insufficient data in label: {}'.format(label))
            print('Dumping used data for reuse')

            # 将已使用的数据放回训练集
            for label in self.labels:
                self.grouped_data_idxs[label].extend(self.used_idxs[label])
                self.used_idxs[label] = []

            # 重新提取数据
            return self.extract(label, n)

    def get_partition(self, partition_size):
        # 从所有标签获取一个分区

        # 获取均匀分布
        dist = distribution.uniform(partition_size, len(self.labels))

        # 根据分布提取数据
        partition_idxs = []
        for i, label in enumerate(self.labels):
            idxs = self.extract(label, dist[i])
            partition_idxs.extend(idxs)

        # 打乱数据分区
        random.shuffle(partition_idxs)

        return partition_idxs

    # 根据文件路径获取数据
    def load_data_from_path(self, data_idxs):
        data = []
        for i in data_idxs:
            if(i == -1):
                break
            data.append(self.trainset[i])
        return data

    def get_testset(self):
        # 返回测试集
        return self.testset


class BiasLoader(Loader):
    """载入有偏好的数据分布，即非IID"""
    # 继承了Loader，重写get_partition即可
    def __init__(self, generator, bias):
        # 从generator获取数据
        self.trainset = generator.trainset
        self.testset = generator.testset
        self.labels = generator.labels
        self.trainset_size = generator.trainset_size
        self.sample_generate_path = generator.sample_generate_path
        self.grouped_data_idxs = generator.grouped_data_idxs

        # 将使用过的数据分开存储
        self.used_idxs = {label: [] for label in self.labels}

        # 记录自己的类型
        self.type = 'bias'

        # bias指主类的比例
        # secondary指是否只有两个类，true指的是仅有两个类，false指所有类都有（未用到）
        self.bias = bias
        self.secondary = False

    def get_partition(self, partition_size, pref):
        # 获取有偏好的非IID分布

        # 计算主类的数量以及剩余数量
        majority = int(partition_size * self.bias)
        minority = partition_size - majority

        # 计算剩余标签数
        len_minor_labels = len(self.labels) - 1

        if self.secondary:
            # 从剩下九个标签随机获取一个，获取数据
            dist = [0] * len_minor_labels
            dist[random.randint(0, len_minor_labels - 1)] = minority
        else:
            # 从剩余所有类型标签获取数据分布
            dist = distribution.uniform(minority, len_minor_labels)

        # 添加主类标签数据
        dist.insert(self.labels.index(pref), majority)

        # 根据分布提取数据
        partition_idxs = []
        for i, label in enumerate(self.labels):
            idxs = self.extract(label, dist[i])
            partition_idxs.extend(idxs)

        # 打乱数据分区
        random.shuffle(partition_idxs)

        return partition_idxs
