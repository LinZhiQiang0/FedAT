from model import fl_model
import model.MNIST.fl_model as MNIST_model
import model.FashionMNIST.fl_model as FashionMNIST_model
import model.CIFAR.fl_model as CIFAR_model
import math
import numpy as np
from utils.setup import ClientSetup
# import logging


class Client():
    """模拟联邦学习客户端，AR模型模拟计算能力变化"""

    def __init__(self, client_id, data_type, compute_power, scale=0, bw=ClientSetup.bandwidth, compute_power_fluctuate_type=ClientSetup.compute_power_fluctuate_types[0], rounds=ClientSetup.max_round_new):
        self.client_id = client_id
        self.data_type = data_type
        # 计算资源波动场景下的时间预测
        self.bar_total_time = 0
        # 计算能力采用高斯分布模拟波动，初始计算能力为客户端的计算能力均值
        # 轮次
        self.rounds = rounds
        # 计算能力，初始计算能力为均值
        self.compute_power = compute_power
        self.curr_compute_power = compute_power
        self.compute_power_list = [self.compute_power]
        # 计算能力的标准差系数
        self.scale = scale * self.compute_power
        # 计算能力波动遵循的分布
        self.compute_power_fluctuate_type = compute_power_fluctuate_type
        # 带宽
        self.bw = bw
        # 训练时间，传输时间，总时间
        self.train_time = 0
        self.communication_time = 0
        self.total_time = 0

        # 选择的次数和初始队列
        self.num_of_choices = 0
        self.queue = 0

        # 一个epoch的时间，最近K个
        self.single_epoch_time_list = []
        # 初始化compute_power_list
        self.get_compute_power_list()

        # 定向选择相关
        self.score = 0
        self.last_selected_rd = 0
        # 计算时间列表，用来保存最近K次计算时间
        self.total_time_list = []

    # 设置非IID相关属性
    def set_bias(self, pref, bias):
        self.pref = pref
        self.bias = bias

    # 为客户端设置数据
    def set_data(self, data_idx, data):
        self.trainset_idx = data_idx
        self.trainset = data
        self.num_of_trainset = len(self.trainset)

        # 计算预测的total_time并返回该值
    def get_bar_total_time(self, rd, k=20):
        # 如果从未被选中，则bar_total_time最大为0
        if (len(self.total_time_list) == 0):
            return 0
        # 如果只要1个时间，则返回该值
        if (len(self.total_time_list) == 1):
            return self.total_time_list[0]

        while (len(self.total_time_list) > 1):
            if (rd - k <= self.total_time_list[0][0]):
                break
            self.total_time_list.pop(0)

        # 取均值
        self.bar_total_time = 0
        for [r, t] in self.total_time_list:
            self.bar_total_time += t
        self.bar_total_time /= len(self.total_time_list)

        # 保留4位小数
        self.bar_total_time = round(self.bar_total_time, 4)
        return self.bar_total_time

    # 设置客户端一些参数
    def configure(self):

        # 设置本地训练次数和批大小
        self.base_epoches = ClientSetup.epoches[self.data_type]
        self.current_epoches = self.base_epoches
        self.batch_size = ClientSetup.batch_size[self.data_type]

        # 系数清零
        # 计算能力
        self.curr_compute_power = self.compute_power
        self.compute_power_list = [self.compute_power]
        self.get_compute_power_list()
        # 训练时间，传输时间，总时间
        self.train_time = 0
        # 通信时间 = 3 * 模型大小 / 网速，上传速率假设小于下载速率
        upload_rate = round(self.bw * math.log(1 + 100, 2) / 8, 4)
        self.communication_time = 3 * ClientSetup.model_size[self.data_type] / upload_rate
        self.total_time = 0
        # 选择的次数
        self.num_of_choices = 0
        # 储备的epoch次数
        self.vqueue = 0
        # 一个epoch的时间，最近K个
        self.single_epoch_time_list = []
        # 定向选择相关
        self.score = 0
        self.last_selected_rd = 0
        # 计算时间列表，用来保存最近K次计算时间
        self.total_time_list = []

        # 设置模型
        if(self.data_type == ClientSetup.data_types[0]):
            self.model = MNIST_model.Net()
        elif(self.data_type == ClientSetup.data_types[1]):
            self.model = FashionMNIST_model.Net()
        elif(self.data_type == ClientSetup.data_types[2]):
            self.model = CIFAR_model.Net()
        self.model.eval()

        # 设置优化器
        #self.optimizer_list = []
        #for lr_rate in range(6):
        #    if(self.data_type == ClientSetup.data_types[0]):
        #        self.optimizer_list.append(MNIST_model.get_optimizer(self.model, 1 - 0.05 * lr_rate))
        #    elif(self.data_type == ClientSetup.data_types[1]):
        #        self.optimizer_list.append(FashionMNIST_model.get_optimizer(self.model, 1 - 0.05 * lr_rate))
        #    elif(self.data_type == ClientSetup.data_types[2]):
        #        self.optimizer_list.append(CIFAR_model.get_optimizer(self.model, 1 - 0.05 * lr_rate))
        #self.optimizer = self.optimizer_list[0]
        if(self.data_type == ClientSetup.data_types[0]):
            self.optimizer = MNIST_model.get_optimizer(self.model)
        elif(self.data_type == ClientSetup.data_types[1]):
            self.optimizer = FashionMNIST_model.get_optimizer(self.model)
        elif(self.data_type == ClientSetup.data_types[2]):
            self.optimizer = CIFAR_model.get_optimizer(self.model)



# 计算本次训练的epoches
    def get_current_epoches(self, rd, max_time=0, has_max=True, server_min_epoch=3):
        self.get_new_curr_compute_power(rd)
        if(max_time == 0):
            return
        # 求实际应该有的epoches
        time = max_time - self.communication_time
        # 求一个epoch的时间
        single_epoch_time = self.num_of_trainset / self.curr_compute_power
        self.single_epoch_time_list.append((rd, single_epoch_time))
        # epoches = 计算时间 / 一个epoch的时间
        new_epoches = int(time // single_epoch_time)
        e_rate = 2
        if(has_max):
            new_epoches = min(new_epoches, math.ceil(e_rate * self.base_epoches))
            # new_epoches = min(new_epoches, self.base_epoches + server_min_epoch)
        # 至少为2
        new_epoches = max(new_epoches, 2)
        self.current_epoches = new_epoches
    
    # 计算本次训练的单个epoch的时间
    def get_current_single_epoch_time(self, rd):
        self.get_new_curr_compute_power(rd)

        # 求一个epoch的时间
        single_epoch_time = self.num_of_trainset / self.curr_compute_power
        self.single_epoch_time_list.append((rd, single_epoch_time))

    # 更新base_epoches
    def update_base_epoches(self, base_epoches):
        self.base_epoches = base_epoches

    # 更新epoch虚拟队列
    def update_current_epoches_vqueue(self):
        self.vqueue = max(self.vqueue - self.current_epoches, 1)

    # 训练
    def train(self, adapt_train_flag=False, server_min_epoch=3, server_max_epoch=5):
        # 调整学习率
        if(adapt_train_flag == True):
            lr_rate = 1
            if(server_min_epoch != self.base_epoches and self.current_epoches >= server_max_epoch):
                lr_rate -= 0.1 * (self.current_epoches - self.base_epoches)
                # lr_rate -= 0.05 * (self.current_epoches - self.base_epoches)
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = 0.005 * lr_rate
            #if(self.data_type == ClientSetup.data_types[0]):
            #    self.optimizer = MNIST_model.get_optimizer(self.model, lr_rate)
            #elif(self.data_type == ClientSetup.data_types[1]):
            #    self.optimizer = FashionMNIST_model.get_optimizer(self.model, lr_rate)
            #elif(self.data_type == ClientSetup.data_types[2]):
            #    self.optimizer = CIFAR_model.get_optimizer(self.model, lr_rate)


            #if(server_min_epoch != self.base_epoches and self.current_epoches >= server_max_epoch):
            #    idx = self.current_epoches - self.base_epoches
            #    self.optimizer = self.optimizer_list[idx]
            #else:
            #    self.optimizer = self.optimizer_list[0]

        # 执行模型训练
        trainloader = fl_model.get_trainloader(self.trainset, self.batch_size)
        if(self.data_type == ClientSetup.data_types[0]):
            MNIST_model.train(self.model, trainloader, self.optimizer, self.current_epoches)
        elif(self.data_type == ClientSetup.data_types[1]):
            FashionMNIST_model.train(self.model, trainloader, self.optimizer, self.current_epoches)
        elif(self.data_type == ClientSetup.data_types[2]):
            CIFAR_model.train(self.model, trainloader, self.optimizer, self.current_epoches)



    # 测试
    def test(self, testloader):

        # 执行模型训练
        if(self.data_type == ClientSetup.data_types[0]):
            return MNIST_model.test(self.model, testloader)
        elif(self.data_type == ClientSetup.data_types[1]):
            return FashionMNIST_model.test(self.model, testloader)
        elif(self.data_type == ClientSetup.data_types[2]):
            return CIFAR_model.test(self.model, testloader)

    # 更新权重
    def load_weights(self, updated_weights):
        fl_model.load_weights(self.model, updated_weights)

    # 获取通讯内容
    def get_report(self):
        self.report = Report(self)
        # 获取权重
        self.report.weights = fl_model.extract_weights(self.model)
        return self.report

    def get_normal(self, low=-0.5, high=0.5, N=ClientSetup.max_round_new):
        e_list = []
        while(len(e_list) < N):
            tmp = np.random.normal(loc=0, scale=high/2, size=None)
            if(tmp >= low and tmp <= high):
                e_list.append(tmp)

        return e_list

    # 根据AR模型生成数据
    def get_data(self, min_comp=0, max_comp=1, N=ClientSetup.max_round_new):
        # 设定AR模型的阶数和系数
        ar_p = 2
        ar_pa = np.array([0.2, 0.8])

        # 生成随机数列e
        low = -0.5
        high = 0.5
        if(self.compute_power_fluctuate_type == ClientSetup.compute_power_fluctuate_types[0]):
            e = self.get_normal(low=low, high=high, N=N)
        else:
            e = [np.random.uniform(low=low, high=high) for i in range(N)]
        
        # 生成AR模型的数据序列data
        data = np.zeros(N)
        data[:ar_p] = e[:ar_p]
        for i in range(ar_p, N):
            data[i] = ar_pa.dot(data[i-ar_p:i][::-1]) + e[i]

        # 将data序列缩放到指定范围
        data = (data - data.min()) / (data.max() - data.min()) * (max_comp - min_comp) + min_comp
        return data

    # 生成compute_power_list
    def get_compute_power_list(self):
        # 两种抽取方式：截尾高斯分布和均匀分布
        # loc为平均计算能力，scale为标准差，保留4位小数
        # 保证抽样的值在[loc - 2*scale, loc + 2*scale]
        min_comp = self.compute_power - 2 * self.scale
        max_comp = self.compute_power + 2 * self.scale

        self.compute_power_list.extend(self.get_data(min_comp=min_comp, max_comp=max_comp, N=self.rounds))
        self.compute_power_list = [round(comp, 4) for comp in self.compute_power_list]

    # 计算能力变化，获取第rd轮的计算能力
    def get_new_curr_compute_power(self, rd):
        self.curr_compute_power = self.compute_power_list[rd]
        return self.curr_compute_power

    # 获取每轮总时间
    def get_total_time(self):
        # 计算时间 = 样本数 * epoches / 当前的计算能力
        self.train_time = self.num_of_trainset * self.current_epoches / self.curr_compute_power
        # 总时间 = 训练时间 + 传输时间
        self.total_time = self.train_time + self.communication_time
        # 保留4位小数
        self.total_time = round(self.total_time, 4)
        return self.total_time



    # 返回最近K次single_epoch_time的均值
    def get_single_epoch_time(self, rd, k=20):
        # 如果没有记录，使用平均计算能力计算
        if(len(self.single_epoch_time_list) == 0 or k <= 0):
            return round(self.num_of_trainset / self.compute_power, 4)
        # 如果只有1个时间，则返回该值
        if(len(self.single_epoch_time_list) == 1):
            return self.single_epoch_time_list[0][1]

        while(len(self.single_epoch_time_list) > 1):
            if(rd - k <= self.single_epoch_time_list[0][0]):
                break
            self.single_epoch_time_list.pop(0)              
        
        # 取均值
        avg_single_epoch_time = 0
        for [r, t] in self.single_epoch_time_list:
            avg_single_epoch_time += t
        avg_single_epoch_time /= len(self.single_epoch_time_list)
        
        # 保留4位小数返回均值
        return round(avg_single_epoch_time, 4)

    # 往total_time队列加入最新的total_time
    def insert_total_time(self, rd):
        self.total_time_list.append((rd, self.total_time))

    # 更新score
    def update_score(self, score):
        self.score = score

    # 返回sps
    def get_sps(self, rd, k=-1):
        if(len(self.single_epoch_time_list) == 0):
            return 0
        # 获得单个epoch的时间
        single_epoch_time = self.get_single_epoch_time(rd, k)
        
        return self.num_of_trainset / single_epoch_time
    
    # 返回最近K次训练时间的均值
    def get_avg_total_time(self, k=-1):
        # 如果没有记录，使用平均计算能力计算
        if(len(self.total_time_list) == 0 or k <= 0):
            return 0
        
        # 只记录k个
        while(len(self.total_time_list) > k):
            self.total_time_list.pop(0)
        
        # 保留4位小数返回均值
        return round(np.mean(self.total_time_list), 4)


    # FedCS初始化获取设备总时间
    def set_score_FedCS(self):
        # 计算时间 = 样本数 * epoches / 当前的计算能力
        train_time = self.num_of_trainset * self.epochs / self.compute_power
        # 训练时间 = 3 * 模型大小 / 网速，上传速率假设小于下载速率
        upload_rate = round(self.bw * math.log(1 + 100, 2) / 8, 4)
        communication_time = 3 * ClientSetup.model_size[self.data_type] / upload_rate
        # 总时间 = 训练时间 + 传输时间
        total_time = train_time + communication_time
        # 保留4位小数
        self.score = round(total_time, 4)

class Report(object):
    """联邦学习通讯的内容"""

    def __init__(self, client):
        self.client_id = client.client_id
        self.num_samples = len(client.trainset)
        self.num_of_choices = client.num_of_choices
        self.epoches = client.current_epoches
