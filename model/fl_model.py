import torch
import torch.nn.functional as F
import logging


# Cuda设置
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# 获取训练集dataloader
def get_trainloader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


# 获取测试集dataloader
def get_testloader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)


# 提取参数
def extract_weights(model):
    weights = []
    for name, weight in model.named_parameters():
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights


# 载入参数
def load_weights(model, weights):
    updated_weights_dict = {}
    for name, weight in weights:
        updated_weights_dict[name] = weight

    model.load_state_dict(updated_weights_dict, strict=False)
