import numpy as np
import random


def uniform(N, k):
    """均匀分布，将N份平均分成k组"""
    dist = []
    avg = N / k
    # 开始分布
    for i in range(k):
        dist.append(int((i + 1) * avg) - int(i * avg))
    # 返回随机排序的序列
    random.shuffle(dist)
    return dist


def normal(N, k):
    """Normal distribution of 'N' items into 'k' groups."""
    dist = []
    # Make distribution
    for i in range(k):
        x = i - (k - 1) / 2
        dist.append(int(N * (np.exp(-x) / (np.exp(-x) + 1)**2)))
    # Add remainders
    remainder = N - sum(dist)
    dist = list(np.add(dist, uniform(remainder, k)))
    # Return non-shuffled distribution
    return dist
