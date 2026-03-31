import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock_2D(nn.Module):
    def __init__(self, ant_size, car_size, d_model, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.ant_mix = nn.Sequential(
            nn.LayerNorm(ant_size * 2),
            FeedForward(ant_size * 2, self.d_model, dropout),
        )
        self.car_mix = nn.Sequential(
            nn.LayerNorm(car_size * 2),
            FeedForward(car_size * 2, self.d_model, dropout),
        )

    def forward(self, x):  # 输入、输出：[b ant car c]
        x = rearrange(x, 'b ant car c -> b car (ant c)')
        x = x + self.ant_mix(x)
        x = rearrange(x, 'b car (ant c) -> b ant car c', c=2)

        x = rearrange(x, 'b ant car c -> b ant (car c)')
        x = x + self.car_mix(x)
        x = rearrange(x, 'b ant (car c) -> b ant car c', c=2)

        return x


class linear2d_complex(nn.Module):
    def __init__(self, size1: int, size2: int):
        super(linear2d_complex, self).__init__()
        self.size1 = size1
        self.size2 = size2
        # 线性层1: 作用在 s1 和 c (c=2) 维度上
        self.linear1 = nn.Linear(self.size1 * 2, self.size1 * 2)
        # 线性层2: 作用在 s2 和 c (c=2) 维度上
        self.linear2 = nn.Linear(self.size2 * 2, self.size2 * 2)

    def forward(self, x):
        # x 的形状是 [... s1 s2 2]
        # ... 代表一个或多个 batch 维度

        output = rearrange(x, '... s1 s2 c -> ... s2 (s1 c)')
        output = self.linear1(output)
        output = rearrange(output, '... s2 (s1 c) -> ... s1 s2 c', c=2)
        output = rearrange(output, '... s1 s2 c -> ... s1 (s2 c)')
        output = self.linear2(output)
        output = rearrange(output, '... s1 (s2 c) -> ... s1 s2 c', c=2)

        return output
