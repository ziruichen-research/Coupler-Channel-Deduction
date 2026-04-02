import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange, repeat


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
    def __init__(self, ant_size, car_size, dropout=0.):
        super().__init__()
        self.ant_mix = nn.Sequential(
            nn.LayerNorm(ant_size * 2),
            FeedForward(ant_size * 2, ant_size * 2 * 2, dropout),
        )
        self.car_mix = nn.Sequential(
            nn.LayerNorm(car_size * 2),
            FeedForward(car_size * 2, car_size * 2 * 2, dropout),
        )

    def forward(self, x):
        x = rearrange(x, 'b ant car c -> b car (ant c)')
        x = x + self.ant_mix(x)
        x = rearrange(x, 'b car (ant c) -> b ant car c', c=2)

        x = rearrange(x, 'b ant car c -> b ant (car c)')
        x = x + self.car_mix(x)
        x = rearrange(x, 'b ant (car c) -> b ant car c', c=2)

        return x


class MLPMixer_us_2D(nn.Module):
    def __init__(self, ant_size, car_size, depth, dropout=0.):
        super(MLPMixer_us_2D, self).__init__()
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock_2D(ant_size, car_size, dropout=dropout))

    def forward(self, x):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        return x


class est_net(nn.Module):
    def __init__(self, input_ant_size, input_car_size, ant_size, car_size, depth, dropout=0.):
        super(est_net, self).__init__()
        self.fc1 = nn.Linear(input_ant_size * 2, ant_size * 2)
        self.fc2 = nn.Linear(input_car_size * 2, car_size * 2)
        self.mlpmixer_us = MLPMixer_us_2D(ant_size, car_size, depth, dropout=dropout)
        self.fc_reverse1 = nn.Linear(ant_size * 2, ant_size * 2)
        self.fc_reverse2 = nn.Linear(car_size * 2, car_size * 2)

    def forward(self, x):
        out = rearrange(x, 'b in_ant in_car c -> b in_car (in_ant c)')
        out = self.fc1(out)
        out = rearrange(out, 'b in_car (ant c) -> b ant (in_car c)', c=2)
        out = self.fc2(out)
        out = rearrange(out, 'b ant (car c) -> b ant car c', c=2)

        out = self.mlpmixer_us(out)

        out = rearrange(out, 'b ant car c -> b car (ant c)')
        out = self.fc_reverse1(out)
        out = rearrange(out, 'b car (ant c)-> b ant (car c)', c=2)
        out = self.fc_reverse2(out)
        out = rearrange(out, 'b ant (car c) -> b ant car c', c=2)

        return out


class pred_net(nn.Module):
    def __init__(self, ant_size, car_size, depth_pred, hidden_size=512, dropout=0.):
        super(pred_net, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(ant_size * car_size * 2, self.hidden_size)
        self.lstm_layer = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=depth_pred,
                                  batch_first=True, dropout=dropout)
        self.fc_reverse1 = nn.Linear(self.hidden_size, ant_size * car_size * 2)

    def forward(self, x_now, x_past):
        x_now = rearrange(x_now, 'b ant car c -> b 1 ant car c')
        x_total = torch.cat((x_past, x_now), dim=1)
        x_total = rearrange(x_total, 'b len ant car c -> b len (ant car c)')
        x_total = self.fc1(x_total)
        out, (h_n, c_n) = self.lstm_layer(x_total)
        out = out[:, -1, :]
        out = self.fc_reverse1(out)
        out = rearrange(out, 'b (ant car c)-> b ant car c', ant=self.ant_size, car=self.car_size, c=2)
        return out


class RCDNet(nn.Module):
    def __init__(self, input_ant_size, input_car_size, ant_size, car_size, depth, depth_pred,
                 hidden_size=512, dropout=0.):
        super(RCDNet, self).__init__()
        self.est_net = est_net(input_ant_size, input_car_size, ant_size, car_size, depth, dropout=dropout)
        self.pred_net = pred_net(ant_size, car_size, depth_pred, hidden_size=hidden_size, dropout=dropout)
        self.est_net2 = est_net(ant_size, car_size, ant_size, car_size, depth, dropout=dropout)

    def forward(self, input_x_now, x_past):
        x_now = self.est_net(input_x_now)
        out = self.pred_net(x_now, x_past)
        out = self.est_net2(out)
        return out
