import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .cmixer import linear2d_complex, MixerBlock_2D


# def scaled_dot_product_attention(query, key, value, mask=None):
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     attention_weights = F.softmax(scores, dim=-1)
#     output = torch.matmul(attention_weights, value)
#
#     return output, attention_weights


class Coupler_layer(nn.Module):
    def __init__(self, ant_size, car_size, d_model):
        super(Coupler_layer, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.layernorm = nn.LayerNorm(ant_size * car_size * 2)
        self.cmixer_layer = MixerBlock_2D(ant_size, car_size, d_model)

    def forward(self, x):  # x[batch,len,ant,car,2]
        batch_size = x.shape[0]
        out = rearrange(x, 'b l ant car c -> b l (ant car c)')
        out_norm = self.layernorm(out)
        out_attn = F.scaled_dot_product_attention(out_norm, out_norm, out_norm)
        out_attn = rearrange(out_attn, 'b l (ant car c) -> b l ant car c',
                             ant=self.ant_size, car=self.car_size, c=2)
        out = x + out_attn

        out = rearrange(out, 'b l ant car c -> (b l) ant car c')
        out = self.cmixer_layer(out)
        out = rearrange(out, '(b l) ant car c -> b l ant car c', b=batch_size)
        return out


class Coupler(nn.Module):
    def __init__(self, input_ant_size, input_car_size, ant_size, car_size,
                 d_model, depth_pred):
        super(Coupler, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.fc1 = nn.Linear(input_ant_size * 2, ant_size * 2)
        self.fc2 = nn.Linear(input_car_size * 2, car_size * 2)

        self.past_embedding = nn.parameter.Parameter(torch.FloatTensor(self.ant_size, self.car_size, 2),
                                                     requires_grad=True)
        self.now_embedding = nn.parameter.Parameter(torch.FloatTensor(self.ant_size, self.car_size, 2),
                                                    requires_grad=True)
        self.past_embedding.data.fill_(0)
        self.now_embedding.data.fill_(0)

        self.ac_layers = nn.ModuleList([])
        for _ in range(depth_pred):
            self.ac_layers.append(Coupler_layer(ant_size, car_size, d_model))
        self.fc_reverse1 = nn.Linear(ant_size * 2, ant_size * 2)
        self.fc_reverse2 = nn.Linear(car_size * 2, car_size * 2)

    def forward(self, input_x_now, x_past):
        x_now = rearrange(input_x_now, 'b in_ant in_car c -> b in_car (in_ant c)')
        x_now = self.fc1(x_now)
        x_now = rearrange(x_now, 'b in_car (ant c) -> b ant (in_car c)', c=2)
        x_now = self.fc2(x_now)
        x_now = rearrange(x_now, 'b ant (car c) -> b ant car c', c=2)

        x_now = rearrange(x_now, 'b ant car c -> b 1 ant car c')
        x_total = torch.cat((x_past, x_now), dim=1)  # 把now放在最后

        past_len = x_total.shape[1] - 1
        past_embedding_matrix = repeat(self.past_embedding, 'ant car c -> past_len ant car c', past_len=past_len)
        now_embedding_matrix = rearrange(self.now_embedding, 'ant car c -> 1 ant car c')
        embedding_matrix = torch.cat((past_embedding_matrix, now_embedding_matrix), dim=0)
        batch_size = x_total.shape[0]
        embedding_matrix = repeat(embedding_matrix, 'len ant car c -> b len ant car c', b=batch_size)
        x_total = x_total + embedding_matrix

        for ac_layer in self.ac_layers:
            x_total = ac_layer(x_total)

        out = x_total[:, -1, :]
        out = rearrange(out, 'b ant car c -> b car (ant c)')
        out = self.fc_reverse1(out)
        out = rearrange(out, 'b car (ant c)-> b ant (car c)', c=2)
        out = self.fc_reverse2(out)
        out = rearrange(out, 'b ant (car c) -> b ant car c', c=2)

        return out
