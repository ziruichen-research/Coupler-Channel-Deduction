import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .cmixer import linear2d_complex, MixerBlock_2D


class MultiHead_Attention(nn.Module):
    def __init__(self, ant_size, car_size, d_model, num_heads=8, dropout=0., has_proj=True):
        super(MultiHead_Attention, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size

        # 计算展平后的特征维度: ant * car * 2 (实部+虚部)
        self.flat_dim = ant_size * car_size * 2
        if has_proj:
            self.input_proj = nn.Linear(self.flat_dim, d_model)
            self.output_proj = nn.Linear(d_model, self.flat_dim)
        else:
            assert d_model == self.flat_dim
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)

    def forward(self, x, mask=None):
        # x shape: [batch, len, ant, car, 2]
        b, l, s1, s2, c = x.shape

        # --- A. 维度展平与投影 ---
        # 将空间和复数维度展平: [b, l, ant*car*2]
        x_flat = rearrange(x, 'b l s1 s2 c -> b l (s1 s2 c)')

        # 映射到 d_model: [b, l, d_model]
        x_embed = self.input_proj(x_flat)

        # --- B. MHA 计算 ---
        # Self-Attention: Q=K=V=x_embed
        # attn_output shape: [b, l, d_model]
        attn_output, _ = self.mha(x_embed, x_embed, x_embed, key_padding_mask=mask)

        # --- C. 恢复维度 ---
        # 映射回物理维度: [b, l, ant*car*2]
        output_flat = self.output_proj(attn_output)

        # 重塑回 5D 张量: [b, l, ant, car, 2]
        output = rearrange(output_flat, 'b l (s1 s2 c) -> b l s1 s2 c',
                           s1=self.ant_size, s2=self.car_size, c=2)

        return output


class AcouplerStandardlayer(nn.Module):
    def __init__(self, ant_size, car_size, d_model, num_heads,
                 dim_feedforward=128, dropout=0., has_proj=True):
        super(AcouplerStandardlayer, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        # 修改点：传递 d_model 和 dropout，移除了 heads_aggregate (标准MHA内部处理聚合)
        self.attention_layer = MultiHead_Attention(
            ant_size=ant_size,
            car_size=car_size,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            has_proj=has_proj
        )
        self.layernorm = nn.LayerNorm(ant_size * car_size * 2)
        self.cmixer_layer = MixerBlock_2D(ant_size, car_size, dim_feedforward, dropout=dropout)

    def forward(self, x):  # x[batch,len,ant,car,2]
        batch_size = x.shape[0]
        out = rearrange(x, 'b l ant car c -> b l (ant car c)')
        out_norm = self.layernorm(out)
        out_norm = rearrange(out_norm, 'b l (ant car c) -> b l ant car c', ant=self.ant_size, car=self.car_size, c=2)
        out_attn = self.attention_layer(out_norm)  # <-- 替换后的代码
        out = x + out_attn

        out = rearrange(out, 'b l ant car c -> (b l) ant car c')
        out = self.cmixer_layer(out)
        out = rearrange(out, '(b l) ant car c -> b l ant car c', b=batch_size)
        return out


class ACouplerStandard(nn.Module):
    def __init__(self, input_ant_size, input_car_size, ant_size, car_size,
                 d_model, num_heads, dim_feedforward, depth_pred, dropout=0., has_proj=True):
        super(ACouplerStandard, self).__init__()
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
            self.ac_layers.append(AcouplerStandardlayer(ant_size, car_size, d_model, num_heads,
                                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                                        has_proj=has_proj))
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
