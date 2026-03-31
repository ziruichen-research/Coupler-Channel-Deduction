import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
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

    def forward(self, x):  # 输入、输出：[b ant car c]
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
            self.mixer_blocks.append(MixerBlock_2D(ant_size, car_size))

    def forward(self, x):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        return x


class AFT_Simple_Layer(nn.Module):
    """
    实现了 AFT-simple
    Y_t = sigmoid(Q_t) * (sum(exp(K_t') * V_t') / sum(exp(K_t')))
    """
    def __init__(self, d_model):
        super(AFT_Simple_Layer, self).__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # AFT-simple 中的激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x 的输入维度: [batch_size, seq_len, ant_size,car_size,2]
        在你的代码中，seq_len 就是 'l'
        """
        # 1. 计算 Q, K, V
        q = self.w_q(x)  # [b, l, d]
        k = self.w_k(x)  # [b, l, d]
        v = self.w_v(x)  # [b, l, d]
        # 2. 计算 AFT-simple 的分子和分母
        # (sum(exp(K_t') * V_t') / sum(exp(K_t')))
        # [b, l, d]
        # k_exp = torch.exp(k)
        # 2. 计算 AFT-simple 的分子和分母（带稳定性）
        # 沿序列维度(dim=1)找到k的最大值
        # k_max 形状: [b, 1, d]
        k_max = torch.max(k, dim=1, keepdim=True)[0]
        # 从 k 中减去最大值，防止 exp 溢出
        # k_stable 形状: [b, l, d]
        k_stable = k - k_max
        # 现在 exp(k_stable) 的最大值是 exp(0) = 1，绝对安全
        k_exp = torch.exp(k_stable)
        # --- 改动结束 ---
        # 沿序列长度 'l' (dim=1) 求和
        # numerator: [b, d]
        numerator = torch.sum(k_exp * v, dim=1)
        # denominator: [b, d]
        denominator = torch.sum(k_exp, dim=1)
        # 3. 计算全局上下文向量
        # 添加一个小的 epsilon 防止除以零
        # [b, d] -> [b, 1, d] (以便广播)
        global_context = (numerator / (denominator + 1e-6)).unsqueeze(1)
        # 4. 计算最终输出
        # Y_t = sigmoid(Q_t) * global_context
        # [b, l, d] * [b, 1, d] -> [b, l, d]
        output = self.sigmoid(q) * global_context
        # AFT-simple 没有像标准 attention 那样的 [b, l, l] 注意力权重矩阵
        # 因此我们返回 None
        return output, None


class AFT_layer(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super(AFT_layer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention_layer = AFT_Simple_Layer(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_layer = FeedForward(d_model, hidden_dim=dim_feedforward)

    def forward(self, x):  # [batch, len, d]
        out_norm = self.norm1(x)
        out_attn, _ = self.attention_layer(out_norm)  # <-- 替换后的代码
        out = x + out_attn
        out_norm = self.norm2(out)
        out = out + self.ffn_layer(out_norm)
        return out


class est_net(nn.Module):
    def __init__(self, input_ant_size, input_car_size, ant_size, car_size, depth):
        super(est_net, self).__init__()
        self.fc1 = nn.Linear(input_ant_size * 2, ant_size * 2)
        self.fc2 = nn.Linear(input_car_size * 2, car_size * 2)
        self.mlpmixer_us = MLPMixer_us_2D(ant_size, car_size, depth)
        self.fc_reverse1 = nn.Linear(ant_size * 2, ant_size * 2)
        self.fc_reverse2 = nn.Linear(car_size * 2, car_size * 2)

    def forward(self, x):  #
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
    def __init__(self, ant_size, car_size, depth_pred, hidden_size=512):
        super(pred_net, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.hidden_size = hidden_size

        self.past_embedding = nn.parameter.Parameter(torch.FloatTensor(self.hidden_size), requires_grad=True)
        self.now_embedding = nn.parameter.Parameter(torch.FloatTensor(self.hidden_size), requires_grad=True)
        self.past_embedding.data.fill_(0)
        self.now_embedding.data.fill_(0)

        self.fc1 = nn.Linear(ant_size * car_size * 2, self.hidden_size)
        self.af_layers = nn.ModuleList([])
        for _ in range(depth_pred):
            self.af_layers.append(AFT_layer(self.hidden_size, dim_feedforward=self.hidden_size))
        self.fc_reverse1 = nn.Linear(self.hidden_size, ant_size * car_size * 2)

    def forward(self, x_now, x_past):  # x_now:[b ant car c] x_past[b len-1 ant car c]  out [b ant car c]
        x_now = rearrange(x_now, 'b ant car c -> b 1 ant car c')
        x_total = torch.cat((x_past, x_now), dim=1)  # 把now放在最后
        x_total = rearrange(x_total, 'b len ant car c -> b len (ant car c)')
        x_total = self.fc1(x_total)

        past_len = x_total.shape[1] - 1
        past_embedding_matrix = repeat(self.past_embedding, 'h -> past_len h', past_len=past_len)
        now_embedding_matrix = rearrange(self.now_embedding, 'h -> 1 h')
        embedding_matrix = torch.cat((past_embedding_matrix, now_embedding_matrix), dim=0)
        batch_size = x_total.shape[0]
        embedding_matrix = repeat(embedding_matrix, 'len h -> b len h', b=batch_size)
        x_total = x_total + embedding_matrix

        for i in range(len(self.af_layers)):
            x_total = self.af_layers[i](x_total)
        out = x_total[:, -1, :]
        out = self.fc_reverse1(out)
        out = rearrange(out, 'b (ant car c)-> b ant car c', ant=self.ant_size, car=self.car_size, c=2)
        return out


class AFCDNet(nn.Module):
    def __init__(self, input_ant_size, input_car_size, ant_size, car_size, depth, depth_pred, hidden_size=512):
        super(AFCDNet, self).__init__()
        self.est_net = est_net(input_ant_size, input_car_size, ant_size, car_size, depth)
        self.pred_net = pred_net(ant_size, car_size, depth_pred, hidden_size=hidden_size)
        self.est_net2 = est_net(ant_size, car_size, ant_size, car_size, depth)

    def forward(self, input_x_now, x_past):
        x_now = self.est_net(input_x_now)
        out = self.pred_net(x_now, x_past)
        out = self.est_net2(out)
        return out
