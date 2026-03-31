import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .cmixer import MixerBlock_2D  # 确保此依赖项路径正确


# --------------------------------------------------------
# 新的展平投影层：替代原有的 linear2d_complex
# --------------------------------------------------------
class linear_flat_complex(nn.Module):
    def __init__(self, size1: int, size2: int):
        super(linear_flat_complex, self).__init__()
        self.size1 = size1
        self.size2 = size2
        # 将三个维度一起展平：ant_size * car_size * 2
        self.flat_dim = size1 * size2 * 2
        self.linear = nn.Linear(self.flat_dim, self.flat_dim)

    def forward(self, x):
        # x 的形状是 [... s1 s2 2]

        # 1. 展平空间维度和复数维度
        output = rearrange(x, '... s1 s2 c -> ... (s1 s2 c)')

        # 2. 统一进行线性投影
        output = self.linear(output)

        # 3. 恢复原有的多维空间结构
        output = rearrange(output, '... (s1 s2 c) -> ... s1 s2 c', s1=self.size1, s2=self.size2, c=2)

        return output


# --------------------------------------------------------
# 核心网络结构更新
# --------------------------------------------------------
class MultiHead1d_Attention(nn.Module):
    def __init__(self, ant_size, car_size, num_heads=1, heads_aggregate=True):
        super(MultiHead1d_Attention, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.num_heads = num_heads

        # 展平后的总特征维度大小
        self.flat_dim = ant_size * car_size * 2

        # 1. 创建多组独立的 linear_flat_complex (保留你原有的多头实现方式)
        self.q_layers = nn.ModuleList([linear_flat_complex(ant_size, car_size) for _ in range(num_heads)])
        self.k_layers = nn.ModuleList([linear_flat_complex(ant_size, car_size) for _ in range(num_heads)])
        self.v_layers = nn.ModuleList([linear_flat_complex(ant_size, car_size) for _ in range(num_heads)])

        # 2. 展平版本的通道聚合层
        self.heads_aggregate = heads_aggregate
        if heads_aggregate:
            # 直接将 (num_heads * flat_dim) 映射回 flat_dim
            self.aggregation = nn.Linear(self.num_heads * self.flat_dim, self.flat_dim)
        else:
            assert num_heads == 1, "Only one head is supported when heads_aggregate is False"

    def forward(self, x, mask=None):
        # x shape: [batch, len, ant, car, 2]

        # --- A. 投影阶段 ---
        q_list = [layer(x) for layer in self.q_layers]
        k_list = [layer(x) for layer in self.k_layers]
        v_list = [layer(x) for layer in self.v_layers]

        # --- B. 堆叠实现并行化 ---
        q = torch.stack(q_list, dim=1)
        k = torch.stack(k_list, dim=1)
        v = torch.stack(v_list, dim=1)

        # --- C. 准备 Attention 输入 ---
        q = rearrange(q, 'b h l s1 s2 c -> b h l (s1 s2 c)')
        k = rearrange(k, 'b h l s1 s2 c -> b h l (s1 s2 c)')
        v = rearrange(v, 'b h l s1 s2 c -> b h l (s1 s2 c)')

        # --- D. 并行计算 Attention ---
        # output shape: [b, heads, l, (s1 s2 c)]
        output = F.scaled_dot_product_attention(q, k, v)

        # --- E. 通道聚合与恢复空间结构 ---
        if self.heads_aggregate:
            # 1. 拼接所有头: [b, h, l, flat_dim] -> [b, l, (h * flat_dim)]
            output = rearrange(output, 'b h l flat_dim -> b l (h flat_dim)')

            # 2. 统一线性映射
            output = self.aggregation(output)

            # 3. 恢复空间结构: [b, l, flat_dim] -> [b, l, ant, car, 2]
            output = rearrange(output, 'b l (s1 s2 c) -> b l s1 s2 c',
                               s1=self.ant_size, s2=self.car_size, c=2)
        else:
            # 只有单头的情况
            output = output[:, 0, ...]
            output = rearrange(output, 'b l (s1 s2 c) -> b l s1 s2 c',
                               s1=self.ant_size, s2=self.car_size, c=2)

        return output


class AcouplerFlat_layer(nn.Module):
    def __init__(self, ant_size, car_size, d_model, num_heads, heads_aggregate=True, dropout=0.):
        super(AcouplerFlat_layer, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.attention_layer = MultiHead1d_Attention(ant_size, car_size, num_heads, heads_aggregate)
        self.layernorm = nn.LayerNorm(ant_size * car_size * 2)
        self.cmixer_layer = MixerBlock_2D(ant_size, car_size, d_model, dropout=dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        out = rearrange(x, 'b l ant car c -> b l (ant car c)')
        out_norm = self.layernorm(out)
        out_norm = rearrange(out_norm, 'b l (ant car c) -> b l ant car c', ant=self.ant_size, car=self.car_size, c=2)
        out_attn = self.attention_layer(out_norm)
        out = x + out_attn

        out = rearrange(out, 'b l ant car c -> (b l) ant car c')
        out = self.cmixer_layer(out)
        out = rearrange(out, '(b l) ant car c -> b l ant car c', b=batch_size)
        return out


class ACouplerFlat(nn.Module):
    def __init__(self, input_ant_size, input_car_size, ant_size, car_size,
                 d_model, num_heads, heads_aggregate, depth_pred, dropout=0.):
        super(ACouplerFlat, self).__init__()
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
            self.ac_layers.append(
                AcouplerFlat_layer(ant_size, car_size, d_model, num_heads, heads_aggregate, dropout=dropout)
            )
        self.fc_reverse1 = nn.Linear(ant_size * 2, ant_size * 2)
        self.fc_reverse2 = nn.Linear(car_size * 2, car_size * 2)

    def forward(self, input_x_now, x_past):
        x_now = rearrange(input_x_now, 'b in_ant in_car c -> b in_car (in_ant c)')
        x_now = self.fc1(x_now)
        x_now = rearrange(x_now, 'b in_car (ant c) -> b ant (in_car c)', c=2)
        x_now = self.fc2(x_now)
        x_now = rearrange(x_now, 'b ant (car c) -> b ant car c', c=2)

        x_now = rearrange(x_now, 'b ant car c -> b 1 ant car c')
        x_total = torch.cat((x_past, x_now), dim=1)

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