import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .cmixer import linear2d_complex, MixerBlock_2D



# 新的多头注意力封装
class MultiHead2d_Attention(nn.Module):
    def __init__(self, ant_size, car_size, num_heads=4, heads_aggregate=True):
        super(MultiHead2d_Attention, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.num_heads = num_heads
        self.sqrt_heads = math.ceil(math.sqrt(num_heads))

        # 1. 创建多组独立的 linear2d_complex
        # 这里产生了 4 个 onehead 的参数量
        self.q_layers = nn.ModuleList([linear2d_complex(ant_size, car_size) for _ in range(num_heads)])
        self.k_layers = nn.ModuleList([linear2d_complex(ant_size, car_size) for _ in range(num_heads)])
        self.v_layers = nn.ModuleList([linear2d_complex(ant_size, car_size) for _ in range(num_heads)])

        # 2. 通道聚合层 (1x1 卷积)
        # 这里的逻辑是：把所有头的 "复数通道(2)" 拼起来，变成 (heads * 2)
        # 然后用 Linear 映射回 2。
        # Linear 作用在最后一个维度，对于 (batch, len, ant, car, channels) 的输入，
        # 它等价于在 (ant, car) 平面上的 1x1 卷积。
        # self.aggregation = nn.Linear(num_heads * 2, 2)
        self.heads_aggregate = heads_aggregate
        if heads_aggregate:
            self.aggregation1 = nn.Linear(ant_size * self.num_heads * 2, ant_size * self.sqrt_heads * 2)
            self.aggregation2 = nn.Linear(car_size * self.sqrt_heads * 2, car_size * 2)
        else:
            assert num_heads == 1, "Only one head is supported when heads_aggregate is False"

    def forward(self, x, mask=None):
        # x shape: [batch, len, ant, car, 2]

        # --- A. 投影阶段 (Projection) ---
        # 虽然这里是 Python 循环，但 CUDA 核心的发射是异步的，性能损失很小
        # 结果: list of [b, l, ant, car, 2]
        q_list = [layer(x) for layer in self.q_layers]
        k_list = [layer(x) for layer in self.k_layers]
        v_list = [layer(x) for layer in self.v_layers]

        # --- B. 堆叠实现并行化 (Parallelization via Stack) ---
        # 关键步骤：将 list 转为 Tensor，增加一个 heads 维度
        # shape: [b, heads, l, ant, car, 2]
        q = torch.stack(q_list, dim=1)
        k = torch.stack(k_list, dim=1)
        v = torch.stack(v_list, dim=1)

        # --- C. 准备 Attention 输入 ---
        # 我们需要保留 heads 维度，同时展平空间特征维度供 attention 计算
        # shape: [b, heads, l, (ant * car * 2)]
        q = rearrange(q, 'b h l s1 s2 c -> b h l (s1 s2 c)')
        k = rearrange(k, 'b h l s1 s2 c -> b h l (s1 s2 c)')
        v = rearrange(v, 'b h l s1 s2 c -> b h l (s1 s2 c)')

        # --- D. 并行计算 Attention ---
        # 这一步是完全并行的，一次算出所有 heads 的结果
        # output: [b, heads, l, (ant * car * 2)]
        # output, weights = scaled_dot_product_attention(q, k, v, mask)
        output = F.scaled_dot_product_attention(q, k, v)

        # --- E. 恢复空间结构 ---
        # output: [b, heads, l, ant, car, 2]
        output = rearrange(output, 'b h l (s1 s2 c) -> b h l s1 s2 c',
                           s1=self.ant_size, s2=self.car_size, c=2)

        if self.heads_aggregate:
            # --- F. 通道聚合 (Channel Aggregation / 1x1 Conv) ---
            # 1. 将 heads 维度合并入通道维度: [b, l, ant, car, heads*2]
            output = rearrange(output, 'b h l s1 s2 c -> b l s2 (s1 h c)')

            output = self.aggregation1(output)
            output = rearrange(output, 'b l s2 (s1 sqrth c) -> b l s1 (s2 sqrth c)', s1=self.ant_size,
                               sqrth=self.sqrt_heads, c=2)
            output = self.aggregation2(output)
            output = rearrange(output, 'b l s1 (s2 c) -> b l s1 s2 c', s2=self.car_size, c=2)

        else:
            output = output[:, 0, ...]   # 只有一个头，且没有输出 aggregate 层

        return output


class Acoupler_layer(nn.Module):
    def __init__(self, ant_size, car_size, d_model, num_heads, heads_aggregate=True, dropout=0.):
        super(Acoupler_layer, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.attention_layer = MultiHead2d_Attention(ant_size, car_size, num_heads, heads_aggregate)
        self.layernorm = nn.LayerNorm(ant_size * car_size * 2)
        self.cmixer_layer = MixerBlock_2D(ant_size, car_size, d_model, dropout=dropout)

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


class ACoupler(nn.Module):
    def __init__(self, input_ant_size, input_car_size, ant_size, car_size,
                 d_model, num_heads, heads_aggregate, depth_pred, dropout=0.):
        super(ACoupler, self).__init__()
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
            self.ac_layers.append(Acoupler_layer(ant_size, car_size, d_model, num_heads, heads_aggregate, dropout=dropout))
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
