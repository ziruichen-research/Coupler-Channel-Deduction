import torch
import torch.nn as nn
from einops import rearrange, repeat
from .cmixer import MixerBlock_2D  # 确保此依赖项路径正确


# --------------------------------------------------------
# 核心网络结构更新：使用 PyTorch 官方 MultiheadAttention
# --------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, ant_size, car_size, num_heads=4, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size

        # 展平后的特征总维度
        self.flat_dim = ant_size * car_size * 2

        # 官方多头注意力的硬性要求：特征维度必须能被头数整除
        assert self.flat_dim % num_heads == 0, f"展平维度 ({self.flat_dim}) 必须能被 num_heads ({num_heads}) 整除"

        # 直接调用 PyTorch 官方 API
        # batch_first=True 极其重要，它允许输入张量的形状为 [batch_size, seq_len, feature_dim]
        self.mha = nn.MultiheadAttention(
            embed_dim=self.flat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x, mask=None):
        # x shape: [batch, len, ant, car, 2]

        # --- A. 展平空间和复数维度 ---
        # shape: [batch, len, flat_dim]
        x_flat = rearrange(x, 'b l ant car c -> b l (ant car c)')

        # --- B. 调用 PyTorch API ---
        # 将 x_flat 同时作为 Query, Key, Value 传入
        # need_weights=False 可以跳过注意力权重矩阵的计算，进一步提升速度
        attn_output, _ = self.mha(
            query=x_flat,
            key=x_flat,
            value=x_flat,
            need_weights=False,
            key_padding_mask=mask  # 如果有 padding mask 可以传这里
        )

        # --- C. 恢复多维空间结构 ---
        # shape: [batch, len, ant, car, 2]
        out = rearrange(attn_output, 'b l (ant car c) -> b l ant car c',
                        ant=self.ant_size, car=self.car_size, c=2)

        return out


class AcouplerFlat_layer_v2(nn.Module):
    def __init__(self, ant_size, car_size, d_model, num_heads, dropout=0.):
        super(AcouplerFlat_layer_v2, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size

        # 接入官方 API 版本的注意力层
        self.attention_layer = MultiHeadAttention(ant_size, car_size, num_heads, dropout)
        self.layernorm = nn.LayerNorm(ant_size * car_size * 2)
        self.cmixer_layer = MixerBlock_2D(ant_size, car_size, d_model, dropout=dropout)

    def forward(self, x):
        batch_size = x.shape[0]

        # 注意力模块前的 LayerNorm 处理
        out = rearrange(x, 'b l ant car c -> b l (ant car c)')
        out_norm = self.layernorm(out)
        out_norm = rearrange(out_norm, 'b l (ant car c) -> b l ant car c', ant=self.ant_size, car=self.car_size, c=2)

        # 注意力计算与残差连接
        out_attn = self.attention_layer(out_norm)
        out = x + out_attn

        # Channel Mixer 处理
        out = rearrange(out, 'b l ant car c -> (b l) ant car c')
        out = self.cmixer_layer(out)
        out = rearrange(out, '(b l) ant car c -> b l ant car c', b=batch_size)

        return out


class ACouplerFlat_v2(nn.Module):
    # 移除了 heads_aggregate 参数，因为标准 MHA 自带输出投影层
    def __init__(self, input_ant_size, input_car_size, ant_size, car_size,
                 d_model, num_heads, depth_pred, dropout=0.):
        super(ACouplerFlat_v2, self).__init__()
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

        self.ac_layers = nn.ModuleList([
            AcouplerFlat_layer_v2(ant_size, car_size, d_model, num_heads, dropout=dropout)
            for _ in range(depth_pred)
        ])

        self.fc_reverse1 = nn.Linear(ant_size * 2, ant_size * 2)
        self.fc_reverse2 = nn.Linear(car_size * 2, car_size * 2)

    def forward(self, input_x_now, x_past):
        # 处理当前的输入
        x_now = rearrange(input_x_now, 'b in_ant in_car c -> b in_car (in_ant c)')
        x_now = self.fc1(x_now)
        x_now = rearrange(x_now, 'b in_car (ant c) -> b ant (in_car c)', c=2)
        x_now = self.fc2(x_now)
        x_now = rearrange(x_now, 'b ant (car c) -> b ant car c', c=2)

        x_now = rearrange(x_now, 'b ant car c -> b 1 ant car c')
        x_total = torch.cat((x_past, x_now), dim=1)

        # 注入位置嵌入
        past_len = x_total.shape[1] - 1
        past_embedding_matrix = repeat(self.past_embedding, 'ant car c -> past_len ant car c', past_len=past_len)
        now_embedding_matrix = rearrange(self.now_embedding, 'ant car c -> 1 ant car c')
        embedding_matrix = torch.cat((past_embedding_matrix, now_embedding_matrix), dim=0)
        batch_size = x_total.shape[0]
        embedding_matrix = repeat(embedding_matrix, 'len ant car c -> b len ant car c', b=batch_size)

        x_total = x_total + embedding_matrix

        # 串行通过多个 Acoupler 层
        for ac_layer in self.ac_layers:
            x_total = ac_layer(x_total)

        # 提取最后一个时间步并做逆向映射
        out = x_total[:, -1, :]
        out = rearrange(out, 'b ant car c -> b car (ant c)')
        out = self.fc_reverse1(out)
        out = rearrange(out, 'b car (ant c)-> b ant (car c)', c=2)
        out = self.fc_reverse2(out)
        out = rearrange(out, 'b ant (car c) -> b ant car c', c=2)

        return out