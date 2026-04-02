import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .cmixer import linear2d_complex, MixerBlock_2D


class MultiHead2d_Attention(nn.Module):
    """2D multi-head attention over flattened spatial-complex tokens."""

    def __init__(self, ant_size, car_size, num_heads=4, heads_aggregate=True):
        super(MultiHead2d_Attention, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.num_heads = num_heads
        self.sqrt_heads = math.ceil(math.sqrt(num_heads))

        # Per-head complex Q/K/V projections.
        self.q_layers = nn.ModuleList([linear2d_complex(ant_size, car_size) for _ in range(num_heads)])
        self.k_layers = nn.ModuleList([linear2d_complex(ant_size, car_size) for _ in range(num_heads)])
        self.v_layers = nn.ModuleList([linear2d_complex(ant_size, car_size) for _ in range(num_heads)])

        # Optional head aggregation:
        # [heads*2] -> [sqrt(heads)*2] -> 2 channels.
        self.heads_aggregate = heads_aggregate
        if heads_aggregate:
            self.aggregation1 = nn.Linear(ant_size * self.num_heads * 2, ant_size * self.sqrt_heads * 2)
            self.aggregation2 = nn.Linear(car_size * self.sqrt_heads * 2, car_size * 2)
        else:
            assert num_heads == 1, "Only one head is supported when heads_aggregate is False"

    def forward(self, x, mask=None):
        # x: [batch, seq_len, ant, car, 2]
        q = torch.stack([layer(x) for layer in self.q_layers], dim=1)
        k = torch.stack([layer(x) for layer in self.k_layers], dim=1)
        v = torch.stack([layer(x) for layer in self.v_layers], dim=1)

        # Flatten spatial+complex dims to apply sequence attention per head.
        q = rearrange(q, 'b h l s1 s2 c -> b h l (s1 s2 c)')
        k = rearrange(k, 'b h l s1 s2 c -> b h l (s1 s2 c)')
        v = rearrange(v, 'b h l s1 s2 c -> b h l (s1 s2 c)')

        output = F.scaled_dot_product_attention(q, k, v)

        # Restore [ant, car, complex] structure.
        output = rearrange(output, 'b h l (s1 s2 c) -> b h l s1 s2 c',
                           s1=self.ant_size, s2=self.car_size, c=2)

        if self.heads_aggregate:
            # Merge head axis into channel axis, then compress back to complex-2 channels.
            output = rearrange(output, 'b h l s1 s2 c -> b l s2 (s1 h c)')
            output = self.aggregation1(output)
            output = rearrange(output, 'b l s2 (s1 sqrth c) -> b l s1 (s2 sqrth c)',
                               s1=self.ant_size, sqrth=self.sqrt_heads, c=2)
            output = self.aggregation2(output)
            output = rearrange(output, 'b l s1 (s2 c) -> b l s1 s2 c', s2=self.car_size, c=2)
        else:
            output = output[:, 0, ...]

        return output


class Acoupler_layer(nn.Module):
    """Residual coupling block: attention branch + cmixer branch."""

    def __init__(self, ant_size, car_size, d_model, num_heads, heads_aggregate=True, dropout=0.0):
        super(Acoupler_layer, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.attention_layer = MultiHead2d_Attention(ant_size, car_size, num_heads, heads_aggregate)
        self.layernorm = nn.LayerNorm(ant_size * car_size * 2)
        self.cmixer_layer = MixerBlock_2D(ant_size, car_size, d_model, dropout=dropout)

    def forward(self, x):
        batch_size = x.shape[0]

        # Attention over sequence dimension with normalized tokens.
        out = rearrange(x, 'b l ant car c -> b l (ant car c)')
        out_norm = self.layernorm(out)
        out_norm = rearrange(out_norm, 'b l (ant car c) -> b l ant car c', ant=self.ant_size, car=self.car_size, c=2)
        out = x + self.attention_layer(out_norm)

        # Per-time-step spatial mixing.
        out = rearrange(out, 'b l ant car c -> (b l) ant car c')
        out = self.cmixer_layer(out)
        out = rearrange(out, '(b l) ant car c -> b l ant car c', b=batch_size)
        return out


class ACoupler(nn.Module):
    """ACoupler: sparse-input lifting + temporal coupling stack + output projection."""

    def __init__(
        self,
        input_ant_size,
        input_car_size,
        ant_size,
        car_size,
        d_model,
        num_heads,
        heads_aggregate,
        depth_pred,
        dropout=0.0,
    ):
        super(ACoupler, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.fc1 = nn.Linear(input_ant_size * 2, ant_size * 2)
        self.fc2 = nn.Linear(input_car_size * 2, car_size * 2)

        # Learnable role embeddings distinguish past channels from current channel.
        self.past_embedding = nn.parameter.Parameter(torch.FloatTensor(self.ant_size, self.car_size, 2),
                                                     requires_grad=True)
        self.now_embedding = nn.parameter.Parameter(torch.FloatTensor(self.ant_size, self.car_size, 2),
                                                    requires_grad=True)
        self.past_embedding.data.fill_(0)
        self.now_embedding.data.fill_(0)

        self.ac_layers = nn.ModuleList(
            [Acoupler_layer(ant_size, car_size, d_model, num_heads, heads_aggregate, dropout=dropout)
             for _ in range(depth_pred)]
        )
        self.fc_reverse1 = nn.Linear(ant_size * 2, ant_size * 2)
        self.fc_reverse2 = nn.Linear(car_size * 2, car_size * 2)

    def forward(self, input_x_now, x_past):
        # Lift sparse current observation to full [ant, car, 2] resolution.
        x_now = rearrange(input_x_now, 'b in_ant in_car c -> b in_car (in_ant c)')
        x_now = self.fc1(x_now)
        x_now = rearrange(x_now, 'b in_car (ant c) -> b ant (in_car c)', c=2)
        x_now = self.fc2(x_now)
        x_now = rearrange(x_now, 'b ant (car c) -> b ant car c', c=2)

        # Concatenate history and current frame along temporal axis.
        x_now = rearrange(x_now, 'b ant car c -> b 1 ant car c')
        x_total = torch.cat((x_past, x_now), dim=1)

        # Add temporal-role embeddings.
        past_len = x_total.shape[1] - 1
        past_embedding_matrix = repeat(self.past_embedding, 'ant car c -> past_len ant car c', past_len=past_len)
        now_embedding_matrix = rearrange(self.now_embedding, 'ant car c -> 1 ant car c')
        embedding_matrix = torch.cat((past_embedding_matrix, now_embedding_matrix), dim=0)
        batch_size = x_total.shape[0]
        embedding_matrix = repeat(embedding_matrix, 'len ant car c -> b len ant car c', b=batch_size)
        x_total = x_total + embedding_matrix

        # Coupling stack.
        for ac_layer in self.ac_layers:
            x_total = ac_layer(x_total)

        # Decode from final token to output channel map.
        out = x_total[:, -1, :]
        out = rearrange(out, 'b ant car c -> b car (ant c)')
        out = self.fc_reverse1(out)
        out = rearrange(out, 'b car (ant c)-> b ant (car c)', c=2)
        out = self.fc_reverse2(out)
        out = rearrange(out, 'b ant (car c) -> b ant car c', c=2)

        return out
