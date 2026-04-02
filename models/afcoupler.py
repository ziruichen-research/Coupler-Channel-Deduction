import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .cmixer import linear2d_complex, MixerBlock_2D


class AFT_Simple_Layer(nn.Module):
    """
    AFT-simple attention:
    Y_t = sigmoid(Q_t) * (sum(exp(K_t') * V_t') / sum(exp(K_t')))
    """

    def __init__(self, ant_size, car_size, is_learnable=True):
        super(AFT_Simple_Layer, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size

        # Use either learnable complex projections or identity passthrough.
        if is_learnable:
            self.w_q = linear2d_complex(ant_size, car_size)
            self.w_k = linear2d_complex(ant_size, car_size)
            self.w_v = linear2d_complex(ant_size, car_size)
        else:
            self.w_q = nn.Identity()
            self.w_k = nn.Identity()
            self.w_v = nn.Identity()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, seq_len, ant_size, car_size, 2].
        """
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Flatten spatial+complex dims so AFT is applied per token vector.
        q = rearrange(q, '... s1 s2 c -> ... (s1 s2 c)')
        k = rearrange(k, '... s1 s2 c -> ... (s1 s2 c)')
        v = rearrange(v, '... s1 s2 c -> ... (s1 s2 c)')

        # Numerically stable exponential normalization along sequence axis.
        k_max = torch.max(k, dim=1, keepdim=True)[0]
        k_stable = k - k_max
        k_exp = torch.exp(k_stable)

        numerator = torch.sum(k_exp * v, dim=1)
        denominator = torch.sum(k_exp, dim=1)
        global_context = (numerator / (denominator + 1e-6)).unsqueeze(1)

        output = self.sigmoid(q) * global_context
        output = rearrange(output, '... (s1 s2 c) -> ... s1 s2 c', s1=self.ant_size, s2=self.car_size, c=2)
        return output, None


class AFcoupler_layer(nn.Module):
    """Residual coupling block: LayerNorm+AFT followed by cmixer."""

    def __init__(self, ant_size, car_size, d_model, is_learnable=True, dropout=0.0):
        super(AFcoupler_layer, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.layernorm = nn.LayerNorm(ant_size * car_size * 2)
        self.attention_layer = AFT_Simple_Layer(ant_size, car_size, is_learnable=is_learnable)
        self.cmixer_layer = MixerBlock_2D(ant_size, car_size, d_model, dropout=dropout)

    def forward(self, x):
        batch_size = x.shape[0]

        # Attention branch over temporal tokens.
        out = rearrange(x, 'b l ant car c -> b l (ant car c)')
        out_norm = self.layernorm(out)
        out_norm = rearrange(out_norm, 'b l (ant car c) -> b l ant car c', ant=self.ant_size, car=self.car_size, c=2)
        out_attn, _ = self.attention_layer(out_norm)
        out = x + out_attn

        # Spatial mixer branch per time step.
        out = rearrange(out, 'b l ant car c -> (b l) ant car c')
        out = self.cmixer_layer(out)
        out = rearrange(out, '(b l) ant car c -> b l ant car c', b=batch_size)
        return out


class AFCoupler(nn.Module):
    """AFCoupler: input lifting + stacked AFT coupling blocks + output projection."""

    def __init__(
        self,
        input_ant_size,
        input_car_size,
        ant_size,
        car_size,
        d_model,
        depth_pred,
        is_learnable=True,
        dropout=0.0,
    ):
        super(AFCoupler, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.fc1 = nn.Linear(input_ant_size * 2, ant_size * 2)
        self.fc2 = nn.Linear(input_car_size * 2, car_size * 2)

        # Temporal role embeddings for past frames and current frame.
        self.past_embedding = nn.parameter.Parameter(torch.FloatTensor(self.ant_size, self.car_size, 2), requires_grad=True)
        self.now_embedding = nn.parameter.Parameter(torch.FloatTensor(self.ant_size, self.car_size, 2), requires_grad=True)
        self.past_embedding.data.fill_(0)
        self.now_embedding.data.fill_(0)

        self.afc_layers = nn.ModuleList(
            [AFcoupler_layer(ant_size, car_size, d_model, is_learnable=is_learnable, dropout=dropout)
             for _ in range(depth_pred)]
        )
        self.fc_reverse1 = nn.Linear(ant_size * 2, ant_size * 2)
        self.fc_reverse2 = nn.Linear(car_size * 2, car_size * 2)

    def forward(self, input_x_now, x_past, snr_now=None, snr_past=None):
        # Lift sparse current observation to full [ant, car, 2] map.
        x_now = rearrange(input_x_now, 'b in_ant in_car c -> b in_car (in_ant c)')
        x_now = self.fc1(x_now)
        x_now = rearrange(x_now, 'b in_car (ant c) -> b ant (in_car c)', c=2)
        x_now = self.fc2(x_now)
        x_now = rearrange(x_now, 'b ant (car c) -> b ant car c', c=2)

        # Concatenate history and current token.
        x_now = rearrange(x_now, 'b ant car c -> b 1 ant car c')
        x_total = torch.cat((x_past, x_now), dim=1)

        # Add temporal-role embeddings before coupling blocks.
        past_len = x_total.shape[1] - 1
        past_embedding_matrix = repeat(self.past_embedding, 'ant car c -> past_len ant car c', past_len=past_len)
        now_embedding_matrix = rearrange(self.now_embedding, 'ant car c -> 1 ant car c')
        embedding_matrix = torch.cat((past_embedding_matrix, now_embedding_matrix), dim=0)
        batch_size = x_total.shape[0]
        embedding_matrix = repeat(embedding_matrix, 'len ant car c -> b len ant car c', b=batch_size)
        x_total = x_total + embedding_matrix

        # Coupling stack.
        for afc_layer in self.afc_layers:
            x_total = afc_layer(x_total)

        # Decode from final token to channel estimate.
        out = x_total[:, -1, :]
        out = rearrange(out, 'b ant car c -> b car (ant c)')
        out = self.fc_reverse1(out)
        out = rearrange(out, 'b car (ant c)-> b ant (car c)', c=2)
        out = self.fc_reverse2(out)
        out = rearrange(out, 'b ant (car c) -> b ant car c', c=2)

        return out
