import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size, bias=True):
        """
        初始化 ConvLSTM 单元
        :param input_dim: 输入张量的通道数 (Channels)
        :param hidden_dim: 隐藏状态的通道数
        :param kernel_size: 卷积核大小 (int or tuple)
        :param bias: 是否使用偏置
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 保持空间维度不变，padding 设置为 kernel_size // 2
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        # 这里的卷积操作对应公式中的 W_x * X 和 W_h * H
        # 为了效率，我们通常将 4 个门 (i, f, c_tilde, o) 的卷积合并到一个卷积层中计算
        # 输出通道数为 4 * hidden_dim
        self.conv_x = nn.Conv2d(in_channels=input_dim,
                                out_channels=4 * hidden_dim,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=bias)

        self.conv_h = nn.Conv2d(in_channels=hidden_dim,
                                out_channels=4 * hidden_dim,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=False)  # 偏置通常只加一次

        # Peephole connections: W_ci, W_cf, W_co
        # 这些是逐元素乘法参数，形状与 Cell State 一致
        self.W_ci = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        self.W_cf = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        self.W_co = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # 1. 计算卷积项: W_x * X + W_h * H_prev
        # input_tensor shape: (B, C_in, H, W)
        x_conv = self.conv_x(input_tensor)
        h_conv = self.conv_h(h_cur)

        # 将卷积结果相加
        combined_conv = x_conv + h_conv

        # 分割成 4 个部分: input gate, forget gate, cell input, output gate
        cc_i, cc_f, cc_c, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 2. 计算各个门 (对应论文 Eq. 3)
        # i_t = sigmoid(conv_x_i + conv_h_i + W_ci * c_prev + b_i)
        i = torch.sigmoid(cc_i + self.W_ci * c_cur)

        # f_t = sigmoid(conv_x_f + conv_h_f + W_cf * c_prev + b_f)
        f = torch.sigmoid(cc_f + self.W_cf * c_cur)

        # c_tilde = tanh(conv_x_c + conv_h_c + b_c) -> 注意：论文中 cell input 没有 peephole
        c_tilde = torch.tanh(cc_c)

        # 3. 更新 Cell State
        # C_t = f_t * C_prev + i_t * c_tilde
        c_next = f * c_cur + i * c_tilde

        # 4. 计算 Output Gate 和 Hidden State
        # o_t = sigmoid(conv_x_o + conv_h_o + W_co * C_t + b_o) -> 注意这里用的是新的 C_t
        o = torch.sigmoid(cc_o + self.W_co * c_next)

        # H_t = o_t * tanh(C_t)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_x.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_x.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True,
                 return_all_layers=False):
        """
        多层 ConvLSTM
        :param input_dim: 输入通道数
        :param hidden_dim: 隐藏层通道数 (可以是列表，对应每层的 hidden_dim)
        :param kernel_size: 卷积核大小
        :param num_layers: 层数
        :param batch_first: 输入数据格式是否为 (Batch, Time, Channels, Height, Width)
        :param return_all_layers: 是否返回所有层的状态
        """
        super(ConvLSTM, self).__init__()

        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        # 如果 hidden_dim 是整数，则每层都一样；如果是列表，则对应每层
        if isinstance(hidden_dim, int):
            self.hidden_dim = [hidden_dim] * num_layers
        else:
            self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=kernel_size,
                                          bias=bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        :param input_tensor: (B, T, C, H, W) if batch_first=True
        :return: last_layer_output, last_state_list
        """
        if not self.batch_first:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, t, _, h, w = input_tensor.size()

        # 初始化隐藏状态
        if hidden_state is None:
            hidden_state = self._init_hidden(b, (h, w))

        layer_output_list = []  # 保存每一层的所有时间步输出
        last_state_list = []  # 保存每一层的最后时间步状态

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h_cur, c_cur = hidden_state[layer_idx]
            output_inner = []

            for time_idx in range(t):
                # 获取当前时间步的输入
                # 对于第一层，输入是 input_tensor 的切片
                # 对于后续层，输入是上一层当前时间步的输出
                if layer_idx == 0:
                    x_t = cur_layer_input[:, time_idx, :, :, :]
                else:
                    x_t = cur_layer_input[:, time_idx, :, :, :]  # 这里取的是上一层的输出序列

                h_cur, c_cur = self.cell_list[layer_idx](x_t, (h_cur, c_cur))
                output_inner.append(h_cur)

            # 将该层所有时间步的输出堆叠起来，作为下一层的输入
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output  # 更新为下一层的输入

            layer_output_list.append(layer_output)
            last_state_list.append((h_cur, c_cur))

        if self.return_all_layers:
            return layer_output_list, last_state_list
        else:
            return layer_output_list[-1], last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class ConvLSTMCoupler(nn.Module):
    def __init__(self, input_ant_size, input_car_size, ant_size, car_size, depth_pred, hidden_dim=512):
        super(ConvLSTMCoupler, self).__init__()
        self.ant_size = ant_size
        self.car_size = car_size
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_ant_size * 2, ant_size * 2)
        self.fc2 = nn.Linear(input_car_size * 2, car_size * 2)

        self.past_embedding = nn.parameter.Parameter(torch.FloatTensor(self.ant_size, self.car_size, 2),
                                                     requires_grad=True)
        self.now_embedding = nn.parameter.Parameter(torch.FloatTensor(self.ant_size, self.car_size, 2),
                                                    requires_grad=True)
        self.past_embedding.data.fill_(0)
        self.now_embedding.data.fill_(0)

        self.convlstm = ConvLSTM(input_dim=2, hidden_dim=hidden_dim, kernel_size=3,
                                 num_layers=depth_pred, batch_first=True)

        self.output_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=2, kernel_size=1)
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

        x_total = rearrange(x_total, 'b len ant car c -> b len c ant car')   # 类似图像的通道顺序
        x_total, _ = self.convlstm(x_total)
        out = x_total[:, -1, ...]   # (b, hidden_dim, ant, car)
        out = self.output_conv(out)   # (b, 2, ant, car)
        out = rearrange(out, 'b c ant car -> b ant car c')
        out = rearrange(out, 'b ant car c -> b car (ant c)')
        out = self.fc_reverse1(out)
        out = rearrange(out, 'b car (ant c) -> b ant (car c)', c=2)
        out = self.fc_reverse2(out)
        out = rearrange(out, 'b ant (car c) -> b ant car c', c=2)

        return out




# # --- 使用示例 ---
# if __name__ == '__main__':
#     # 假设输入：Batch=2, Time=5, Channels=3 (RGB), Height=64, Width=64
#     input_tensor = torch.randn(2, 5, 3, 64, 64)
#
#     # 定义模型：输入通道3，隐藏通道64，卷积核3x3，2层堆叠
#     convlstm = ConvLSTM(input_dim=3, hidden_dim=64, kernel_size=3, num_layers=2)
#
#     # 前向传播
#     output, last_states = convlstm(input_tensor)
#
#     # 输出检查
#     # output shape 应为 (2, 5, 64, 64, 64) -> (Batch, Time, Hidden, H, W)
#     print("Output shape:", output.shape)
#
#     # 检查状态
#     h_last, c_last = last_states[-1]
#     print("Last hidden state shape:", h_last.shape)
