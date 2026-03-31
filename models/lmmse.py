from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from utils.dataset import DICHASUSDataset, DeepmimoO1Dataset


# ==========================================
# [新增] LMMSE 算法模型封装
# ==========================================
class LMMSEModel:
    def __init__(self, R_xx, input_indices, num_ant, num_car, noise_mode='sample'):
        """
        Args:
            R_xx: 全局信道相关矩阵 [N_ant*N_car, N_ant*N_car] (绝对值尺度)
            input_indices: 观测(输入)位置在展平向量中的索引 [N_input]
            num_ant, num_car: 天线和载波数量
        """
        # 初始时，数据可能在 CPU 或 计算用的 GPU 上
        self.device = R_xx.device
        self.num_ant = num_ant
        self.num_car = num_car
        self.noise_mode = noise_mode

        # 1. 保存参数
        # 保持 R_xx 为复数
        self.R_xx = R_xx.cfloat()
        self.input_indices = input_indices.long()

        # 2. 预计算 R_xy (互相关) 和 R_yy_signal (观测信号自相关)
        # R_xy = R_xx * Phi^H (取列)
        self.R_xy = self.R_xx[:, self.input_indices]  # [M, K]
        # R_yy_signal = Phi * R_xx * Phi^H (取子矩阵)
        self.R_yy_signal = self.R_xy[self.input_indices, :]  # [K, K]

        # 3. [新增] 如果是 Global 模式，预计算每根天线的全局平均功率
        self.global_ant_power = None
        if self.noise_mode == 'global':
            # R_xx 对角线是 E[|x|^2]
            diag_R = torch.diagonal(self.R_xx).real  # [M]
            # Reshape 成 [Ant, Car] 并对载波取平均，得到每根天线的平均功率
            # 假设同一天线不同载波的大尺度衰落一致
            ant_power = diag_R.view(num_ant, num_car).mean(dim=1)  # [Ant]
            self.global_ant_power = ant_power.to(self.device)
            print(f"LMMSE Noise Mode: 'global' (Pre-calculated antenna power: {self.global_ant_power.shape})")
        else:
            print(f"LMMSE Noise Mode: 'sample' (Adaptive to input instance power)")

    def to(self, device):
        """ 模拟 nn.Module.to(device) """
        self.device = device
        self.R_xx = self.R_xx.to(device)
        self.input_indices = self.input_indices.to(device)
        self.R_xy = self.R_xy.to(device)
        self.R_yy_signal = self.R_yy_signal.to(device)
        if self.global_ant_power is not None:
            self.global_ant_power = self.global_ant_power.to(device)
        return self

    def eval(self):
        """ 模拟 nn.Module.eval() """
        pass

    def __call__(self, h_now_input, h_past_input=None, snr_now=None):
        """
        Args:
            h_now_input: [Batch, Ant_in, Car_in, 2] 观测到的部分信道
            snr_now: [Batch, Ant] 每个天线的 SNR (dB)
        Returns:
            h_est: [Batch, Ant, Car, 2]
        """
        # 确保输入数据在正确的设备上
        if h_now_input.device != self.device:
            h_now_input = h_now_input.to(self.device)

        B = h_now_input.shape[0]
        K = self.input_indices.shape[0]  # 观测维度

        # 1. 向量化输入 y: [B, K] complex
        y = torch.view_as_complex(h_now_input).reshape(B, -1)  # [B, K]

        # 2. 确定噪声协方差矩阵 Sigma_n
        if snr_now is not None:
            # # --- 步骤 A: 计算当前样本的总接收功率 (信号 + 噪声) ---
            # # [B, 1]
            # # 注意：这里的 y 包含噪声，所以这是 P_total
            # total_power_abs = torch.mean(torch.abs(y) ** 2, dim=1, keepdim=True)

            # --- 步骤 A: 对齐 SNR ---
            # 1. 计算每个观测点对应的天线 ID
            ant_indices_all = self.input_indices // self.num_car
            # 2. 获取唯一的天线 ID (并排序)
            ant_indices_unique = torch.unique(ant_indices_all, sorted=True)
            # 3. 使用索引提取 SNR
            if snr_now.device != self.device:
                snr_now = snr_now.to(self.device)
            snr_input = snr_now[:, ant_indices_unique]  # [B, Ant_in]

            # --- 步骤 C: 计算绝对噪声功率 ---
            # P_noise = P_total / (SNR + 1)
            snr_linear = 10 ** (snr_input / 10.0)  # [B, Ant_in]

            if self.noise_mode == 'sample':
                # [Sample Mode] 使用当前样本的总能量
                # 优点：自适应 Path Loss
                # 缺点：深衰落时可能低估噪声
                # [B, 1] 广播到所有天线
                power_basis = torch.mean(torch.abs(y) ** 2, dim=1, keepdim=True)

            elif self.noise_mode == 'global':
                # [Global Mode] 使用全局统计的平均能量
                # 优点：噪声基底稳定，抗深衰落
                # 缺点：对 Path Loss 差异巨大的样本可能不够精确 (除非 SNR 极其准确)

                # 提取对应输入天线的全局功率 [Ant_in]
                # self.global_ant_power 是 [Ant_total]
                global_power_selected = self.global_ant_power[ant_indices_unique]
                # 扩展到 Batch 维度 [1, Ant_in] -> [B, Ant_in]
                power_basis = global_power_selected.unsqueeze(0).expand(B, -1)

            else:
                raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

            # --- 步骤 C: 计算绝对噪声功率 ---
            # 公式: P_noise = P_total / (1 + SNR)
            # 无论是 Sample 还是 Global，我们都假设 power_basis 代表了 "信号+噪声" 的总水平
            # 对于 Global 模式，R_xx 是由含噪数据计算的，所以对角线也包含了噪声功率，使用此公式是自洽的
            noise_power_abs = power_basis / (1.0 + snr_linear)  # [B, Ant_in]

            # --- 步骤 D: 扩展到所有载波并构建对角矩阵 ---
            # 假设同一天线下不同载波的噪声功率相同 (平坦噪声谱)
            num_car_in = h_now_input.shape[2]
            # [B, Ant_in, 1] -> [B, Ant_in, Car_in] -> [B, K]
            noise_var_flat = noise_power_abs.unsqueeze(-1).repeat(1, 1, num_car_in).reshape(B, -1)

            # Sigma_n: [B, K, K]
            Sigma_n = torch.diag_embed(noise_var_flat).cfloat()

        else:
            # 无 SNR 信息时，给一个极小的正则化项
            Sigma_n = torch.eye(K, device=self.device).unsqueeze(0).repeat(B, 1, 1).cfloat() * 1e-4

        # 3. LMMSE 核心公式
        # W = R_xy * (R_yy + Sigma_n)^-1

        # R_total: [B, K, K]
        R_total = self.R_yy_signal.unsqueeze(0) + Sigma_n

        # z = (R_yy + Sigma_n)^-1 * y
        # solve(A, B) -> A^-1 B
        z = torch.linalg.solve(R_total, y.unsqueeze(-1))  # [B, K, 1]

        # h_est = R_xy * z
        h_est_flat = torch.matmul(self.R_xy, z).squeeze(-1)  # [B, M]

        # 4. Reshape
        h_est = h_est_flat.view(B, self.num_ant, self.num_car)
        h_est = torch.view_as_real(h_est)

        return h_est

def get_lmmse_model(conf):
    """
    工厂函数：计算相关矩阵并返回 LMMSEModel 实例。
    不需要传入 device，内部会自动选择可用设备进行计算，
    返回的模型可以通过 .to(device) 移动。
    """
    print("====== Initializing LMMSE Model ======")

    # 0. 读取噪声模式配置
    # 默认为 'sample' 以保持向后兼容
    lmmse_conf = conf.get('lmmse', {})
    noise_mode = lmmse_conf.get('noise_mode', 'sample')
    print(f"Configured Noise Estimation Mode: {noise_mode}")

    # 自动选择计算设备 (仅用于初始化时的矩阵运算)
    calc_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Calculating Correlation Matrix (R_xx) on {calc_device}...")

    # 1. 准备数据加载器
    dichasus_conf = conf['dichasus']
    train_h5 = dichasus_conf['train_dataset_name']
    test_h5 = dichasus_conf['test_dataset_name']
    data_dir = conf['data_dirs'][0]  # 默认用第一个路径

    # 辅助函数：创建简单的 DataLoader
    def _create_temp_loader(d_dir, d_name):
        dataset_type = conf['dataset_type']
        if dataset_type == 'DICHASUS':
            # 可加载一部分训练数据来计算相关性，这里先用全部样本
            ds = DICHASUSDataset(d_dir, dataset_name=d_name, max_samples=None, load_to_memory=True)
        # elif dataset_type == 'DeepmimoO1':
        #     # DeepMIMO 逻辑
        #     m = mode if mode else 'specifictest_mobile'
        #     ds = DeepmimoO1Dataset(d_dir, mode=m)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        if len(ds) == 0: return None
        return DataLoader(ds, batch_size=100, shuffle=False)

    if train_h5:
        loader = _create_temp_loader(data_dir, d_name=train_h5)
    else:
        print("Warning: No training set specified. Using Test set for Covariance.")
        loader = _create_temp_loader(data_dir, d_name=test_h5)

    # 2. 累积计算 R_xx
    num_ant = conf['num_ant']
    num_car = conf['num_car']
    total_features = num_ant * num_car

    R_acc = torch.zeros((total_features, total_features), device=calc_device, dtype=torch.complex64)
    N_samples = 0

    for batch in tqdm(loader, desc="Calc Covariance"):
        # data_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]
        csi_data = batch if isinstance(batch, torch.Tensor) else batch['csi']
        B, S, Ant, Car, C = csi_data.shape
        csi_data = csi_data.reshape(B * S, Ant, Car, C)

        # [B, Ant, Car, 2] -> [B, Ant*Car] complex
        csi = torch.view_as_complex(csi_data.to(calc_device).float())
        B = csi.shape[0]
        flat = csi.view(B, -1)  # [B, M]

        # R += X^H * X
        R_batch = torch.einsum('bi,bj->ij', flat, flat.conj())
        R_acc += R_batch
        N_samples += B

    R_xx = R_acc / N_samples
    print(f"Computed R_xx with {N_samples} samples. (Absolute Power Scale)")

    # 3. 计算输入索引 Mask
    step_ant = conf['num_ant'] // conf['input_ant_size']
    step_car = conf['num_car'] // conf['input_car_size']

    indices_list = []
    idx_ant = list(range(0, step_ant * (conf['input_ant_size'] - 1) + 1, step_ant))
    idx_car = list(range(0, step_car * (conf['input_car_size'] - 1) + 1, step_car))

    for a in idx_ant:
        for c in idx_car:
            flat_idx = a * num_car + c
            indices_list.append(flat_idx)

    input_indices = torch.tensor(indices_list, dtype=torch.long, device=calc_device)
    print(f"Input Indices count: {len(input_indices)}")

    # 4. 创建并返回实例 (数据目前在 calc_device 上)
    model = LMMSEModel(R_xx, input_indices, num_ant, num_car, noise_mode=noise_mode)
    return model



