import os
import csv
from typing import Optional, Union, Dict

import h5py
import torch
import numpy as np
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm
from calflops import calculate_flops

from models.lmmse import LMMSEModel
from utils.dataset import DeepmimoO1Dataset, DICHASUSDataset
from utils.metrics import Nmse, Corr, Nmse_ant_snr_weighted
# from utils.trace_smooth_impl import run_trace_smooth_deepmimoO1
from base_trainer import BaseTrainer
from utils.trace_smooth_implV2 import run_trace_smooth_deepmimoO1, run_trace_smooth_DICHASUS


class Tester(BaseTrainer):
    def __init__(self, conf, device, rank, world_size, distributed):
        if distributed or world_size > 1:
            raise RuntimeError(
                "❌ Error: Tester is designed for Single GPU (Non-DDP) mode only.\n"
                "Please do NOT use 'torchrun' or set 'WORLD_SIZE' environment variable.\n"
                "Run with: python main.py --conf conf/test.yaml --gpu <id>"
            )

        # 1. 配置模型参数
        scheme = conf['scheme']
        # conf['model'] = conf[scheme]
        # [修改] 如果是 LMMSE，跳过普通模型初始化，后续单独处理
        if scheme != 'lmmse':
            conf['model'] = conf[scheme]
        else:
            conf['model'] = None  # 占位

        # 2. 初始化 BaseTrainer (构建模型结构)
        super().__init__(conf, device, rank, world_size, distributed)
        self.const_h = self.conf.get('const_h', 10000.0)

        # 3. [修改] 模型加载逻辑
        if scheme == 'lmmse':
            print("🚀 [Scheme] LMMSE Baseline Selected")
            # self._init_lmmse_model()
        else:
            # 3. 加载模型权重
            if self.conf.get('checkpoint_path'):
                # 获取裸模型 (剥离 DDP 包装)，确保权重键值匹配
                bare_model = self.model.module if hasattr(self.model, 'module') else self.model
                # 调用 BaseTrainer 的通用加载函数
                self.load_model(bare_model, self.conf['checkpoint_path'])
            else:
                if self.rank == 0:
                    print("Warning: No 'checkpoint_path' specified in config! Testing with random weights.")

        # 4. 预计算常量
        self.step_ant = self.conf['num_ant'] // self.conf['input_ant_size']
        self.step_car = self.conf['num_car'] // self.conf['input_car_size']

        # 5. [修改] 预加载数据集
        self.loaders = {}  # 用于存储 (dir, mode) -> DataLoader 的映射
        self._preload_datasets()

    # def _init_lmmse_model(self):
    #     """ [新增] 初始化 LMMSE 模型：计算相关矩阵 """
    #     # 1. 获取训练集 DataLoader 用于统计 R_xx
    #     print("====== Calculating Correlation Matrix for LMMSE ======")
    #     dichasus_conf = self.conf['dichasus']
    #     train_h5 = dichasus_conf['train_dataset_name']
    #     data_dir = self.conf['data_dirs'][0]  # 默认用第一个路径
    #
    #     if train_h5:
    #         # 可加载一部分训练数据来计算相关性，这里先用全部样本
    #         ds = DICHASUSDataset(data_dir, dataset_name=train_h5, max_samples=None, load_to_memory=True)
    #         loader = DataLoader(ds, batch_size=100, shuffle=False)
    #     else:
    #         print("Warning: No training set specified for Covariance Calc. Using Test set (Oracle LMMSE).")
    #         # 如果没指定训练集，就用测试集（Oracle LMMSE，性能上限）
    #         loader = self._create_loader(data_dir, mode='test')
    #
    #     # 2. 累积计算 R_xx = E[x x^H]
    #     num_ant = self.conf['num_ant']
    #     num_car = self.conf['num_car']
    #     total_features = num_ant * num_car
    #
    #     R_acc = torch.zeros((total_features, total_features), device=self.device, dtype=torch.complex64)
    #     N_samples = 0
    #
    #     for batch in tqdm(loader, desc="Calc Covariance"):
    #         # data_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]
    #         csi_data = batch if isinstance(batch, torch.Tensor) else batch['csi']
    #
    #         # DICHASUS: [B, Ant, Car, 2] -> [B, Ant*Car] complex
    #         csi = torch.view_as_complex(csi_data.to(self.device).float())
    #         B = csi.shape[0]
    #         flat = csi.view(B, -1)  # [B, M]
    #
    #         # R += X^H * X (Batch accumulation)
    #         # einsum: bi, bj -> ij
    #         R_batch = torch.einsum('bi,bj->ij', flat, flat.conj())
    #         R_acc += R_batch
    #         N_samples += B
    #
    #     R_xx = R_acc / N_samples
    #     print(f"Computed R_xx with {N_samples} samples.")
    #
    #     # 3. 确定输入索引 (Mask)
    #     # 根据 _slice_inputs 的逻辑生成索引
    #     indices_list = []
    #     # Python 切片逻辑: start:end:step
    #     idx_ant = list(range(0, self.step_ant * (self.conf['input_ant_size'] - 1) + 1, self.step_ant))
    #     idx_car = list(range(0, self.step_car * (self.conf['input_car_size'] - 1) + 1, self.step_car))
    #
    #     # 生成展平后的索引
    #     for a in idx_ant:
    #         for c in idx_car:
    #             flat_idx = a * num_car + c
    #             indices_list.append(flat_idx)
    #
    #     input_indices = torch.tensor(indices_list, dtype=torch.long)
    #     print(f"Input Indices count: {len(input_indices)} "
    #           f"(Expected {self.conf['input_ant_size'] * self.conf['input_car_size']})")
    #
    #     # 4. 实例化 LMMSE 伪模型
    #     self.model = LMMSEModel(R_xx, input_indices, num_ant, num_car, self.device)

    def _preload_datasets(self):
        """ 初始化时预加载所有需要的测试集 """
        print("====== Preloading Datasets ======")
        data_dirs = self.conf['data_dirs']
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')

        # 确定需要加载的模式列表
        if dataset_type == 'DeepmimoO1':   # DeepmimoO1 固定只加载这两种模式
            modes = ['specifictest_mobile', 'specifictest_static']
        else:   # DICHASUS 或其他数据集，默认使用 'test' 作为统一标识
            modes = ['test']

        # 遍历所有数据目录和模式进行预加载
        for d_dir in data_dirs:
            for mode in modes:
                print(f"Preloading: {os.path.basename(d_dir)}/{mode} ...")
                loader = self._create_loader(d_dir, mode)  # 调用原 _get_loader 改名后的内部方法
                if loader is not None:
                    self.loaders[(d_dir, mode)] = loader
                    print(f"  -> Loaded {len(loader.dataset)} samples.")
                else:
                    print(f"  -> Warning: Failed to load or empty dataset.")
        print("====== Preloading Done ======\n")

    def _create_loader(self, d_dir, mode: Optional[str] = 'specifictest_mobile'):
        """ [修改] 原 _get_loader，仅作为内部工厂方法，不再直接被 task 调用 """
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        if dataset_type == 'DeepmimoO1':
            assert mode is not None, "Please specify mode for DeepmimoO1 validation!"
            ds = DeepmimoO1Dataset(d_dir, mode=mode)
        elif dataset_type == 'DICHASUS':
            dichasus_conf = self.conf['dichasus']
            testset_name = dichasus_conf.get('test_dataset_name', 'test_nearest32_seq36000.h5')
            ds = DICHASUSDataset(d_dir, dataset_name=testset_name, load_to_memory=True)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        if len(ds) == 0:
            return None

        loader = DataLoader(
            ds,
            batch_size=self.conf['batch_size'],
            shuffle=False,
            num_workers=self.conf['num_workers'],
            pin_memory=True
        )
        return loader

    def test(self):
        """ 测试主入口：分发任务 """
        # LMMSEModel 没有 eval() 方法，判断一下
        if hasattr(self.model, 'eval'):
            self.model.eval()

        activate_tasks = self.conf.get('activate_tasks', [])
        task_list = self.conf.get('test_tasks', [])

        if not activate_tasks:
            print("Warning: No activated test tasks found in config (test_tasks)!")
            return

        with torch.no_grad():
            for task_type in activate_tasks:
                assert task_type in task_list, f"Invalid task type {task_type}!"
                print(f"\n====== Starting Task: {task_type} ======")
                task_conf = task_list[task_type]

                if task_type == 'standard':
                    self.run_standard_test(task_conf)
                elif task_type == 'history_length':
                    self.run_history_length_test(task_conf)
                elif task_type == 'production_noise':
                    self.run_production_noise_test(task_conf)
                elif task_type == 'channel_vis':   # 仅生成数据，供后续可视化
                    self.run_channel_vis_test(task_conf)
                elif task_type == 'trace_smooth':
                    self.run_trace_smooth_test(task_conf)
                elif task_type == 'complexity':  # [新增]
                    self.run_complexity_test(task_conf)
                else:
                    print(f"Unknown task type: {task_type}")

    # def _get_loader(self, d_dir, mode: Optional[str]='specifictest_mobile'):
    #     """ 获取普通的 DataLoader (单卡) """
    #     dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
    #     if dataset_type == 'DeepmimoO1':
    #         assert mode is not None, "Please specify mode for DeepmimoO1 validation!"
    #         ds = DeepmimoO1Dataset(d_dir, mode=mode)  # 默认验证集
    #     elif dataset_type == 'DICHASUS':
    #         dichasus_conf = self.conf['dichasus']
    #         testset_name = dichasus_conf.get('test_dataset_name', 'test_nearest32_seq5000.h5')
    #         ds = DICHASUSDataset(d_dir, dataset_name=testset_name, load_to_memory=True)
    #     else:
    #         raise ValueError(f"Unsupported dataset type: {dataset_type}")
    #
    #     if len(ds) == 0:
    #         return None
    #
    #     loader = DataLoader(
    #         ds,
    #         batch_size=self.conf['batch_size'],
    #         shuffle=False,
    #         num_workers=self.conf['num_workers'],
    #         pin_memory=True
    #     )
    #     return loader

    def _inference_kernel(self, h_now, h_now_input, h_past_input,
                          snr_now=None, snr_past=None):
        """ 统一的推理与指标计算核心 """
        with torch.no_grad():
            # [修改] 判断模型是否接受 snr 参数 (针对 LMMSE)
            if self.conf['scheme'] == 'lmmse':
                output = self.model(h_now_input, h_past_input=None, snr_now=snr_now)
            else:
                output = self.model(h_now_input, h_past_input)

        h_gt = h_now.cpu().numpy()
        h_pred = output.cpu().numpy()

        # 向量化计算指标 [Batch]
        nmse = Nmse(h_pred, h_gt)
        corr = Corr(h_pred, h_gt)

        if snr_now is not None:
            snr_now = snr_now.cpu().numpy()
            nmse_snrW = Nmse_ant_snr_weighted(h_pred, h_gt, snr_now, **self.conf['mse_weighted_by_snr'])
        else:
            nmse_snrW = -1.0 * np.ones_like(nmse)

        return {
            'nmse': nmse,
            'corr': corr,
            'nmse_snrW': nmse_snrW
        }

    def save_results_csv(self, filename, results_dict):
        """ 保存结果到 CSV """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        keys = list(results_dict.keys())
        if not keys: return
        length = len(results_dict[keys[0]])

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(length):
                row = [results_dict[k][i] for k in keys]
                writer.writerow(row)
        print(f"  -> Saved to {filename}")

    def _slice_inputs(self, data_batch, idx_now, idx_past):
        """ 辅助切片函数 """
        data_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]
        if isinstance(data_batch, torch.Tensor):
            csi_data = data_batch
            snr_data = None
        else:
            csi_data = data_batch['csi']
            snr_data = data_batch['snr']  # [batch, K, ant]

        csi_data = csi_data.to(self.device).float() * self.const_h
        h_now = csi_data[:, idx_now]
        h_now_input = h_now[:, 0:self.step_ant * (self.conf['input_ant_size'] - 1) + 1:self.step_ant,
                            0:self.step_car * (self.conf['input_car_size'] - 1) + 1:self.step_car, :]
        h_past_input = csi_data[:, idx_past, ...]

        if snr_data is not None:
            snr_data = snr_data.to(self.device).float()
            snr_now, snr_past = snr_data[:, idx_now], snr_data[:, idx_past]
        else:
            snr_now, snr_past = None, None

        return {
            'h_now': h_now,
            'h_now_input': h_now_input,
            'h_past_input': h_past_input,
            'snr_now': snr_now,
            'snr_past': snr_past
        }

    # ================= 具体的测试任务 =================

    def run_standard_test(self, task_conf):
        data_dirs = self.conf['data_dirs']
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        if dataset_type == 'DeepmimoO1':
            scenarios = task_conf['scenarios']
        else:   # 其他情况测试集不做区分
            scenarios = ['test']
        if 'past_len' not in task_conf:
            set_pastlen_prefix = False
            past_len = 16
        else:
            set_pastlen_prefix = True
            past_len = task_conf['past_len']
        dataset_configs = itertools.product(data_dirs, scenarios)

        for d_dir, mode in dataset_configs:
            # [修改] 从 self.loaders 获取，而非重新创建
            loader = self.loaders.get((d_dir, mode))
            if loader is None: continue

            results = {'Sample_ID': [], 'nmse': [], 'nmse-snrW': [], 'corr': []}
            dir_name = os.path.basename(d_dir.rstrip('/'))
            print(f"Running Standard Test on Dataset (past_len={past_len}): {dir_name}/{mode}...")

            start_sample_id = 0
            for data_batch in tqdm(loader):
                # data_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]
                csi_data = data_batch if isinstance(data_batch, torch.Tensor) else data_batch['csi']

                # 默认取全序列，最后一帧为当前
                index = list(range(csi_data.shape[1]))
                idx_now = index[-1]
                idx_past = index[-1-past_len: -1]
                sliced_data = self._slice_inputs(data_batch, idx_now, idx_past)

                metrics = self._inference_kernel(**sliced_data)
                nmse, corr, nmse_snrW = metrics['nmse'], metrics['corr'], metrics['nmse_snrW']

                results['Sample_ID'].extend([start_sample_id + i for i in range(len(nmse))])
                results['nmse'].extend(nmse.tolist())
                results['nmse-snrW'].extend(nmse_snrW.tolist())
                results['corr'].extend(corr.tolist())
                start_sample_id += len(nmse)

            filename = f"standard_pastlen{past_len}.csv" if set_pastlen_prefix else "standard.csv"
            self.save_results_csv(
                f"{self.exp_dir}/{dir_name}/{mode}/{filename}",
                results
            )

    def run_history_length_test(self, task_conf):
        data_dirs = self.conf['data_dirs']
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        if dataset_type == 'DeepmimoO1':
            scenarios = task_conf['scenarios']
        else:  # 其他情况测试集不做区分
            scenarios = ['test']
        dataset_configs = itertools.product(data_dirs, scenarios)
        past_lens = task_conf['past_lens']

        for d_dir, mode in dataset_configs:
            loader = self.loaders.get((d_dir, mode))
            if loader is None: continue

            results = {'Sample_ID': [], 'past_len': [], 'nmse': [], 'nmse-snrW': [], 'corr': []}
            dir_name = os.path.basename(d_dir.rstrip('/'))
            print(f"Running History Length Test on Dataset: {dir_name}/{mode}...")

            for past_len in past_lens:
                start_sample_id = 0
                for data_batch in tqdm(loader, desc=f"Past Len {past_len}"):
                    # data_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]
                    csi_data = data_batch if isinstance(data_batch, torch.Tensor) else data_batch['csi']

                    index = list(range(csi_data.shape[1]))
                    idx_now = index[-1]
                    idx_past = index[-1-past_len: -1]
                    sliced_data = self._slice_inputs(data_batch, idx_now, idx_past)

                    metrics = self._inference_kernel(**sliced_data)
                    nmse, corr, nmse_snrW = metrics['nmse'], metrics['corr'], metrics['nmse_snrW']

                    results['Sample_ID'].extend([start_sample_id + i for i in range(len(nmse))])
                    results['past_len'].extend([past_len] * len(nmse))
                    results['nmse'].extend(nmse.tolist())
                    results['nmse-snrW'].extend(nmse_snrW.tolist())
                    results['corr'].extend(corr.tolist())
                    start_sample_id += len(nmse)

            self.save_results_csv(
                f"{self.exp_dir}/{dir_name}/{mode}/history_len.csv",
                results
            )

    def run_production_noise_test(self, task_conf):
        data_dirs = self.conf['data_dirs']
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        if dataset_type == 'DeepmimoO1':
            scenarios = task_conf['scenarios']
        else:  # 其他情况测试集不做区分
            scenarios = ['test']
        dataset_configs = itertools.product(data_dirs, scenarios)
        noise_targets = task_conf['noise_targets']  # ['all', 'now', 'past']
        noise_levels = task_conf['noise_levels']
        noise_configs = list(itertools.product(noise_targets, noise_levels))

        for d_dir, mode in dataset_configs:
            loader = self.loaders.get((d_dir, mode))
            if loader is None: continue

            results = {'Sample_ID': [],  'noise_target': [], 'noise_level': [],
                       'nmse': [], 'nmse-snrW': [], 'corr': []}
            dir_name = os.path.basename(d_dir.rstrip('/'))
            print(f"Running Noise Test on {dir_name}/{mode}...")

            for target, level in noise_configs:
                start_sample_id = 0
                for data_batch in tqdm(loader, desc=f"{target} {level}"):
                    # data_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]
                    csi_data = data_batch if isinstance(data_batch, torch.Tensor) else data_batch['csi']

                    index = list(range(csi_data.shape[1]))
                    idx_now = index[-1]
                    idx_past = index[0: -1]
                    sliced_data = self._slice_inputs(data_batch, idx_now, idx_past)

                    # 应用加噪
                    if target in ['all', 'now']:
                        h_now_in = sliced_data['h_now_input']
                        noise = torch.randn_like(h_now_in) * level
                        sliced_data['h_now_input'] = h_now_in * (1 + noise)

                    if target in ['all', 'past']:
                        h_past_in = sliced_data['h_past_input']
                        noise = torch.randn_like(h_past_in) * level
                        sliced_data['h_past_input'] = h_past_in * (1 + noise)

                    metrics = self._inference_kernel(**sliced_data)
                    nmse, corr, nmse_snrW = metrics['nmse'], metrics['corr'], metrics['nmse_snrW']

                    results['Sample_ID'].extend([start_sample_id + i for i in range(len(nmse))])
                    results['noise_target'].extend([target] * len(nmse))
                    results['noise_level'].extend([level] * len(nmse))
                    results['nmse'].extend(nmse.tolist())
                    results['nmse-snrW'].extend(nmse_snrW.tolist())
                    results['corr'].extend(corr.tolist())

                    start_sample_id += len(nmse)

            self.save_results_csv(
                f"{self.exp_dir}/{dir_name}/{mode}/production_noise.csv",
                results
            )

    def run_channel_vis_test(self, task_conf):
        data_dirs = self.conf['data_dirs']
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        if dataset_type == 'DeepmimoO1':
            scenarios = task_conf['scenarios']
        else:  # 其他情况测试集不做区分
            scenarios = ['test']
        dataset_configs = itertools.product(data_dirs, scenarios)
        max_samples = task_conf.get('max_samples', 50)

        for d_dir, mode in dataset_configs:
            loader = self.loaders.get((d_dir, mode))
            if loader is None: continue

            # 准备路径
            dir_name = os.path.basename(d_dir.rstrip('/'))
            save_path = f"{self.exp_dir}/{dir_name}/{mode}/channel_recon.h5"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"Saving {max_samples} samples to: {save_path}")

            with h5py.File(save_path, 'w') as f:
                curr_idx = 0
                dset_true = f.create_dataset('true',
                                             shape=(max_samples, self.conf['num_ant'], self.conf['num_car']),
                                             dtype='complex64')
                dset_pred = f.create_dataset('pred',
                                             shape=(max_samples, self.conf['num_ant'], self.conf['num_car']),
                                             dtype='complex64')

                for data_batch in loader:
                    if curr_idx >= max_samples:
                        break

                    # 1. 模型推理
                    # input_batch = input_batch.to(self.device).float() * self.const_h
                    # data_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]
                    csi_data = data_batch if isinstance(data_batch, torch.Tensor) else data_batch['csi']

                    index = list(range(csi_data.shape[1]))
                    idx_now = index[-1]
                    idx_past = index[0: -1]
                    sliced_data = self._slice_inputs(data_batch, idx_now, idx_past)
                    h_now, h_now_in, h_past_in = \
                        sliced_data['h_now'], sliced_data['h_now_input'], sliced_data['h_past_input']

                    output = self.model(h_now_in, h_past_in)

                    # 2. 转 Numpy 并合成复数 (Vectorized)
                    true_np = h_now.cpu().numpy() / self.const_h  # [B, Ant, Car, 2]
                    pred_np = output.cpu().numpy() / self.const_h

                    # [B, Ant, Car]
                    h_true = true_np[..., 0] + 1j * true_np[..., 1]
                    h_pred = pred_np[..., 0] + 1j * pred_np[..., 1]

                    # 4. 计算写入位置和长度 (防止超出 max_samples)
                    batch_len = h_true.shape[0]
                    write_len = min(batch_len, max_samples - curr_idx)

                    # 5. 直接填入数据
                    dset_true[curr_idx: curr_idx + write_len] = h_true[:write_len]
                    dset_pred[curr_idx: curr_idx + write_len] = h_pred[:write_len]

                    curr_idx += write_len

            print(f"Done. Saved {curr_idx} samples.")

    def run_trace_smooth_test(self, task_conf):
        """
        API 包装器：调用外部实现的 Trace Smooth 测试
        """
        data_dir = self.conf['data_dirs'][0]
        dir_name = os.path.basename(data_dir.rstrip('/'))
        save_dir = f"{self.exp_dir}/{dir_name}/trace_smooth/"
        os.makedirs(save_dir, exist_ok=True)

        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        past_len_list = task_conf.get('past_len_list', [8])
        for past_len in past_len_list:
            if dataset_type == 'DeepmimoO1':
                run_trace_smooth_deepmimoO1(
                    model=self.model,
                    past_len=past_len,
                    conf=self.conf,
                    device=self.device,
                    save_dir=save_dir
                )
            elif dataset_type == 'DICHASUS':
                run_trace_smooth_DICHASUS(
                    model=self.model,
                    past_len=past_len,
                    conf=self.conf,
                    device=self.device,
                    save_dir=save_dir
                )
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

    def run_complexity_test(self, task_conf):
        """ 计算并保存模型复杂度 """
        print("Running Complexity (FLOPs/Params) Test...")

        # 1. 准备参数
        input_ant = self.conf['input_ant_size']
        input_car = self.conf['input_car_size']
        num_ant = self.conf['num_ant']
        num_car = self.conf['num_car']
        past_len = task_conf.get('past_len', 16)
        # 2. 构造输入
        input1 = torch.randn(1, input_ant, input_car, 2).to(self.device)
        input2 = torch.randn(1, past_len, num_ant, num_car, 2).to(self.device)
        inputs = {
            "input_x_now": input1,
            "x_past": input2
        }
        # 3. 计算
        flops, macs, params = calculate_flops(model=self.model, kwargs=inputs,
                                              print_results=task_conf.get('print_structure', False))
        print(f"✅ FLOPs:{flops}  MACs:{macs}  Params:{params}")
        # 4. 保存结果
        save_path = os.path.join(self.exp_dir, 'complexity_test.csv')
        os.makedirs(self.exp_dir, exist_ok=True)

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model_Scheme', 'Past_Len', 'FLOPs', 'MACs', 'Params'])
            writer.writerow([self.conf['scheme'], past_len, flops, macs, params])


























