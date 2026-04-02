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

from utils.dataset import DeepmimoO1Dataset, DICHASUSDataset
from utils.metrics import Nmse, Corr, Nmse_ant_snr_weighted
from base_trainer import BaseTrainer
from utils.trace_smooth_impl import run_trace_smooth_deepmimoO1, run_trace_smooth_DICHASUS


class Tester(BaseTrainer):
    """Offline evaluation runner for configured test tasks."""

    def __init__(self, conf, device, rank, world_size, distributed):
        if distributed or world_size > 1:
            raise RuntimeError(
                "❌ Error: Tester is designed for Single GPU (Non-DDP) mode only.\n"
                "Please do NOT use 'torchrun' or set 'WORLD_SIZE' environment variable.\n"
                "Run with: python main.py --conf conf/test.yaml --gpu <id>"
            )

        scheme = conf['scheme']
        if scheme != 'lmmse':
            conf['model'] = conf[scheme]
        else:
            conf['model'] = None

        super().__init__(conf, device, rank, world_size, distributed)
        self.const_h = self.conf.get('const_h', 10000.0)

        if scheme == 'lmmse':
            print("🚀 [Scheme] LMMSE Baseline Selected")
        else:
            if self.conf.get('checkpoint_path'):
                bare_model = self.model.module if hasattr(self.model, 'module') else self.model
                self.load_model(bare_model, self.conf['checkpoint_path'])
            else:
                if self.rank == 0:
                    print("Warning: No 'checkpoint_path' specified in config! Testing with random weights.")

        self.step_ant = self.conf['num_ant'] // self.conf['input_ant_size']
        self.step_car = self.conf['num_car'] // self.conf['input_car_size']

        self.loaders = {}
        self._preload_datasets()

    def _preload_datasets(self):
        """Preload all dataset/mode loaders used by the selected tasks."""
        print("====== Preloading Datasets ======")
        data_dirs = self.conf['data_dirs']
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')

        if dataset_type == 'DeepmimoO1':
            modes = ['specifictest_mobile', 'specifictest_static']
        else:
            modes = ['test']

        for d_dir in data_dirs:
            for mode in modes:
                print(f"Preloading: {os.path.basename(d_dir)}/{mode} ...")
                loader = self._create_loader(d_dir, mode)
                if loader is not None:
                    self.loaders[(d_dir, mode)] = loader
                    print(f"  -> Loaded {len(loader.dataset)} samples.")
                else:
                    print(f"  -> Warning: Failed to load or empty dataset.")
        print("====== Preloading Done ======\n")

    def _create_loader(self, d_dir, mode: Optional[str] = 'specifictest_mobile'):
        """Build one DataLoader for a directory/mode pair."""
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        if dataset_type == 'DeepmimoO1':
            assert mode is not None, "Please specify mode for DeepmimoO1 validation!"
            ds = DeepmimoO1Dataset(d_dir, mode=mode)
        elif dataset_type == 'DICHASUS':
            dichasus_conf = self.conf.get('dichasus', {})
            testset_name = dichasus_conf.get(
                'test_dataset_name',
                self.conf.get('test_dataset_name', 'test_nearest32_seq36000.h5')
            )
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
        """Dispatch configured test tasks."""
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
                elif task_type == 'channel_vis':
                    self.run_channel_vis_test(task_conf)
                elif task_type == 'trace_smooth':
                    self.run_trace_smooth_test(task_conf)
                elif task_type == 'complexity':
                    self.run_complexity_test(task_conf)
                else:
                    print(f"Unknown task type: {task_type}")

    def _inference_kernel(self, h_now, h_now_input, h_past_input,
                          snr_now=None, snr_past=None):
        """Run model inference and return vectorized metrics."""
        with torch.no_grad():
            # LMMSE uses current input + SNR
            if self.conf['scheme'] == 'lmmse':
                output = self.model(h_now_input, h_past_input=None, snr_now=snr_now)
            else:
                output = self.model(h_now_input, h_past_input)

        h_gt = h_now.cpu().numpy()
        h_pred = output.cpu().numpy()

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
        """Save a dictionary of result columns into a CSV file."""
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
        """Slice and move model inputs/targets for a selected time split."""
        data_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]
        if isinstance(data_batch, torch.Tensor):
            csi_data = data_batch
            snr_data = None
        else:
            csi_data = data_batch['csi']
            snr_data = data_batch['snr']

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

    # ================= Specific Test Tasks =================

    def run_standard_test(self, task_conf):
        """Run standard evaluation for each dataset and scenario."""
        data_dirs = self.conf['data_dirs']
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        if dataset_type == 'DeepmimoO1':
            scenarios = task_conf['scenarios']
        else:
            scenarios = ['test']
        if 'past_len' not in task_conf:
            set_pastlen_prefix = False
            past_len = 16
        else:
            set_pastlen_prefix = True
            past_len = task_conf['past_len']
        dataset_configs = itertools.product(data_dirs, scenarios)

        for d_dir, mode in dataset_configs:
            loader = self.loaders.get((d_dir, mode))
            if loader is None: continue

            results = {'Sample_ID': [], 'nmse': [], 'nmse-snrW': [], 'corr': []}
            dir_name = os.path.basename(d_dir.rstrip('/'))
            print(f"Running Standard Test on Dataset (past_len={past_len}): {dir_name}/{mode}...")

            start_sample_id = 0
            for data_batch in tqdm(loader):
                csi_data = data_batch if isinstance(data_batch, torch.Tensor) else data_batch['csi']

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
        """Evaluate performance under different history lengths."""
        data_dirs = self.conf['data_dirs']
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        if dataset_type == 'DeepmimoO1':
            scenarios = task_conf['scenarios']
        else:
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
        """Evaluate robustness by injecting multiplicative noise into inputs."""
        data_dirs = self.conf['data_dirs']
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        if dataset_type == 'DeepmimoO1':
            scenarios = task_conf['scenarios']
        else:
            scenarios = ['test']
        dataset_configs = itertools.product(data_dirs, scenarios)
        noise_targets = task_conf['noise_targets']
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
                    csi_data = data_batch if isinstance(data_batch, torch.Tensor) else data_batch['csi']

                    index = list(range(csi_data.shape[1]))
                    idx_now = index[-1]
                    idx_past = index[0: -1]
                    sliced_data = self._slice_inputs(data_batch, idx_now, idx_past)

                    # Multiplicative perturbation simulates production input drift.
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
        """Export reconstructed channels for downstream visualization."""
        data_dirs = self.conf['data_dirs']
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        if dataset_type == 'DeepmimoO1':
            scenarios = task_conf['scenarios']
        else:
            scenarios = ['test']
        dataset_configs = itertools.product(data_dirs, scenarios)
        max_samples = task_conf.get('max_samples', 50)

        for d_dir, mode in dataset_configs:
            loader = self.loaders.get((d_dir, mode))
            if loader is None: continue

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

                    csi_data = data_batch if isinstance(data_batch, torch.Tensor) else data_batch['csi']

                    index = list(range(csi_data.shape[1]))
                    idx_now = index[-1]
                    idx_past = index[0: -1]
                    sliced_data = self._slice_inputs(data_batch, idx_now, idx_past)
                    h_now, h_now_in, h_past_in = \
                        sliced_data['h_now'], sliced_data['h_now_input'], sliced_data['h_past_input']

                    output = self.model(h_now_in, h_past_in)

                    true_np = h_now.cpu().numpy() / self.const_h
                    pred_np = output.cpu().numpy() / self.const_h

                    h_true = true_np[..., 0] + 1j * true_np[..., 1]
                    h_pred = pred_np[..., 0] + 1j * pred_np[..., 1]

                    batch_len = h_true.shape[0]
                    write_len = min(batch_len, max_samples - curr_idx)

                    dset_true[curr_idx: curr_idx + write_len] = h_true[:write_len]
                    dset_pred[curr_idx: curr_idx + write_len] = h_pred[:write_len]

                    curr_idx += write_len

            print(f"Done. Saved {curr_idx} samples.")

    def run_trace_smooth_test(self, task_conf):
        """Run trace-smooth evaluation through the external implementation."""
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
        """Compute FLOPs/MACs/params and export a CSV summary."""
        print("Running Complexity (FLOPs/Params) Test...")

        input_ant = self.conf['input_ant_size']
        input_car = self.conf['input_car_size']
        num_ant = self.conf['num_ant']
        num_car = self.conf['num_car']
        past_len = task_conf.get('past_len', 16)
        input1 = torch.randn(1, input_ant, input_car, 2).to(self.device)
        input2 = torch.randn(1, past_len, num_ant, num_car, 2).to(self.device)
        inputs = {
            "input_x_now": input1,
            "x_past": input2
        }
        flops, macs, params = calculate_flops(model=self.model, kwargs=inputs,
                                              print_results=task_conf.get('print_structure', False))
        print(f"✅ FLOPs:{flops}  MACs:{macs}  Params:{params}")
        save_path = os.path.join(self.exp_dir, 'complexity_test.csv')
        os.makedirs(self.exp_dir, exist_ok=True)

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model_Scheme', 'Past_Len', 'FLOPs', 'MACs', 'Params'])
            writer.writerow([self.conf['scheme'], past_len, flops, macs, params])


























