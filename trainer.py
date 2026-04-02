import glob
import os
import random
import csv
from typing import Optional, Union, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from calflops import calculate_flops

from base_trainer import BaseTrainer
from utils.dataset import DeepmimoO1Dataset, DICHASUSDataset
from utils.metrics import mse_weighted_by_ant_snr, Nmse, Nmse_ant_snr_weighted
from utils.lr_scheduler import TransformerWarmupScheduler


class Trainer(BaseTrainer):
    """Training workflow with gradient accumulation, validation, and checkpointing."""

    def __init__(self, conf, device, rank, world_size, distributed):
        scheme = conf['scheme']
        conf['model'] = conf[scheme]

        super().__init__(conf, device, rank, world_size, distributed)
        self.const_h = self.conf.get('const_h', 10000.0)

        lr = self.conf['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.target_batch = self.conf['batch_size']
        self.micro_batch = self.conf.get('micro_batch', self.target_batch)

        if self.target_batch % self.micro_batch != 0:
            raise ValueError(f"batch_size ({self.target_batch}) must be divisible by small_batch ({self.micro_batch})")

        self.grad_accum_steps = self.target_batch // self.micro_batch

        if self.rank == 0:
            print(f"Gradient Accumulation Enabled: Target Batch={self.target_batch}, "
                  f"Micro Batch={self.micro_batch}, Accum Steps={self.grad_accum_steps}")

        self.scheduler, self.scheduler_mode = self._get_scheduler()

        self.global_step = 0
        self.start_epoch = 1

        if self.rank == 0:
            print("Initializing Validation Loaders...")
        self.train_loader, self.train_sampler = self._get_train_dataloader()
        self.val_loaders = self._init_val_loaders()

        self._setup_training_state()

        self.step_ant = self.conf['num_ant'] // self.conf['input_ant_size']
        self.step_car = self.conf['num_car'] // self.conf['input_car_size']

        # Rank 0 owns logging and complexity bookkeeping.
        if self.rank == 0:
            self.log_dir = os.path.join(self.exp_dir, 'log')
            os.makedirs(self.log_dir, exist_ok=True)

            self.writer = None
            if conf['tensorboard_writer']:
                self.writer = SummaryWriter(self.log_dir)

            self.train_csv_path = os.path.join(self.log_dir, 'train_metrics.csv')
            self.val_csv_path = os.path.join(self.log_dir, 'val_metrics.csv')
            self.complexity_csv_path = os.path.join(self.log_dir, 'model_complexity.csv')

            if not os.path.exists(self.train_csv_path):
                with open(self.train_csv_path, 'w', newline='') as f:
                    csv.writer(f).writerow(['global_step', 'epoch', 'loss', 'lr'])

            if not os.path.exists(self.val_csv_path):
                with open(self.val_csv_path, 'w', newline='') as f:
                    csv.writer(f).writerow(['epoch', 'mode', 'nmse', 'nmse_ant_snr_weighted'])

            # Keep one complexity snapshot per launch.
            self._calculate_and_log_complexity()

    def train(self):
        """Run the full training loop."""
        for epoch in range(self.start_epoch, self.conf['num_epochs'] + 1):
            if self.distributed and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.train_one_epoch(epoch, self.train_loader)

            # Epoch-based scheduler updates once per epoch.
            if self.scheduler_mode == 'epoch':
                self.scheduler.step()

            if epoch % self.conf['save_epoch_inter'] == 0:
                self.save_checkpoint(epoch, is_final=False)

            if epoch % self.conf['test_epoch_inter'] == 0:
                self.validate(epoch)

        # Export the final model in exp_dir/model.pth.
        self.save_checkpoint(self.conf['num_epochs']-1, is_final=True)

    def train_one_epoch(self, epoch, loader):
        """Run one training epoch."""
        self.model.train()

        # Show tqdm only on rank 0 to avoid duplicated logs.
        iterator = tqdm(loader, desc=f"Epoch {epoch}") if self.rank == 0 else loader

        # Reset grads once at epoch start for accumulation mode.
        self.optimizer.zero_grad()

        for i, input_batch in enumerate(iterator):
            loss_val = self.train_step(input_batch, batch_idx=i)

            # Log using optimizer-step index (global_step).
            if self.rank == 0:
                if isinstance(iterator, tqdm):
                    iterator.set_postfix({'loss': f"{loss_val:.2e}"})
                if self.global_step % self.conf['log_step_inter'] == 0:
                    self.log_metrics(loss_val, epoch)


    def train_step(self, input_batch, batch_idx):
        """Run one micro step: forward, loss, backward, and optional optimizer step."""
        if isinstance(input_batch, torch.Tensor):
            csi_data = input_batch
            snr_data = None
        else:
            csi_data = input_batch['csi']
            snr_data = input_batch['snr']   # [batch, K, ant]

        # Prepare target/current/past tensors and optional SNR slices.
        prepared_data = self.prepare_data(csi_data, snr_data=snr_data)
        h_now, h_now_input, h_past_input = prepared_data['h_now'], prepared_data['h_now_input'], prepared_data['h_past_input']
        snr_now, snr_past = prepared_data['snr_now'], prepared_data['snr_past']

        h_now_output = self.model(h_now_input, h_past_input)

        loss_type = self.conf.get('loss_type', 'mse')
        if loss_type == 'mse':
            criterion = nn.MSELoss()
            loss = criterion(h_now_output, h_now)
        elif loss_type == 'mse_weighted_by_ant_snr':
            assert snr_now is not None
            kwargs_mse_weighted_by_snr = self.conf.get('mse_weighted_by_snr', {})
            loss = mse_weighted_by_ant_snr(h_now_output, h_now, snr_now, **kwargs_mse_weighted_by_snr)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        # Scale loss so accumulated gradients match large-batch average.
        loss = loss / self.grad_accum_steps

        loss.backward()

        # Apply optimizer/scheduler only when one logical batch is complete.
        if (batch_idx + 1) % self.grad_accum_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            if self.scheduler_mode == 'step':
                self.scheduler.step()

        return loss.item() * self.grad_accum_steps

    def prepare_data(self, csi_data, snr_data=None):
        """Randomly sample current/past frames and build model inputs."""
        csi_data = csi_data.to(self.device).float() * self.const_h

        seq_len = csi_data.shape[1]
        idx_all = list(range(seq_len))

        len_seq = random.choice(idx_all) + 1
        indices = random.sample(idx_all, len_seq)
        idx_now = indices[-1]
        idx_past = indices[:-1]

        # Optional augmentation: duplicate current frame into history.
        num_cur_range = self.conf.get('num_cur_range_train', (0, 1))  # For purely mobile traces this is often [0, 0].
        for _ in range(random.randint(num_cur_range[0], num_cur_range[1])): idx_past.append(idx_now)
        random.shuffle(idx_past)

        h_now = csi_data[:, idx_now, ...]
        h_now_input = h_now[:, 0:self.step_ant * (self.conf['input_ant_size'] - 1) + 1:self.step_ant,
                            0:self.step_car * (self.conf['input_car_size'] - 1) + 1:self.step_car, :]
        h_past_input = csi_data[:, idx_past, ...]

        # Keep SNR slices aligned with selected frame indices.
        snr_now, snr_past = None, None
        if snr_data is not None:
            snr_data = snr_data.to(self.device).float()
            snr_now = snr_data[:, idx_now]
            snr_past = snr_data[:, idx_past]

        train_snrdb_range = self.conf.get('train_snrdb_range', None)
        if train_snrdb_range is not None and h_past_input.shape[1] > 0:
            # Add Gaussian noise to past input with random SNR in dB range.
            B, L, *rest = h_past_input.shape
            snr_db = torch.empty((B, L, 1, 1, 1), device=h_past_input.device).uniform_(
                train_snrdb_range[0], train_snrdb_range[1]
            )
            snr_linear = 10 ** (snr_db / 10.0)
            signal_power = torch.mean(h_past_input ** 2, dim=(2, 3, 4), keepdim=True)
            noise_power = signal_power / snr_linear
            noise_std = torch.sqrt(noise_power)
            noise = torch.randn_like(h_past_input) * noise_std
            h_past_input = h_past_input + noise

        return {
            'h_now': h_now,
            'h_now_input': h_now_input,
            'h_past_input': h_past_input,
            'snr_now': snr_now,
            'snr_past': snr_past,
        }

    def log_metrics(self, loss, epoch):
        """Write training metrics to TensorBoard and CSV."""
        current_lr = self.optimizer.param_groups[0]['lr']
        if self.writer:
            self.writer.add_scalar('Train/Loss', loss, self.global_step)
            self.writer.add_scalar('Train/LR', current_lr, self.global_step)

        with open(self.train_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.global_step, epoch, loss, current_lr])

    def validate(self, epoch):
        """Run validation across all preloaded validation modes."""
        # Keep ranks synchronized around validation to prevent DDP stragglers.
        if self.distributed:
            torch.distributed.barrier()

        # Run validation only on rank 0 to avoid duplicated compute in DDP.
        if self.distributed and self.rank != 0:
            torch.distributed.barrier()
            return

        self.model.eval()

        with torch.no_grad():
            for val_data_mode, val_loader in self.val_loaders.items():
                total_nmse = 0.0
                total_nmse_ant_weighted = 0.0
                count = 0

                for data_batch in tqdm(val_loader, desc=f"Val-{val_data_mode}"):
                    data_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]
                    if isinstance(data_batch, torch.Tensor):
                        csi_data = data_batch
                        snr_data = None
                    else:
                        csi_data = data_batch['csi']
                        snr_data = data_batch['snr']  # [batch, K, ant]

                    # Validation uses full history and last frame as target.
                    csi_data = csi_data.to(self.device).float() * self.const_h
                    index = list(range(csi_data.shape[1]))
                    index_now = index[-1]
                    index_past = index[0:-1]
                    h_now = csi_data[:, index_now, ...]
                    h_now_input = h_now[:, 0:self.step_ant * (self.conf['input_ant_size'] - 1) + 1:self.step_ant,
                                        0:self.step_car * (self.conf['input_car_size'] - 1) + 1:self.step_car, :]
                    h_past_input = csi_data[:, index_past, ...]

                    snr_now, snr_past = None, None
                    if snr_data is not None:
                        snr_data = snr_data.to(self.device).float()
                        snr_now, snr_past = snr_data[:, index_now], snr_data[:, index_past]

                    h_out = self.model(h_now_input, h_past_input)

                    # Aggregate batch metrics.
                    h_out, h_now = h_out.cpu().numpy(), h_now.cpu().numpy()
                    batch_nmse = np.mean(Nmse(h_out, h_now))
                    total_nmse += batch_nmse
                    if snr_now is None:
                        batch_nmse_ant_weighted = -1
                    else:
                        snr_now = snr_now.cpu().numpy()
                        batch_nmse_ant_weighted = np.mean(
                            Nmse_ant_snr_weighted(h_out, h_now, snr_now, **self.conf['mse_weighted_by_snr'])
                        )
                    total_nmse_ant_weighted += batch_nmse_ant_weighted
                    count += 1

                avg_nmse = total_nmse / (count + 1e-10)
                avg_nmse_snrW = total_nmse_ant_weighted / (count + 1e-10)
                if self.rank == 0:
                    print(f"Epoch {epoch} | Val-{val_data_mode} | "
                          f"NMSE: {avg_nmse:.6f} | NMSE-snrW: {avg_nmse_snrW:.6f} ")
                    if self.writer:
                        self.writer.add_scalar(f'Val-{val_data_mode}/NMSE', avg_nmse, epoch)
                        self.writer.add_scalar(f'Val-{val_data_mode}/NMSE-snrW', avg_nmse_snrW, epoch)
                    with open(self.val_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, val_data_mode, avg_nmse, avg_nmse_snrW])

        if self.distributed:
            torch.distributed.barrier()

    def _get_train_dataloader(self):
        """Build training loader, supporting multi-directory dataset concatenation."""
        data_dirs = self.conf['data_dirs']
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        max_samples = self.conf.get('max_samples', None)
        datasets = []
        for d_dir in data_dirs:
            if dataset_type == 'DeepmimoO1':
                ds = DeepmimoO1Dataset(d_dir, mode='train', max_samples=max_samples)
            elif dataset_type == 'DICHASUS':
                dichasus_conf = self.conf.get('dichasus', {})
                trainset_name = dichasus_conf.get('train_dataset_name', self.conf.get('train_dataset_name'))
                ds = DICHASUSDataset(d_dir, dataset_name=trainset_name,
                                     max_samples=max_samples, load_to_memory=True)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            if len(ds) > 0:
                datasets.append(ds)

        if len(datasets) == 0:
            raise ValueError("No valid training data found in config data_dirs!")

        full_dataset = ConcatDataset(datasets)

        if self.distributed:
            sampler = DistributedSampler(full_dataset, shuffle=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        loader = DataLoader(
            full_dataset,
            batch_size=self.micro_batch,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.conf['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        return loader, sampler

    def _get_val_dataloader(self, mode: Optional[str]='specifictest_mobile'):
        """Build one validation loader for a given mode."""
        d_dir = self.conf['data_dirs'][0]
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        if dataset_type == 'DeepmimoO1':
            assert mode is not None, "Please specify mode for DeepmimoO1 validation!"
            ds = DeepmimoO1Dataset(d_dir, mode=mode)
        elif dataset_type == 'DICHASUS':
            dichasus_conf = self.conf.get('dichasus', {})
            testset_name = dichasus_conf.get('test_dataset_name', self.conf.get('test_dataset_name'))
            ds = DICHASUSDataset(d_dir, dataset_name=testset_name, load_to_memory=True)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        loader = DataLoader(ds, batch_size=self.micro_batch, shuffle=False, num_workers=self.conf['num_workers'])
        return loader

    def _init_val_loaders(self):
        """Initialize all validation loaders once at startup."""
        dataset_type = self.conf.get('dataset_type', 'DeepmimoO1')
        loaders = {}
        if dataset_type == 'DeepmimoO1':
            loaders['static'] = self._get_val_dataloader(mode='specifictest_static')
            loaders['mobile'] = self._get_val_dataloader(mode='specifictest_mobile')
        elif dataset_type == 'DICHASUS':
            loaders['test'] = self._get_val_dataloader(mode=None)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        return loaders

    def _get_scheduler(self):
        """Create the learning-rate scheduler from config."""
        scheduler_conf = self.conf['lr_scheduler']
        scheduler_list = self.conf['lr_scheduler_list']
        stype = scheduler_conf['type']
        mode = scheduler_conf['update_mode']
        assert mode in ['epoch', 'step'], f"Invalid LR update mode {mode}!"
        params = scheduler_list[stype]

        if self.rank == 0:
            print(f"Using Scheduler: {stype} | Update Mode: {mode} | Params: {params}")

        if stype == 'TransformerWarmupScheduler':
            scheduler = TransformerWarmupScheduler(self.optimizer, **params)
        elif stype == 'StepLR':
            scheduler = StepLR(self.optimizer, **params)
        elif stype == 'MultiStepLR':
            scheduler = MultiStepLR(self.optimizer, **params)
        elif stype == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(self.optimizer, **params)
        else:
            scheduler = StepLR(self.optimizer, step_size=99999999, gamma=1.0)

        return scheduler, mode

    def _setup_training_state(self):
        """
        Initialize training state: resume from latest checkpoint if available,
        otherwise load pretrained weights if provided.
        """
        ckpt_dir = os.path.join(self.exp_dir, 'ckpts')
        resume_path = None
        if os.path.exists(ckpt_dir):
            ckpts = glob.glob(os.path.join(ckpt_dir, 'ckpt_*.pth'))
            if ckpts:
                resume_path = max(ckpts, key=os.path.getctime)

        if resume_path:
            if self.rank == 0: print(f"Resuming training from: {resume_path}")
            self.load_checkpoint(resume_path)
        elif self.conf.get('checkpoint_path'):
            bare_model = self.model.module if hasattr(self.model, 'module') else self.model
            self.load_model(bare_model, self.conf['checkpoint_path'])
        else:
            if self.rank == 0: print("No checkpoint found. Starting from scratch.")

    def _calculate_and_log_complexity(self):
        """Compute FLOPs/MACs/params and save them to CSV."""
        print("Running Model Complexity Analysis (calflops)...")
        input_ant = self.conf['input_ant_size']
        input_car = self.conf['input_car_size']
        num_ant = self.conf['num_ant']
        num_car = self.conf['num_car']

        past_len = 16
        input1 = torch.randn(1, input_ant, input_car, 2).to(self.device)
        input2 = torch.randn(1, past_len, num_ant, num_car, 2).to(self.device)
        inputs = {
            "input_x_now": input1,
            "x_past": input2
        }
        # Use the bare model to avoid DDP wrapper side effects.
        bare_model = self.model.module if hasattr(self.model, 'module') else self.model
        flops, macs, params = calculate_flops(model=bare_model, kwargs=inputs, print_results=False)
        print(f"✅ FLOPs:{flops}  MACs:{macs}  Params:{params}")

        with open(self.complexity_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Past_Len', 'FLOPs', 'MACs', 'Params'])
            writer.writerow([past_len, flops, macs, params])


    def save_checkpoint(self, epoch, is_final=False):
        """
        Save a full training checkpoint (model + optimizer + scheduler + step).
        """
        if self.rank == 0:
            save_path = self.exp_dir if is_final else os.path.join(self.exp_dir, 'ckpts')
            os.makedirs(save_path, exist_ok=True)

            bare_model = self.model.module if hasattr(self.model, 'module') else self.model
            state = {
                'epoch': epoch,
                'model': bare_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'global_step': self.global_step
            }

            filename = 'model.pth' if is_final else f'ckpt_{epoch:06d}.pth'
            torch.save(state, os.path.join(save_path, filename))
            print(f"Saved training state to {filename}")

    def load_checkpoint(self, path):
        """
        Load a full checkpoint for resume training.
        """
        state = torch.load(path, map_location=self.device)

        self.start_epoch = state['epoch'] + 1
        self.global_step = state.get('global_step', self.start_epoch * 100)

        bare_model = self.model.module if hasattr(self.model, 'module') else self.model
        bare_model.load_state_dict(state['model'])

        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])

        if self.rank == 0:
            print(f"State loaded. Resuming from Epoch {self.start_epoch}")
