import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from models import get_model


class BaseTrainer:
    """Shared model setup and weight-loading utilities for train/test workers."""

    def __init__(self, conf, device, rank, world_size, distributed):
        self.conf = conf
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.exp_dir = conf['exp_dir']

        # 初始化模型
        self.init_model()

    def init_model(self):
        """Build the model and wrap it with DDP when distributed training is enabled."""
        model = get_model(self.conf)
        model = model.to(self.device)

        # DDP 包装 (仅在分布式模式下)
        if self.distributed:
            self.model = DDP(model, device_ids=[self.device.index], output_device=self.device.index,
                             find_unused_parameters=False)
        else:
            self.model = model

    def load_model(self, model, path):
        """Load model weights only (no optimizer/scheduler state)."""
        if self.rank == 0:
            print(f"Loading model weights from {path} ...")

        # In DDP, every rank must load identical weights.
        state_dict = torch.load(path, map_location=self.device)

        # Accept both plain state_dict and full training checkpoints.
        if 'model' in state_dict:
            state_dict = state_dict['model']

        # Strip possible DDP prefix.
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
