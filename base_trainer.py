import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from models import get_model


class BaseTrainer:
    def __init__(self, conf, device, rank, world_size, distributed):
        self.conf = conf
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.exp_dir = conf['exp_dir']

        # # TensorBoard 初始化 (仅主进程)
        # self.writer = None
        # if self.rank == 0 and conf.get('tensorboard_writer', False):
        #     log_dir = os.path.join(self.exp_dir, 'log')
        #     os.makedirs(log_dir, exist_ok=True)
        #     self.writer = SummaryWriter(log_dir)

        # 初始化模型
        self.init_model()

    def init_model(self):
        # 1. 从工厂获取模型
        model = get_model(self.conf)
        model = model.to(self.device)

        # 2. 此时暂不加载权重，权重加载逻辑由 Trainer(续训) 或 Tester(测试) 决定
        # 但如果是 Tester 或者特定的 Finetune 任务指定了 checkpoint_path，可以在这里加载基础权重
        # 为了灵活性，我们留给子类调用 load_model，或者在 DDP 包装前手动调用

        # 3. DDP 包装 (仅在分布式模式下)
        if self.distributed:
            # find_unused_parameters=False 通常能提升速度，除非你的模型有条件未执行的分支
            self.model = DDP(model, device_ids=[self.device.index], output_device=self.device.index,
                             find_unused_parameters=False)
        else:
            self.model = model

    def load_model(self, model, path):
        """
        仅加载模型权重 (用于推理或迁移学习)
        """
        if self.rank == 0:
            print(f"Loading model weights from {path} ...")

            state_dict = torch.load(path, map_location=self.device)

            # 兼容性处理：如果包含 'model' 键 (是 Trainer 保存的全状态字典)，则提取之
            if 'model' in state_dict:
                state_dict = state_dict['model']

            # 移除 DDP 可能产生的 'module.' 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            # 严格加载，有问题直接报错
            model.load_state_dict(new_state_dict, strict=True)


    # def load_checkpoint(self, path, model_instance):
    #     """
    #     加载权重。
    #     注意：我们总是在 DDP wrap 之前加载到 model_instance (bare model) 上，
    #     这样可以避免处理 'module.' 前缀的麻烦。
    #     """
    #     if self.rank == 0:
    #         print(f"Loading checkpoint from {path} ...")
    #
    #     # 映射到当前设备
    #     map_loc = self.device
    #     state_dict = torch.load(path, map_location=map_loc)
    #
    #     # 兼容性处理：去除可能存在的 'module.' 前缀 (来自旧 DDP 保存的权重)
    #     new_state_dict = {}
    #     for k, v in state_dict.items():
    #         if k.startswith('module.'):
    #             new_state_dict[k[7:]] = v
    #         else:
    #             new_state_dict[k] = v
    #
    #     # 尝试加载
    #     # try:
    #     model_instance.load_state_dict(new_state_dict, strict=True)
    #     if self.rank == 0: print("Loaded successfully (strict=True).")
    #     # except Exception as e:
    #     #     if self.rank == 0: print(f"Strict load failed, trying strict=False. Error: {e}")
    #     #     model_instance.load_state_dict(new_state_dict, strict=False)
    #
    # def save_checkpoint(self, epoch):
    #     # 仅主进程保存
    #     if self.rank == 0:
    #         save_path = os.path.join(self.exp_dir, 'ckpts')
    #         os.makedirs(save_path, exist_ok=True)
    #         path = os.path.join(save_path, f'ckpt_{epoch:06d}.pth')
    #
    #         # 获取原始模型 (解包 DDP)
    #         model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
    #
    #         # 保存
    #         torch.save(model_to_save.state_dict(), path)
    #         print(f"Saved checkpoint to {path}")
