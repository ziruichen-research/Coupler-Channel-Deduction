import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import re
import argparse
import yaml
import torch
import torch.distributed as dist
import numpy as np
from trainer import Trainer
from tester import Tester


def get_exp_conf(args):
    """Load config, apply CLI overrides, and persist/restore task config under exp_dir."""
    with open(args.conf, 'r') as file:
        conf = yaml.safe_load(file)

    if args.mod:
        for item in args.mod:
            key, value_str = item.split('=', 1)

            try:
                value = yaml.safe_load(value_str)
            except:
                value = value_str

            update_config(conf, key, value)

    exp_dir = conf.get('exp_dir', './exp/default')
    task = conf.get('task', 'main')
    target_conf_path = os.path.join(exp_dir, f'{task}.yaml')

    if os.path.exists(target_conf_path):
        print(f'Load existing configurations: {target_conf_path}')
        with open(target_conf_path) as file:
            conf = yaml.safe_load(file)
    else:
        os.makedirs(exp_dir, exist_ok=True)
        with open(target_conf_path, 'w') as file:
            yaml.dump(conf, file, sort_keys=False)
        print(f'Saved new configuration to: {target_conf_path}')

    return conf


def update_config(conf, key_path, value):
    """
    Update a nested config value using dot/list path syntax.

    Example:
        key_path="test_tasks[2].noise_levels"
    """
    # Split "a.b[2].c" into ["a", "b", "2", "c"].
    keys = [k.replace('[', '').replace(']', '') for k in re.findall(r'[a-zA-Z0-9_]+|\[\d+\]', key_path)]

    current = conf
    for i, k in enumerate(keys[:-1]):
        if k.isdigit():
            k = int(k)

        if isinstance(current, list):
            current = current[k]
        else:
            current = current.get(k)

        if current is None:
            raise KeyError(f"Key path '{key_path}' not found in config.")

    last_key = keys[-1]
    if last_key.isdigit():
        current[int(last_key)] = value
    else:
        current[last_key] = value


if __name__ == '__main__':
    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./conf/train.yaml', help='Path to config file')
    parser.add_argument('--mod', nargs='*', type=str, help='Modify config, e.g. --mod model.depth=4')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID for single card mode (ignored in DDP)')
    parser.add_argument('--cpu', type=int, default=8, help='Number of CPU threads per process')
    args = parser.parse_args()

    # Set CPU threading limits before heavy compute starts.
    if args.cpu > 0:
        torch.set_num_threads(args.cpu)
        os.environ['OMP_NUM_THREADS'] = str(args.cpu)
        os.environ['MKL_NUM_THREADS'] = str(args.cpu)
        print(f"Set CPU threads to {args.cpu} per process.")

    # Auto-detect execution mode: single GPU vs DDP.
    if 'WORLD_SIZE' in os.environ:
        # DDP mode launched by torchrun.
        distributed = True
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            print(f"🚀 [Mode] DDP Distributed (World Size: {world_size})")
    else:
        # Single-process mode launched by python.
        distributed = False
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        print(f"🚗 [Mode] Single GPU / Standard (Device: {device})")

    if distributed:
        # Avoid concurrent config file read/write races across ranks.
        if rank == 0:
            conf = get_exp_conf(args)
        else:
            conf = None
        obj_list = [conf]
        dist.broadcast_object_list(obj_list, src=0)
        conf = obj_list[0]
    else:
        conf = get_exp_conf(args)

    if conf['task'] == 'train':
        worker = Trainer(conf, device, rank, world_size, distributed)
        worker.train()
    elif conf['task'] == 'test':
        worker = Tester(conf, device, rank, world_size, distributed)
        worker.test()
    else:
        print(f"Error: Unknown task {conf['task']}")

    if distributed:
        dist.destroy_process_group()

