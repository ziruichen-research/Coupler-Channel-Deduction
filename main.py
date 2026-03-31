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


# def get_exp_conf(args):
#     # 1. 创建临时配置文件
#     _conf_path = f'_{os.getpid()}.yaml'
#     os.system(f'cp {args.conf} {_conf_path}')
#
#     # 2. 解析 --mod 参数
#     if args.mod is None:
#         args.mod = []
#     mod_dict = {}
#     for item_str in args.mod:
#         item = item_str.split('=')
#         mod_dict[item[0]] = '='.join(item[1:])
#
#     # 3. 读取并修改 YAML
#     with open(_conf_path, 'r') as file:
#         lines = file.readlines()
#
#     names = []
#     with open(_conf_path, 'w') as file:
#         for line in lines:
#             if ':' in line:
#                 spaced_name = line.split(':')[0]
#                 name = spaced_name.lstrip()
#                 # 计算缩进层级
#                 count = (len(spaced_name) - len(name)) // 2
#                 names = names[:count] + [name]
#                 key = '.'.join(names)
#                 if key in mod_dict:
#                     # 替换值
#                     line = f'{spaced_name}: {mod_dict[key]}\n'
#             file.write(line)
#
#     # 4. 加载最终配置
#     with open(_conf_path) as file:
#         exp_conf = yaml.safe_load(file)
#
#     # 5. 处理实验目录
#     exp_dir = exp_conf['exp_dir']
#     task = exp_conf['task']
#     conf_path = os.path.join(exp_dir, f'{task}.yaml')
#
#     # 如果目标目录已有配置，则加载旧的；否则保存新的
#     if os.path.exists(conf_path):
#         print(f'Load existing configurations: {conf_path}')
#         os.system(f'rm {_conf_path}')
#         with open(conf_path) as file:
#             exp_conf = yaml.safe_load(file)
#     else:
#         os.makedirs(exp_dir, exist_ok=True)
#         os.system(f'mv {_conf_path} {conf_path}')
#
#     return exp_conf


def get_exp_conf(args):
    # 1. 直接加载原始 YAML 为 Python 对象
    with open(args.conf, 'r') as file:
        conf = yaml.safe_load(file)

    # 2. 解析 --mod 参数并修改对象
    if args.mod:
        for item in args.mod:
            key, value_str = item.split('=', 1)

            try:
                value = yaml.safe_load(value_str)
            except:
                value = value_str

            # 递归修改配置字典
            update_config(conf, key, value)

    # 3. 处理实验目录逻辑 (保持你原有的逻辑：优先加载已存在的，否则保存新的)
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
    支持字典(dot)和列表(index)的递归更新
    例如: key_path="test_tasks[2].noise_levels"
    """
    # 将 key_path 分割为层级，同时处理列表索引 [i]
    # 正则解释: 匹配 "word" 或 "[digits]"
    keys = [k.replace('[', '').replace(']', '') for k in re.findall(r'[a-zA-Z0-9_]+|\[\d+\]', key_path)]

    current = conf
    for i, k in enumerate(keys[:-1]):
        # 判断当前键是列表索引还是字典键
        if k.isdigit():
            k = int(k)

        # 深入下一层
        if isinstance(current, list):
            current = current[k]
        else:
            current = current.get(k)

        if current is None:
            raise KeyError(f"Key path '{key_path}' not found in config.")

    # 设置最后一个键的值
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
    parser.add_argument('--conf', type=str, default='./conf/config.yaml', help='Path to config file')
    parser.add_argument('--mod', nargs='*', type=str, help='Modify config, e.g. --mod model.depth=4')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID for single card mode (ignored in DDP)')
    parser.add_argument('--cpu', type=int, default=8, help='Number of CPU threads per process')
    args = parser.parse_args()

    # 设置 CPU 线程限制，必须在任何繁重的计算或 DDP 初始化之前设置
    if args.cpu > 0:
        torch.set_num_threads(args.cpu)
        os.environ['OMP_NUM_THREADS'] = str(args.cpu)
        os.environ['MKL_NUM_THREADS'] = str(args.cpu)
        print(f"Set CPU threads to {args.cpu} per process.")

    # --- 核心：自动检测运行模式 (单卡 vs DDP) ---
    if 'WORLD_SIZE' in os.environ:
        # [DDP 模式]：由 torchrun 启动
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
        # [单卡 模式]：直接 python main.py 启动
        distributed = False
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        print(f"🚗 [Mode] Single GPU / Standard (Device: {device})")

    # 获取配置
    conf = get_exp_conf(args)

    # 实例化并运行
    if conf['task'] == 'train':
        worker = Trainer(conf, device, rank, world_size, distributed)
        worker.train()
    elif conf['task'] == 'test':
        worker = Tester(conf, device, rank, world_size, distributed)
        worker.test()
    else:
        print(f"Error: Unknown task {conf['task']}")

    # 清理 DDP
    if distributed:
        dist.destroy_process_group()

