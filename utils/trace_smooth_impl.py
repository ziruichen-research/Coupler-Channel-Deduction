import os
import h5py
import numpy as np
import torch
import hdf5storage
from einops import rearrange
from tqdm import tqdm
from utils.metrics import Nmse, Corr, Nmse_ant_snr_weighted


def channel_index(x_index, y_index, path, high=181):
    """Load one DeepMIMO channel snapshot from a `.mat` file by grid index."""
    index = x_index * high + y_index
    mat_path = os.path.join(path, 'mat', 'DeepMIMO_dataset_' + str(index) + '.mat')
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Mat file not found: {mat_path}")
    channel_data_mat = hdf5storage.loadmat(mat_path)
    ans = channel_data_mat['DeepMIMO_dataset']
    return ans


def core_trace_smooth_inference(model, truth_channel_list, past_len, conf, device, snr_list=None):
    """
    Core trace-smooth inference with memory-conscious execution.

    Design:
    - Keep history tensors on CPU.
    - Move only the required current/past window to GPU.
    - Compute both AR (error propagation) and Ideal (teacher-forced history) metrics.

    Args:
        model: Prediction model.
        truth_channel_list: List of CPU tensors, each shaped [1, Ant, Car, 2].
        past_len: History length used by the model.
        conf: Configuration dict.
        device: Target compute device.
        snr_list: Optional list of CPU tensors, each shaped [1, Ant].

    Returns:
        Dict of metric series.
    """
    input_ant_size = conf['input_ant_size']
    input_car_size = conf['input_car_size']
    ant_size = conf['num_ant']
    car_size = conf['num_car']

    step_ant = ant_size // input_ant_size
    step_car = car_size // input_car_size

    # AR history is kept on CPU to avoid linear GPU memory growth.
    deduction_channel_list = []

    metrics = {
        'nmse_list_ar': [],
        'nmse_snrW_list_ar': [],
        'corr_list_ar': [],
        'nmse_list_ideal': [],
        'nmse_snrW_list_ideal': [],
        'corr_list_ideal': [],
    }

    len_trace = len(truth_channel_list)
    model.eval()

    print(f"[CoreInference] Running inference on {len_trace} frames, past_len={past_len}...")
    pbar = tqdm(range(len_trace))
    for i in pbar:
        pbar_dict = {}

        # Current ground truth (CPU -> GPU for compute).
        h_now_gt_cpu = truth_channel_list[i].cpu()
        h_now_gt_gpu = h_now_gt_cpu.to(device)
        h_now_gt = h_now_gt_gpu.cpu().detach().numpy()

        # Sparse current observation fed to the model.
        h_now_input = h_now_gt_gpu[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                                   0:step_car * (input_car_size - 1) + 1:step_car, :]
        B, Ant, Car, C = h_now_gt_gpu.shape

        # Warm-up phase: fill history with ground truth before first prediction.
        if i < past_len:
            deduction_channel_list.append(h_now_gt_cpu)
            continue

        # ----------------------
        # Part 1: AR branch
        # ----------------------
        h_past_list = deduction_channel_list[i - past_len:i]
        if len(h_past_list) == 0:
            h_past_input = torch.zeros((B, 0, Ant, Car, C), device=device)
        else:
            h_past_input = torch.stack(h_past_list, dim=0)  # [Len, B, Ant, Car, C]
            h_past_input = rearrange(h_past_input, 'len b ant car c -> b len ant car c').to(device)

        with torch.no_grad():
            h_now_output_ar = model(h_now_input, h_past_input)

        # Push prediction back to CPU history (key OOM safeguard).
        deduction_channel_list.append(h_now_output_ar.detach().cpu())

        h_now_pred_ar = h_now_output_ar.cpu().detach().numpy()
        nmse_ar = Nmse(h_now_pred_ar, h_now_gt)
        rou_ar = Corr(h_now_pred_ar, h_now_gt)
        metrics['nmse_list_ar'].append(np.mean(nmse_ar))
        metrics['corr_list_ar'].append(np.mean(rou_ar))
        pbar_dict['AR'] = f"{metrics['nmse_list_ar'][-1]:.4f}"

        if snr_list is not None:
            snr_now = snr_list[i].cpu().numpy()
            nmse_snrW_ar = Nmse_ant_snr_weighted(h_now_pred_ar, h_now_gt, snr_now)
            metrics['nmse_snrW_list_ar'].append(np.mean(nmse_snrW_ar))
            pbar_dict['AR-snrW'] = f"{metrics['nmse_snrW_list_ar'][-1]:.4f}"

        # ----------------------
        # Part 2: Ideal branch
        # ----------------------
        h_past_list_truth = truth_channel_list[i - past_len:i]
        if len(h_past_list_truth) == 0:
            h_past_input_truth = torch.zeros((B, 0, Ant, Car, C), device=device)
        else:
            h_past_input_truth = torch.stack(h_past_list_truth, dim=0)
            h_past_input_truth = rearrange(h_past_input_truth, 'len b ant car c -> b len ant car c').to(device)

        with torch.no_grad():
            h_now_output_ideal = model(h_now_input, h_past_input_truth)

        h_now_pred_ideal = h_now_output_ideal.cpu().detach().numpy()
        nmse_ideal = Nmse(h_now_pred_ideal, h_now_gt)
        rou_ideal = Corr(h_now_pred_ideal, h_now_gt)
        metrics['nmse_list_ideal'].append(np.mean(nmse_ideal))
        metrics['corr_list_ideal'].append(np.mean(rou_ideal))
        pbar_dict['Ideal'] = f"{metrics['nmse_list_ideal'][-1]:.4f}"

        if snr_list is not None:
            snr_now = snr_list[i].cpu().numpy()
            nmse_snrW_ideal = Nmse_ant_snr_weighted(h_now_pred_ideal, h_now_gt, snr_now)
            metrics['nmse_snrW_list_ideal'].append(np.mean(nmse_snrW_ideal))
            pbar_dict['Ideal-snrW'] = f"{metrics['nmse_snrW_list_ideal'][-1]:.4f}"

        if len(metrics['nmse_list_ar']) > 0 and i % 10 == 0:
            pbar.set_postfix(pbar_dict)

    if len(metrics['nmse_list_ar']) > 0:
        print(f"Average AR NMSE: {np.mean(metrics['nmse_list_ar']):.5f}")
        print(f"Average Ideal NMSE: {np.mean(metrics['nmse_list_ideal']):.5f}")
        if snr_list is not None:
            print(f"Average AR-snrW NMSE: {np.mean(metrics['nmse_snrW_list_ar']):.5f}")
            print(f"Average Ideal-snrW NMSE: {np.mean(metrics['nmse_snrW_list_ideal']):.5f}")

    return metrics


def run_trace_smooth_deepmimoO1(model, past_len, conf, device, save_dir):
    """DeepMIMO wrapper for trace-smooth evaluation."""
    path = conf['data_dirs'][0]

    x_trace = np.load(os.path.join(path, 'x_trace_smooth.npy'))
    y_trace = np.load(os.path.join(path, 'y_trace_smooth.npy'))
    len_trace = x_trace.shape[0]

    truth_channel_list = []
    const_h = conf.get('const_h', 10000.0)
    print("[DeepMIMO] Loading ground truth from .mat files...")

    for i in range(len_trace):
        reverse_x_index = x_trace[i]
        y_index = y_trace[i]
        x_index = 899 - reverse_x_index

        channel_data = channel_index(x_index, y_index, path)
        channel_data = torch.tensor(channel_data)
        channel_data = rearrange(channel_data, '(b ant) car c -> b ant car c', b=1)
        channel_data = channel_data.float() * const_h
        truth_channel_list.append(channel_data)

    metrics = core_trace_smooth_inference(model, truth_channel_list, past_len, conf, device)

    h5_path = os.path.join(save_dir, f'trace_smooth_metrics_deepmimo_pastlen{past_len}.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('nmse_list_ar', data=np.array(metrics['nmse_list_ar']))
        f.create_dataset('nmse_list_ideal', data=np.array(metrics['nmse_list_ideal']))
        f.create_dataset('corr_list_ar', data=np.array(metrics['corr_list_ar']))
        f.create_dataset('corr_list_ideal', data=np.array(metrics['corr_list_ideal']))
        f.attrs['past_len'] = past_len
        f.attrs['description'] = 'DICHASUS trace smooth results with trajectory'

    print(f"[DeepMIMO] Metrics saved to {h5_path}")


def run_trace_smooth_DICHASUS(model, past_len, conf, device, save_dir):
    """
    DICHASUS trace-smooth evaluation.

    Expected config:
        conf['dichasus']['trace_src_dataset']
        conf['dichasus']['trace_part_name']
    """
    dataset_path = conf['dichasus']['trace_src_dataset']
    part_name = conf['dichasus']['trace_part_name']
    print(f"[DICHASUS] Opening dataset: {dataset_path}")
    print(f"[DICHASUS] Target Part: {part_name}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with h5py.File(dataset_path, 'r') as f:
        # Parse part index from dataset metadata.
        if 'part_names' in f.attrs:
            part_names_str = f.attrs['part_names']
            if isinstance(part_names_str, bytes):
                part_names_str = part_names_str.decode('utf-8')
            part_list = part_names_str.split(',')
        else:
            print("[DICHASUS] Warning: 'part_names' attribute not found. Assuming single part or using all data.")
            part_list = [part_name]

        try:
            target_idx = part_list.index(part_name)
        except ValueError:
            raise ValueError(f"Part '{part_name}' not found in dataset attributes: {part_list}")

        # Select source samples for the requested part.
        if 'source_indices' in f:
            source_indices = f['source_indices'][:]
            mask = source_indices == target_idx
            print(f"[DICHASUS] Found {np.sum(mask)} samples for part {part_name}.")
        else:
            print("[DICHASUS] 'source_indices' not found. Loading ALL data.")
            mask = slice(None)

        data_complex = f['data'][()][mask]
        loc_data = f['loc'][()][mask]
        time_data = f['time'][()][mask]
        snr_data = f['snr'][()][mask]

    # Sort by timestamp for stable temporal order.
    sort_indices = np.argsort(time_data)
    data_complex = data_complex[sort_indices]
    loc_data = loc_data[sort_indices]
    time_data = time_data[sort_indices]
    snr_data = snr_data[sort_indices]

    print("[DICHASUS] Data loaded and sorted by timestamp.")

    # Keep tensors on CPU; move only required windows in core inference.
    data_tensor = torch.from_numpy(data_complex)
    data_real_imag = torch.view_as_real(data_tensor).float()   # [N, Ant, Car] complex -> [N, Ant, Car, 2]
    data_real_imag = data_real_imag.unsqueeze(1)               # [N, 1, Ant, Car, 2]
    snr_tensor = torch.from_numpy(snr_data)                    # [N, Ant]
    snr_data = snr_tensor.unsqueeze(1)                         # [N, 1, Ant]

    # Match the list-based interface expected by core_trace_smooth_inference.
    truth_channel_list = [t for t in data_real_imag]
    snr_list = [t for t in snr_data]

    metrics = core_trace_smooth_inference(model, truth_channel_list, past_len, conf, device, snr_list=snr_list)

    h5_filename = f'trace_smooth_metrics_{part_name}_pastlen{past_len}.h5'
    h5_save_path = os.path.join(save_dir, h5_filename)

    print(f"[DICHASUS] Saving metrics and trajectory to {h5_save_path}...")
    with h5py.File(h5_save_path, 'w') as f:
        f.create_dataset('nmse_list_ar', data=np.array(metrics['nmse_list_ar']))
        f.create_dataset('nmse_list_ideal', data=np.array(metrics['nmse_list_ideal']))
        f.create_dataset('nmse_snrW_list_ar', data=np.array(metrics['nmse_snrW_list_ar']))
        f.create_dataset('nmse_snrW_list_ideal', data=np.array(metrics['nmse_snrW_list_ideal']))
        f.create_dataset('corr_list_ar', data=np.array(metrics['corr_list_ar']))
        f.create_dataset('corr_list_ideal', data=np.array(metrics['corr_list_ideal']))

        # Raw trajectory metadata; metric sequences start after warm-up.
        f.create_dataset('time', data=time_data)
        f.create_dataset('loc', data=loc_data)

        f.attrs['part_name'] = part_name
        f.attrs['past_len'] = past_len
        f.attrs['mean_nmse_list_ar'] = float(np.mean(metrics['nmse_list_ar']))
        f.attrs['mean_nmse_list_ideal'] = float(np.mean(metrics['nmse_list_ideal']))
        f.attrs['mean_nmse_snrW_list_ar'] = float(np.mean(metrics['nmse_snrW_list_ar']))
        f.attrs['mean_nmse_snrW_list_ideal'] = float(np.mean(metrics['nmse_snrW_list_ideal']))
        f.attrs['description'] = 'DICHASUS trace smooth results with trajectory'

    print("[DICHASUS] Done.")
