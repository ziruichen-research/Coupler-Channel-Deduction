import os
import random
import h5py
import hdf5storage
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    base_data_dir = '/mnt/data/ruanhongning/Data/ALLoc_data_share/' # Replacing '/home/chenzirui/Desktop/home_mnt/'
    raw_data_dir = os.path.join(base_data_dir, '32ant*32car_3.5GHz_40MHz_R501-1400_V1_24.4.12')
    data_dir = '/home/ruanhongning/repos/ALLoc-new/dataset_deepmimo/o1/'
    
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    num_ant = 32
    num_car = 32
    num_train_data = 40000
    train_data_start = 0
    num_test_data = 20000
    test_data_start = 80000

    os.makedirs(data_dir, exist_ok=True)
    mat_path = os.path.join(raw_data_dir, 'mat', 'DeepMIMO_local.mat')
    data_loc = hdf5storage.loadmat(mat_path)['DeepMIMO_loc'].astype(np.float32)
    print(data_loc.shape) # [162900, 3]

    indices = np.arange(data_loc.shape[0])
    np.random.shuffle(indices)
    for dataset_type in ['train', 'test']:
        dataset_path = os.path.join(data_dir, f'{dataset_type}.h5')
        if os.path.exists(dataset_path):
            continue
        num_data = eval(f'num_{dataset_type}_data')
        data_start = eval(f'{dataset_type}_data_start')
        dataset = h5py.File(dataset_path, 'w')
        data_dataset = dataset.create_dataset('data',
                                              (num_data, num_ant, num_car),
                                              maxshape=(None, num_ant, num_car),
                                              chunks=(1, num_ant, num_car),
                                              compression='gzip',
                                              compression_opts=8,
                                              dtype='complex64')
        loc_dataset = dataset.create_dataset('data_loc', (num_data, 3), dtype='float32')
        for i in tqdm(range(num_data), desc=dataset_type):
            j = indices[i + data_start]
            mat_path = os.path.join(raw_data_dir, 'mat', f'DeepMIMO_dataset_{j}.mat')
            data = hdf5storage.loadmat(mat_path)['DeepMIMO_dataset'].astype(np.float32)
            data_dataset[i] = data[..., 0] + 1j * data[..., 1]
            loc_dataset[i] = data_loc[j]
        dataset.close()