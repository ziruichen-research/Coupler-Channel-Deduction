import os
import random
import h5py
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    data_dict = np.load('/home/ruanhongning/repos/ALLoc-new/dataset_dichasus/data_ca01.npz')
    data = data_dict['data'].astype(np.complex64) # [num_data, 32, 32]
    data_loc = data_dict['loc'].astype(np.float32) # [num_data, 3]
    print(np.mean(np.abs(data) ** 2)) # 9.048489
    
    assignments = [
        [0,13,31,29,3,7,1,12],
        [30,26,21,25,24,8,22,15],
        [28,5,10,14,6,2,16,18],
        [19,4,23,17,20,11,9,27]
    ]
    assignments = np.ravel(assignments)
    
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    num_car = 32
    num_ant = 32
    num_train_data = 40000
    train_data_start = 0
    num_test_data = 20000
    test_data_start = 80000
    
    data_dir = f'/home/ruanhongning/repos/ALLoc-new/dataset_dichasus/ca01/'
    os.makedirs(data_dir, exist_ok=True)

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
                                              dtype='complex64')
        loc_dataset = dataset.create_dataset('data_loc', (num_data, 3), dtype='float32')
        for i in tqdm(range(num_data), desc=f'ca01 {dataset_type}'):
            j = indices[i + data_start]
            data_dataset[i] = data[j, assignments]
            loc_dataset[i] = data_loc[j]
        dataset.close()