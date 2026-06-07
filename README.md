# Coupler-Channel-Deduction

Official implementation for Coupler-based wireless channel deduction models (DeepMIMO + DICHASUS), corresponding to the paper:

**Full-Domain Staggered Coupler: A Native-AI Neural Backbone for Wireless Channel Representation and Deduction**  
Zirui Chen, Ziqing Xing, Zhaoyang Zhang, Hongning Ruan, Zhaohui Yang, Chongwen Huang, Mérouane Debbah.

This project supports channel deduction tasks (estimation/interpolation/prediction-like settings) on both simulation and measured datasets, with multiple model schemes:

- `conv_coupler`
- `acoupler`
- `afcoupler`
- `acdnet`
- `rcdnet`
- `convlstm`
- `lmmse` (testing baseline)

## 1. Project Overview

The core workflow is:

1. Prepare dataset files in the required directory layout.
2. Configure `conf/train.yaml` or override by `--mod`.
3. Run `main.py` for training/testing.
4. Analyze CSV/H5 outputs under `exp/...`.

Entry points:

- `main.py`: unified launcher (`task=train` or `task=test`), supports single GPU and DDP (training only).
- `trainer.py`: training loop, validation, checkpointing, complexity logging.
- `tester.py`: offline evaluation with multiple test tasks.
- `base_trainer.py`: shared model init / DDP wrapper / checkpoint loading.

## 2. Directory Structure

```text
.
├── conf/
│   ├── train.yaml                  # Training config template
│   └── test.yaml                   # Testing config template
├── models/                         # Model definitions and factory
├── utils/
│   ├── dataset.py                  # DeepMIMO / DICHASUS dataset loaders
│   ├── metrics.py                  # NMSE/Corr and related metrics
│   ├── lr_scheduler.py             # Scheduler implementations
│   └── trace_smooth_impl.py        # Trace smoothing evaluation helpers
├── data_prepare_DeepMIMO/
│   ├── data_division.py            # Generate train/test from DeepMIMO raw MAT files
│   └── data_division_specifictest.py # Build specific mobile/static test splits
├── data_prepare_DICHASUS/
│   ├── gen_dataset_DICHASUS.py     # TFRecord -> merged source H5
│   ├── split_train_test_src_data.py# Source H5 -> train.h5/test.h5
│   ├── build_neighborhood_dataset.py # Build KNN neighborhood training/testing sets
│   └── vis_data_distribution.py    # Data distribution visualization
├── environment.yml                 # Environment definition (single-file)
├── extract_exp.py                  # Extract pretrained exp archive (supports root remap)
├── extract_datasets.py             # Extract tar.gz with root remap (e.g., data -> data-tmp)
├── plot_figs_DeepMIMO.ipynb        # Visualization notebook for DeepMIMO test results
├── plot_figs_DICHASUS.ipynb        # Visualization notebook for DICHASUS test results
├── train_DeepMIMO.sh               # Example training command for DeepMIMO
├── test_DeepMIMO.sh                # Example testing command for DeepMIMO
├── train_DICHASUS.sh               # Example training command for DICHASUS
└── test_DICHASUS.sh                # Example testing command for DICHASUS
```

## 3. Environment

Recommended:

- Python 3.10+
- PyTorch + CUDA (for GPU training)
- TensorFlow (required by DICHASUS TFRecord preprocessing)

Common dependencies used by this repo:

- `torch`, `torchvision` (optional), `tensorboard`
- `numpy`, `scipy`, `h5py`, `hdf5storage`
- `tqdm`, `einops`, `scikit-learn`, `matplotlib`
- `pyyaml`, `calflops`, `tensorflow`

Set up environment with:

```bash
conda env create -f environment.yml
conda activate <your-env-name>
```

Notes:

- Adjust PyTorch / CUDA installation according to your machine and driver.
- If you only run model training/testing (without DICHASUS TFRecord preprocessing), TensorFlow is optional.

## 4. Data Preparation

### 4.1 DeepMIMO-O1

#### 4.1.1 Raw data source and expected layout

This project uses DeepMIMO O1 raw MAT files under:

`data/DeepMIMO/O1/32antx32car_3p5GHz_40MHz_R501-1400_V1_23-9-25/mat/`

The training/testing scripts also use:

- `x_trace_smooth.npy`
- `y_trace_smooth.npy`

under the same O1 directory.

#### 4.1.2 Using provided tar.gz package

Download `DeepMIMO-O1.tar.gz` from the Google Drive folder:

https://drive.google.com/drive/folders/1txq8lt7TbZAF9zNReNRpWBAvzefrBs-I?usp=sharing

1. Put archive at:

```bash
./data-exp-zip/DeepMIMO-O1.tar.gz
```

2. Edit `extract_datasets.py` if needed:

- `ARCHIVE_ROOT_NAME = "data"` (archive internal root)
- `EXTRACT_ROOT_NAME = "data"` or `"data-tmp"` (your target root)

3. Extract:

```bash
python extract_datasets.py --datasets DeepMIMO-O1
```

By default, the script remaps archive root by replacing the first path component (e.g., `data/...` -> `data-tmp/...`).

#### 4.1.3 Rebuild train/test splits from raw MAT

From `data_prepare_DeepMIMO/`:

1. `data_division.py`:
- split scene into spatial boxes,
- sample points per box,
- generate `train/data_*.npy`, `test/data_*.npy`.

2. `data_division_specifictest.py`:
- build `specifictest_mobile/` and `specifictest_static/` from `test/`.

Run:

```bash
cd data_prepare_DeepMIMO
python data_division.py
python data_division_specifictest.py
cd ..
```

### 4.2 DICHASUS-015x

#### 4.2.1 Download raw TFRecords

Please download from official dataset website:

- https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-015x/

Download required `dichasus-015*.tfrecords` files (and offset json if needed), then place them under:

```bash
./data/DICHASUS/src/015x/
```

#### 4.2.2 Build model-ready H5 files

Pipeline in `data_prepare_DICHASUS/`:

1. `gen_dataset_DICHASUS.py`
- parse TFRecords,
- apply STO/CPO calibration,
- reorder antennas,
- optional bandwidth crop/downsampling,
- output merged source H5 (e.g., `src_h5/.../data_0152-0157.h5`).

2. `split_train_test_src_data.py`
- split source H5 into `train.h5` and `test.h5`,
- keep source/time ordering metadata.

3. `build_neighborhood_dataset.py`
- build neighborhood samples via FPS/random center + KNN,
- output `train_nearest32_seq36000.h5` and `test_nearest32_seq36000.h5`.

4. Optional visualization:
- `vis_data_distribution.py`, `visualization.py`.

## 5. Training

### 5.1 Single-GPU training (recommended start)

DeepMIMO example:

```bash
bash train_DeepMIMO.sh
```

DICHASUS example:

```bash
bash train_DICHASUS.sh
```

Both scripts call `main.py --conf conf/train.yaml --mod ...`.

### 5.2 DDP training

`main.py` supports DDP when launched by `torchrun`.

Example (2 GPUs):

```bash
torchrun --nproc_per_node=2 main.py --conf ./conf/train.yaml --cpu 8 --mod \
scheme=afcoupler \
exp_dir=./exp/DeepMIMO/input_3antx3car/data2000/AFCoupler-DDP
```

Notes:

- DDP is for training path.
- `tester.py` is intentionally single-process only.

## 6. Testing

DeepMIMO example:

```bash
bash test_DeepMIMO.sh
```

DICHASUS example:

```bash
bash test_DICHASUS.sh
```

Testing is configured by:

- `activate_tasks` in `conf/test.yaml`
- corresponding entries in `test_tasks`

Supported tasks include `complexity`, `standard`, `history_length`, `production_noise`, `trace_smooth`.

## 7. Configuration Examples

### 7.1 Training config fields (`conf/train.yaml`)

Key fields:

- `scheme`: model name
- `exp_dir`: output directory
- `dataset_type`: `DeepmimoO1` or `DICHASUS`
- `data_dirs`: dataset root list
- `const_h`: scale factor (`10000` for DeepMIMO, typically `1` for DICHASUS)
- `input_ant_size`, `input_car_size`: sparse observed grid size
- `batch_size`, `micro_batch`: logical batch + gradient accumulation micro batch
- `learning_rate`, `lr_scheduler.*`

CLI override example:

```bash
python main.py --conf ./conf/train.yaml --gpu 0 --mod \
scheme=convlstm \
dataset_type=DICHASUS \
data_dirs=['./data/DICHASUS/h5_CD/0152-0157_DirectSample512to32'] \
const_h=1 \
input_ant_size=16 \
input_car_size=16 \
batch_size=100 \
micro_batch=100 \
num_epochs=2000 \
exp_dir=./exp/DICHASUS/example_run
```

### 7.2 Testing config fields (`conf/test.yaml`)

Key fields:

- `checkpoint_path`: trained checkpoint (`model.pth`)
- `activate_tasks`: test task list
- `test_tasks.*`: per-task parameters

CLI override example:

```bash
python main.py --conf ./conf/test.yaml --gpu 0 --mod \
scheme=afcoupler \
checkpoint_path=./exp/DeepMIMO/input_3antx3car/data2000/AFCoupler/model.pth \
exp_dir=./exp/DeepMIMO/input_3antx3car/data2000/AFCoupler/test \
activate_tasks=['standard','history_length']
```

## 8. Outputs

Training outputs (under `exp_dir`):

- `model.pth`: final checkpoint
- `ckpts/ckpt_*.pth`: periodic checkpoints
- `log/train_metrics.csv`, `log/val_metrics.csv`, `log/model_complexity.csv`
- optional TensorBoard files in `log/`

Testing outputs (under test `exp_dir`):

- per-task CSV files, e.g. `standard.csv`, `history_len.csv`, `production_noise.csv`
- trace-smooth outputs under `trace_smooth/`
- complexity summary `complexity_test.csv`

## 9. Visualization Notebooks

### 9.1 `plot_figs_DeepMIMO.ipynb`

Main content:

- Compare model performance across schemes and input sparsity settings.
- Plot history-length and robustness results from test CSV outputs.
- Visualize trace-smooth related outputs for qualitative analysis.

How to use:

1. Run DeepMIMO tests first (or extract pretrained `exp` results).
2. Open notebook: `jupyter lab plot_figs_DeepMIMO.ipynb`.
3. In the configuration cell, set experiment root/path to your local `exp/...` results.
4. Run cells top-to-bottom to generate figures and exported tables.

### 9.2 `plot_figs_DICHASUS.ipynb`

Main content:

- Grouped NMSE boxplots for different models and input configurations.
- NMSE spatial distribution on trajectories (ideal vs auto-regressive).
- NMSE time-curve comparison across history lengths, with optional LMMSE reference.

How to use:

1. Run DICHASUS tests first (or extract pretrained `exp` results).
2. Open notebook: `jupyter lab plot_figs_DICHASUS.ipynb`.
3. Update the top configuration section (result directories, model list, target `past_len`, output path).
4. Execute all cells to save figures and summary CSV files.

## 10. Use Pretrained exp

Users can download the pretrained experiment archive `exp-pretrained.tar.gz`
from the Google Drive folder:

https://drive.google.com/drive/folders/1txq8lt7TbZAF9zNReNRpWBAvzefrBs-I?usp=sharing

Then place it under:

```bash
./data-exp-zip/
```

Extract with:

```bash
python extract_exp.py --archive-path ./data-exp-zip/exp-pretrained.tar.gz
```

By default, archive root `exp/` is restored to local `exp/`.

After extraction, checkpoints in
`./exp...` can be referenced directly by test scripts.
