# Coupler-Channel-Deduction

Official implementation for Coupler-based wireless channel deduction, corresponding to the paper:

**Full-Domain Coupler: A Wireless Native Neural Backbone for Channel Representation and Deduction**  
Zirui Chen, Ziqing Xing, Zhaoyang Zhang, Hongning Ruan, Yuzhi Yang, Zhaohui Yang, Chongwen Huang, Mérouane Debbah.

This repository supports channel deduction tasks on simulation and measured datasets, including estimation/interpolation/prediction-like settings.

Supported model schemes:

- `conv_coupler`
- `acoupler`
- `afcoupler`
- `acdnet`
- `rcdnet`
- `convlstm`
- `lmmse` (testing baseline)

## 1. Repository Structure

```text
.
├── conf/
│   ├── train.yaml                   # Training configuration template
│   └── test.yaml                    # Testing configuration template
├── models/                          # Model definitions and model factory
├── utils/
│   ├── dataset.py                   # DeepMIMO / DICHASUS dataset loaders
│   ├── metrics.py                   # NMSE, correlation, and related metrics
│   ├── lr_scheduler.py              # Learning-rate schedulers
│   └── trace_smooth_impl.py         # Trace-smoothing evaluation utilities
├── data_prepare_DeepMIMO/
│   ├── data_division.py             # DeepMIMO raw MAT -> train/test splits
│   └── data_division_specifictest.py# Mobile/static specific-test splits
├── data_prepare_DICHASUS/
│   ├── gen_dataset_DICHASUS.py      # TFRecord -> merged source H5
│   ├── split_train_test_src_data.py # Source H5 -> train.h5/test.h5
│   ├── build_neighborhood_dataset.py# KNN neighborhood datasets
│   ├── vis_data_distribution.py     # Data distribution visualization
│   └── visualization.py             # Additional DICHASUS visualization helpers
├── main.py                          # Unified train/test launcher
├── trainer.py                       # Training, validation, checkpointing
├── tester.py                        # Offline evaluation tasks
├── base_trainer.py                  # Shared model/DDP/checkpoint utilities
├── environment.yml                  # Conda environment definition
├── extract_datasets.py              # Dataset archive extraction with root remap
├── extract_exp.py                   # Pretrained experiment archive extraction
├── plot_figs_DeepMIMO.ipynb         # DeepMIMO result visualization
├── plot_figs_DICHASUS.ipynb         # DICHASUS result visualization
├── train_DeepMIMO.sh                # DeepMIMO training example
├── test_DeepMIMO.sh                 # DeepMIMO testing example
├── train_DICHASUS.sh                # DICHASUS training example
└── test_DICHASUS.sh                 # DICHASUS testing example
```

## 2. Environment

The default environment is defined in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate <env-name>
```

Main dependencies include PyTorch, NumPy, SciPy, HDF5 utilities, TensorBoard, scikit-learn, Matplotlib, YAML parsing, `calflops`, and TensorFlow. TensorFlow is required for DICHASUS TFRecord preprocessing; it is not needed for ordinary model training/testing if prepared H5 files are already available.

Adjust the PyTorch/CUDA build according to the local GPU driver.

## 3. Data

### 3.1 DeepMIMO-O1

Expected O1 raw-data root:

```text
data/DeepMIMO/O1/32antx32car_3p5GHz_40MHz_R501-1400_V1_23-9-25/
```

The project expects raw MAT files under `mat/`. The trace-smoothing tests also use `x_trace_smooth.npy` and `y_trace_smooth.npy` under the same O1 directory.

Relevant scripts:

- `data_prepare_DeepMIMO/data_division.py`: builds train/test `.npy` splits from raw MAT files.
- `data_prepare_DeepMIMO/data_division_specifictest.py`: builds `specifictest_mobile/` and `specifictest_static/` splits.
- `extract_datasets.py`: extracts provided dataset archives, including root-name remapping.

The provided DeepMIMO archive can be downloaded from:

https://drive.google.com/drive/folders/1txq8lt7TbZAF9zNReNRpWBAvzefrBs-I?usp=sharing

Example extraction command:

```bash
python extract_datasets.py --datasets DeepMIMO-O1
```

### 3.2 DICHASUS-015x

Official dataset page:

https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-015x/

Expected raw TFRecord location:

```text
data/DICHASUS/src/015x/
```

Relevant scripts:

- `data_prepare_DICHASUS/gen_dataset_DICHASUS.py`: parses TFRecords, applies calibration/reordering/cropping, and writes merged source H5 files.
- `data_prepare_DICHASUS/split_train_test_src_data.py`: splits source H5 files into `train.h5` and `test.h5`.
- `data_prepare_DICHASUS/build_neighborhood_dataset.py`: builds KNN-neighborhood datasets such as `train_nearest32_seq36000.h5` and `test_nearest32_seq36000.h5`.
- `data_prepare_DICHASUS/vis_data_distribution.py` and `visualization.py`: visualization helpers.

## 4. Training and Testing

Training and testing are launched through `main.py`. The shell scripts provide concrete command templates for the two datasets.

```bash
bash train_DeepMIMO.sh
bash test_DeepMIMO.sh

bash train_DICHASUS.sh
bash test_DICHASUS.sh
```

`main.py` reads a YAML config and supports runtime overrides through `--mod`:

```bash
python main.py --conf ./conf/train.yaml --gpu 0 --mod \
scheme=afcoupler \
input_ant_size=3 \
input_car_size=3 \
batch_size=100 \
micro_batch=100 \
exp_dir=./exp/DeepMIMO/example_run
```

For training, `main.py` also supports DDP when launched with `torchrun`. Testing is implemented as a single-process path in `tester.py`.

## 5. Configuration and Outputs

### 5.1 Training Configuration

Main fields in `conf/train.yaml`:

- `scheme`: model scheme.
- `exp_dir`: output directory.
- `dataset_type`: `DeepmimoO1` or `DICHASUS`.
- `data_dirs`: dataset root list.
- `const_h`: channel scaling factor (`10000` for DeepMIMO, typically `1` for DICHASUS).
- `input_ant_size`, `input_car_size`: sparse observed antenna/carrier grid size.
- `batch_size`, `micro_batch`: logical batch size and gradient-accumulation micro batch.
- `learning_rate`, `lr_scheduler.*`: optimizer schedule settings.

Training outputs under `exp_dir`:

- `model.pth`: final checkpoint.
- `ckpts/ckpt_*.pth`: periodic checkpoints.
- `log/train_metrics.csv`, `log/val_metrics.csv`, `log/model_complexity.csv`.
- TensorBoard logs under `log/` when enabled.

### 5.2 Testing Configuration

Main fields in `conf/test.yaml`:

- `checkpoint_path`: model checkpoint.
- `activate_tasks`: enabled test tasks.
- `test_tasks.*`: per-task parameters.

Supported test tasks include:

- `complexity`
- `standard`
- `history_length`
- `production_noise`
- `trace_smooth`

Testing outputs under the test `exp_dir` include per-task CSV files such as `standard.csv`, `history_len.csv`, and `production_noise.csv`, trace-smoothing results under `trace_smooth/`, and `complexity_test.csv`.

## 6. Visualization and Pretrained Results

Visualization notebooks:

- `plot_figs_DeepMIMO.ipynb`: compares DeepMIMO model performance, input sparsity, history length, robustness, and trace-smoothing outputs.
- `plot_figs_DICHASUS.ipynb`: plots DICHASUS NMSE distributions, spatial trajectory results, and history-length curves.

The pretrained experiment archive `exp-pretrained.tar.gz` is available from the same Google Drive folder:

https://drive.google.com/drive/folders/1txq8lt7TbZAF9zNReNRpWBAvzefrBs-I?usp=sharing

Example extraction command:

```bash
python extract_exp.py --archive-path ./data-exp-zip/exp-pretrained.tar.gz
```

After extraction, checkpoints and result files under `exp/...` can be referenced by the test scripts and visualization notebooks.

## 7. Citation

If you use this code, data, or pretrained results in your research, please cite:

Z. Chen, Z. Xing, Z. Zhang, H. Ruan, Y. Yang, Z. Yang, C. Huang, and M. Debbah, "Full-Domain Coupler: A Wireless Native Neural Backbone for Channel Representation and Deduction," arXiv:2606.22038, 2026. [Online]. Available: https://arxiv.org/abs/2606.22038

```bibtex
@misc{chen2026fulldomaincoupler,
  title={Full-Domain Coupler: A Wireless Native Neural Backbone for Channel Representation and Deduction},
  author={Chen, Zirui and Xing, Ziqing and Zhang, Zhaoyang and Ruan, Hongning and Yang, Yuzhi and Yang, Zhaohui and Huang, Chongwen and Debbah, Mérouane},
  year={2026},
  eprint={2606.22038},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  doi={10.48550/arXiv.2606.22038},
  url={https://arxiv.org/abs/2606.22038}
}
```
