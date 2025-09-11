# MMOS
Hierarchical Domain Knowledge Guided Multiple Instance Learning for Ordinal Scoring of Metabolic Dysfunction-Associated Steatohepatitis

---

## 🔧 Installation

```bash
# create and activate a conda environment
conda create -n mmos python=3.10 -y
conda activate mmos

# install Python requirements
pip install -r requirements.txt
```

---

## 📂 Dataset Layout

Organize your dataset like this:

```
dataset_root/
├── patch_features/
│   ├── case1_patch.h5
│   ├── case2_patch.h5
│   └── ...
├── slide_features/
│   ├── case1_slide.h5
│   ├── case2_slide.h5
│   └── ...
└── labels.csv
```

- `features/` contains per-case feature files. Supported formats: `.h5`, `.pt`, `.csv`.
- `labels.csv` is a CSV table with case IDs as the first column (index) and target label columns.

datasets is is currently being reorganized and anonymized to comply with medical ethics review, coming soon!
**Example `labels.csv` (index is `case_id`):**

| case_id | steatosis | inflammation | ballooning
|---------|-----------|--------------|-----------
| case1   | 1         | 1            | 1
| case2   | 2         | 0            | 1

> If you use HDF5 (`.h5`), ensure the feature matrix is stored under a consistent key (e.g. `features`). Configure the key in your config as needed.

---

## ⚙️ Configuration

Provide a config file (YAML) with dataset and training options. Minimal example (`configs/config.yaml`):

```yaml
# =====================================================
# Model configuration
# =====================================================
model:
  name: "mmos"          # model type: mmos | mamba | abmil | mean | max | dsmil | clam-mb | transmil | wikg
  feats_dim: 768        # feature dimension
  in_dim: 768           # input dimension
  dropout: 0.5          # dropout probability
  act: "gelu"           # activation function

num_class: 4            # number of classes, e.g. 2 / 3 / 5

# =====================================================
# Optimizer configuration
# =====================================================
optimizer:
  lr: 0.0002            # learning rate
  epoch: 50             # number of training epochs

# =====================================================
# Dataset configuration
# =====================================================
dataset:
  patch_image_dir: "datasets/patch_features"   # patch-level feature directory
  slide_image_dir: "datasets/slide_features"   # slide-level feature directory (optional)
  label_dir: "datasets/labels.csv"             # path to label CSV file
  target: "inflammation"                       # target column in labels.csv
  file_suffix: ".h5"                           # file extension: .h5 / .pt / .csv
  h5_key: "features"                           # dataset key in HDF5 files
  bs: 1                                        # batch size (commonly 1 in MIL)
  k_fold: 5                                    # number of folds for KFold CV
  random: 7777                                 # random seed for splitting
  num_workers: 4                               # DataLoader workers

# =====================================================
# Text embeddings (optional)
# =====================================================
text_embeddings_dir: "text_embedding/text_embeddings.h5"   # path to pre-computed text embeddings
embedding_level: "all"                                     # level: patch / slide / basecell / all

# =====================================================
# Output & logging
# =====================================================
output: "exp/mmos/inflam"            # directory to save checkpoints & logs

# =====================================================
# WandB (optional)
# =====================================================
use_wandb: False                     # whether to enable WandB logging
# wandb_project: "MIL_UHR"

# =====================================================
# Reproducibility
# =====================================================
random_seed: 7777

```

---

## 🚀 Quick Start — Training

### 1) K-Fold training (using config file)
```bash
python train.py --config exp/config.yaml
```


## 🧪 Useful Tips

- If your bag (case) contains many patches, consider enabling cluster-based sampling (or configure sampling ratio) to reduce memory usage during training.
- Keep `batch_size` small (often 1) for whole-slide-level training.
- Use `num_workers` to accelerate DataLoader I/O; set to 0 on Windows for easier debugging.

---

