import os
from typing import List, Tuple

import h5py
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.cluster import KMeans

# ------------------------
# 数据集类
# ------------------------
class MILDataset(Dataset):
    """通用 MIL 数据集，支持 .h5 / .pt / .csv 特征"""

    def __init__(
        self,
        root_dir: str,
        bag_label_list: List[Tuple[str, int]],
        file_suffix: str = ".h5",
        h5_key: str = "features",
        frac: str = "test",
    ):
        """
        Args:
            root_dir: 特征文件根目录
            bag_label_list: [(bag_name, label), ...]
            file_suffix: 特征文件后缀 (.h5/.pt/.csv)
            h5_key: 如果是 h5 文件，读取哪个 key 的数据
        """
        self.root_dir = root_dir
        self.bag_label_list = bag_label_list
        self.file_suffix = file_suffix
        self.h5_key = h5_key
        self.bag_feats_list = []
        self.bag_labels = []
        self.frac = frac
        for bag_name, label in tqdm(self.bag_label_list, desc="Loading bags"):
            file_path = os.path.join(root_dir, bag_name + file_suffix)

            if file_suffix == ".h5":
                with h5py.File(file_path, "r") as f:
                    patch_feats = torch.from_numpy(f[self.h5_key][:]).float()
                slide_path = file_path.replace("patch", "slide")
                if os.path.exists(slide_path):
                    with h5py.File(slide_path, "r") as f:
                        slide_feats = torch.from_numpy(f["features"][:]).float()
                else:
                    slide_feats = None
            elif file_suffix == ".pt":
                bag_feats = torch.load(file_path)["bag_feats"].float()
            elif file_suffix == ".csv":
                bag_feats = torch.tensor(
                    pd.read_csv(file_path, index_col=0).values, dtype=torch.float32
                )
            else:
                raise ValueError(f"Unsupported file suffix: {file_suffix}")

            patch_feats = torch.unique(patch_feats, dim=0)
            self.bag_feats_list.append((patch_feats, slide_feats))
            self.bag_labels.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self):
        return len(self.bag_label_list)

    def __getitem__(self, idx):
        if self.frac == "train":
            return (self.cluster_sampling(self.bag_feats_list[idx][0]),
                self.bag_feats_list[idx][1]), self.bag_labels[idx]
        else: 
            return self.bag_feats_list[idx], self.bag_labels[idx]
        # return self.bag_feats_list[idx], self.bag_labels[idx]

    def cluster_sampling(self, sample: torch.Tensor) -> torch.Tensor:
        # sample: [N, D] tensor
        if sample.shape[0] < 100 :
            return sample
        n_clusters = 16
        sample_np = sample.cpu().numpy()  
        method = KMeans(n_clusters=n_clusters).fit(sample_np)
        cluster_labels = method.labels_

        ratio = torch.randint(20, 100, (1,), dtype=torch.float32).item() * 0.01
        new_sample = []
        for c in range(n_clusters):
            mask = cluster_labels == c
            cluster_sample = sample[mask]
            if len(cluster_sample) == 0:
                continue
            sample_n = int(
                torch.ceil(torch.tensor(cluster_sample.shape[0] * ratio)).item()
            )
            sample_n = max(sample_n, 1)  # 确保至少选一个
            idx = torch.multinomial(
                torch.ones(len(cluster_sample)), sample_n, replacement=False
            )
            new_sample.append(cluster_sample[idx])

        return torch.cat(new_sample, dim=0)


# ------------------------
# KFold DataLoader 构建函数
# ------------------------
def build_kfold_dataloaders(cfg) -> List[Tuple[DataLoader, DataLoader]]:
    """
    构建 KFold DataLoader
    Returns:
        List[(train_loader, test_loader), ...]
    """
    dataset_cfg = cfg["dataset"]
    fold = dataset_cfg.get("k_fold", 5)
    root_dir = dataset_cfg["patch_image_dir"]
    labels_dir = dataset_cfg["label_dir"]
    target = dataset_cfg.get("target", "steatosis")
    batch_size = dataset_cfg.get("bs", 1)
    file_suffix = dataset_cfg.get("file_suffix", ".h5")
    random_seed = dataset_cfg.get("random", 7777)
    h5_key = dataset_cfg.get("h5_key", "features")

    # 读取标签
    df_labels = pd.read_csv(labels_dir, index_col=0)
    df_labels.index = df_labels.index.map(str)

    bags_list = [
        str(i) for i in df_labels.index if str(i) + file_suffix in os.listdir(root_dir)
    ]
    labels = df_labels.loc[bags_list, target].tolist()

    # 支持自定义 label 映射
    label_map = dataset_cfg.get("label_map", None)
    if label_map:
        labels = [label_map[l] for l in labels]
    data_map_list = list(zip(bags_list, labels))
    kfold = KFold(n_splits=fold, shuffle=True, random_state=random_seed)

    loaders = []
    for train_idx, test_idx in kfold.split(data_map_list):
        train_list = [data_map_list[i] for i in train_idx]
        test_list = [data_map_list[i] for i in test_idx]

        train_dataset = MILDataset(
            root_dir, train_list, frac="train", file_suffix=file_suffix, h5_key=h5_key
        )
        test_dataset = MILDataset(
            root_dir, test_list, file_suffix=file_suffix, h5_key=h5_key
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataset_cfg.get("num_workers", 4),
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataset_cfg.get("num_workers", 4),
            pin_memory=True,
        )

        loaders.append((train_loader, test_loader))

    return loaders
