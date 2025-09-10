import os
from typing import List, Tuple, Dict, Any

import h5py
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.cluster import KMeans


# ======================================================
# MIL Dataset
# ======================================================
class MILDataset(Dataset):
    """General Multiple Instance Learning (MIL) dataset.
    Supports .h5 / .pt / .csv feature files.
    """

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
            root_dir: Root directory containing feature files.
            bag_label_list: List of (bag_name, label) pairs.
            file_suffix: File extension of feature files (.h5/.pt/.csv).
            h5_key: Dataset key if file type is HDF5 (.h5).
            frac: "train" or "test", controls data return style.
        """
        self.root_dir = root_dir
        self.bag_label_list = bag_label_list
        self.file_suffix = file_suffix
        self.h5_key = h5_key
        self.frac = frac

        self.bag_feats_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.bag_labels: List[torch.Tensor] = []

        for bag_name, label in tqdm(self.bag_label_list, desc="Loading bags"):
            file_path = os.path.join(root_dir, bag_name + file_suffix)

            patch_feats, slide_feats = None, None

            if file_suffix == ".h5":
                # Load patch-level features
                with h5py.File(file_path, "r") as f:
                    patch_feats = torch.from_numpy(f[self.h5_key][:]).float()

                # Optional: load slide-level features if available
                slide_path = file_path.replace("patch", "slide")
                if os.path.exists(slide_path):
                    with h5py.File(slide_path, "r") as f:
                        slide_feats = torch.from_numpy(f["features"][:]).float()

            elif file_suffix == ".pt":
                data = torch.load(file_path)
                patch_feats = data["bag_feats"].float()

            elif file_suffix == ".csv":
                patch_feats = torch.tensor(
                    pd.read_csv(file_path, index_col=0).values, dtype=torch.float32
                )

            else:
                raise ValueError(f"Unsupported file suffix: {file_suffix}")

            # Deduplicate patch features
            patch_feats = torch.unique(patch_feats, dim=0)

            self.bag_feats_list.append((patch_feats, slide_feats))
            self.bag_labels.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.bag_label_list)

    def __getitem__(self, idx: int):
        if self.frac == "train":
            sampled_patches = self.cluster_sampling(self.bag_feats_list[idx][0])
            return (sampled_patches, self.bag_feats_list[idx][1]), self.bag_labels[idx]
        else:
            return self.bag_feats_list[idx], self.bag_labels[idx]

    def cluster_sampling(self, sample: torch.Tensor) -> torch.Tensor:
        """Cluster-based sampling for large bags.
        Args:
            sample: Tensor of shape [N, D].
        Returns:
            Sampled tensor after clustering.
        """
        if sample.shape[0] < 100:
            return sample

        n_clusters = 16
        sample_np = sample.cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters).fit(sample_np)
        cluster_labels = kmeans.labels_

        ratio = torch.randint(20, 100, (1,), dtype=torch.float32).item() * 0.01
        new_sample = []

        for c in range(n_clusters):
            mask = cluster_labels == c
            cluster_sample = sample[mask]
            if len(cluster_sample) == 0:
                continue

            sample_n = int(torch.ceil(torch.tensor(cluster_sample.shape[0] * ratio)).item())
            sample_n = max(sample_n, 1)  # At least one sample per cluster

            idx = torch.multinomial(torch.ones(len(cluster_sample)), sample_n, replacement=False)
            new_sample.append(cluster_sample[idx])

        return torch.cat(new_sample, dim=0)


# ======================================================
# KFold DataLoader Builder
# ======================================================
def build_kfold_dataloaders(cfg: Dict[str, Any]) -> List[Tuple[DataLoader, DataLoader]]:
    """Build train/test DataLoaders for KFold cross-validation.

    Args:
        cfg: Dictionary containing dataset and training configuration.

    Returns:
        List of (train_loader, test_loader) pairs for each fold.
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

    # Load labels
    df_labels = pd.read_csv(labels_dir, index_col=0)
    df_labels.index = df_labels.index.map(str)

    bags_list = [
        str(i) for i in df_labels.index if str(i) + file_suffix in os.listdir(root_dir)
    ]
    labels = df_labels.loc[bags_list, target].tolist()

    # Optional: apply label mapping
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
