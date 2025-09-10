import argparse
import logging
import os
import random
import sys
import time
from typing import Dict
import pandas as pd

import h5py
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
# import wandb

from mmdatasets import build_kfold_dataloaders
from models import build_model
from utils.metrics import calculate_metrics
from utils.text import load_text_embeddings


# ------------------------
# 工具函数
# ------------------------
def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def save_checkpoint(state: dict, filename: str):
    torch.save(state, filename)
    logging.info(f"[INFO] Checkpoint saved to {filename}")

def softlabel(bag_label, num_class, sigma):
    return torch.softmax(-torch.square(bag_label.repeat(num_class) - torch.arange(num_class)).float()/ sigma, dim=0)


# ------------------------
# 训练 & 验证函数
# ------------------------
def train_one_epoch(
    model,
    train_loader,
    base_text,
    patch_text,
    slide_text,
    criterion,
    criterion_or,
    optimizer,
    scaler,
    device,
):
    model.train()
    total_loss = 0.0
    base_text, patch_text, slide_text = (
        base_text.to(device).unsqueeze(0),
        patch_text.to(device).unsqueeze(0),
        slide_text.to(device).unsqueeze(0),
    )
    for (patch_emb, slide_emb), labels in train_loader:
        patch_emb, slide_emb, labels = (
            patch_emb.to(device),
            slide_emb.to(device).unsqueeze(0),
            labels.to(device),
        )
        optimizer.zero_grad()
        # print(patch_emb.shape, slide_emb.shape, labels.shape)
        with autocast("cuda",dtype=torch.bfloat16):
            outputs = model(base_text, patch_text, slide_text, patch_emb, slide_emb)
        loss_ce = criterion(outputs.squeeze(0), labels.long())

        loss_or = criterion_or(torch.log_softmax(outputs.squeeze(0),dim=-1), softlabel(labels.cpu(),model.num_classes,1).to(device))
        loss = 0.5*loss_ce + 0.5*loss_or
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(
    model,
    test_loader,
    base_text,
    patch_text,
    slide_text,
    criterion,
    device,
    num_classes,
):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    base_text, patch_text, slide_text = (
        base_text.to(device).unsqueeze(0),
        patch_text.to(device).unsqueeze(0),
        slide_text.to(device).unsqueeze(0),
    )
    for (patch_emb, slide_emb), labels in test_loader:
        patch_emb, slide_emb, labels = (
            patch_emb.to(device),
            slide_emb.to(device).unsqueeze(0),
            labels.to(device),
        )

        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(base_text, patch_text, slide_text, patch_emb, slide_emb)
        outputs = outputs.squeeze(0)
        loss = criterion(outputs, labels.long())
        probs = torch.softmax(outputs, dim=-1).to(torch.float32).cpu().numpy()
        total_loss += loss.item()
        preds = probs.argmax(axis=-1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    metrics = calculate_metrics(all_labels, all_preds, all_probs, num_classes)
    metrics["loss"] = total_loss / len(test_loader)
    return metrics, all_labels, all_preds, all_probs


def main(cfg_path: str):
    # ------------------------ Load config ------------------------
    cfg = yaml.safe_load(open(cfg_path, "r"))
    set_random_seed(cfg.get("random_seed", 42))
    save_dir = cfg.get("output", "runs")
    setup_logger(save_dir)
    logging.info(f"Loaded config from {cfg_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ------------------------ Dataloader ------------------------
    kfold_loaders = build_kfold_dataloaders(cfg)

    # ------------------------ Model ------------------------
    model = build_model(cfg["model"], cfg["num_class"]).to(device)

    # ------------------------ Loss & Optimizer ------------------------
    if cfg["num_class"] > 2:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    criterion_or = nn.KLDivLoss(reduction="batchmean")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg["optimizer"]["epoch"], eta_min=0
    )
    scaler = GradScaler()

    # ------------------------ Training ------------------------
    best_metrics_all_folds = []

    # load text embedding
    text_embeddings_dir = cfg['text_embeddings_dir']
    base_text, patch_text, slide_text, _ = load_text_embeddings(text_embeddings_dir, tasks=[cfg["dataset"]["target"]])
    for fold, (train_loader, test_loader) in enumerate(kfold_loaders, start=1):
        logging.info(f"========== Fold {fold} ==========")
        minimal_loss = float("inf")
        best_metrics = None
        best_acc = 0
        start_epoch = 1

        checkpoint_path = os.path.join(save_dir, f"checkpoint_fold{fold}.pth")
        # Resume from checkpoint if exists
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt["epoch"] + 1
            minimal_loss = ckpt.get("best_loss", float("inf"))
            logging.info(f"Resuming from checkpoint at epoch {start_epoch}")

        for epoch in range(start_epoch, cfg["optimizer"]["epoch"] + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                base_text,
                patch_text,
                slide_text,
                criterion,
                criterion_or,
                optimizer,
                scaler,
                device,
            )
            test_metrics, labels, preds, probs = evaluate(
                model, test_loader, base_text, patch_text, slide_text, criterion, device, cfg["num_class"]
            )

            scheduler.step()
            logging.info(
                f"Epoch [{epoch}/{cfg['optimizer']['epoch']}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_metrics['loss']:.4f}, ACC: {test_metrics['acc']:.4f}, AUC: {test_metrics['auc']:.4f}"
            )

            # Save checkpoint every epoch
            ckpt_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_loss": minimal_loss,
            }
            # save_checkpoint(ckpt_state, checkpoint_path)

            # Save best model
            if test_metrics["acc"] + test_metrics["auc"]+ test_metrics["kappa"] > best_acc:
                best_metrics = test_metrics
                best_acc = (
                    test_metrics["acc"] + test_metrics["auc"] + test_metrics["kappa"]
                )
                best_model_path = os.path.join(save_dir, f"best_model_fold{fold}.pth")
                # torch.save(model.state_dict(), best_model_path)
                logging.info(f"[INFO] New best model saved to {best_model_path}")


        best_metrics_all_folds.append(best_metrics)
        logging.info(f"Fold {fold} Best Metrics: {best_metrics}")

    # ------------------------ Save summary ------------------------

    pd.DataFrame(best_metrics_all_folds).to_csv(
        os.path.join(save_dir, "fold_results.csv"), index=False
    )
    logging.info(f"Training completed. Summary saved to {save_dir}/fold_results.csv")



# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIL Training Pipeline")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()
    main(args.config)
