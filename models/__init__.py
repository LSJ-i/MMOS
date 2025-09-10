from typing import Dict
import torch


def build_model(model_cfg: Dict, num_classes: int) -> torch.nn.Module:
    """
    根据配置构建 MIL 模型

    Args:
        model_cfg: dict, 包含 model 配置，例如 {"name": "mamba", "feats_dim": 384, ...}
        num_classes: int, 分类数
    Returns:
        torch.nn.Module
    """
    name = model_cfg["name"].lower()
    feats_dim = model_cfg.get("feats_dim", 384)

    if name == "abmil":
        from .abmil import GatedAttention

        return GatedAttention(feats_dim, num_classes)

    elif name == "mean":
        from .linear import Mean

        return Mean(feats_dim, num_classes)

    elif name == "max":
        from .linear import MaxMeanClass

        return MaxMeanClass(embed_dim=feats_dim, n_classes=num_classes)

    elif name == "dsmil":
        from .dsmil import MILNet, FCLayer, BClassifier

        i_classifier = FCLayer(in_size=feats_dim, out_size=num_classes)
        b_classifier = BClassifier(feats_dim, num_classes)
        return MILNet(i_classifier, b_classifier)

    elif name == "clam-mb":
        from .clam import CLAM_MB

        return CLAM_MB(feats_dim=feats_dim, n_classes=num_classes)

    elif name == "mamba":
        from .mambamil import MambaMIL

        in_dim = model_cfg.get("in_dim", feats_dim)
        dropout = model_cfg.get("dropout", 0.5)
        act = model_cfg.get("act", "gelu")
        return MambaMIL(in_dim=in_dim, n_classes=num_classes, dropout=dropout, act=act)

    elif name == "transmil":
        from .TransMIL import TransMIL

        return TransMIL(feats_dim=feats_dim, n_classes=num_classes)

    elif name == "wikg":
        from .wikg import WiKG

        return WiKG(dim_in=feats_dim, n_classes=num_classes)
    
    elif name == "mmos":
        from .mmos import MMOS
        
        return MMOS(num_classes=num_classes,hidden_size=feats_dim)
    elif name == "mmos_v2":
        from .mmos_v2 import MMOS
        
        return MMOS(num_classes=num_classes,hidden_size=feats_dim)
    else:
        raise ValueError(f"Unsupported model name: {name}")
