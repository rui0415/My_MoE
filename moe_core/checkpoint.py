import os

import torch

from .model import MoEClassifier


def save_checkpoint(model: MoEClassifier, cfg, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": cfg.input_dim,
                "hidden_dim": cfg.hidden_dim,
                "output_dim": cfg.output_dim,
                "num_experts": cfg.num_experts,
                "top_k": cfg.top_k,
                "dataset": cfg.dataset,
            },
        },
        path,
    )
    print("checkpoint saved:", path)


def load_model_from_checkpoint(path: str, device: torch.device) -> MoEClassifier:
    checkpoint = torch.load(path, map_location=device)
    saved_cfg = checkpoint["config"]
    model = MoEClassifier(
        input_dim=saved_cfg["input_dim"],
        hidden_dim=saved_cfg["hidden_dim"],
        output_dim=saved_cfg["output_dim"],
        num_experts=saved_cfg["num_experts"],
        top_k=saved_cfg["top_k"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("checkpoint loaded:", path)
    return model
