import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    input_dim: int = 16
    hidden_dim: int = 64
    output_dim: int = 4
    num_experts: int = 4
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    num_samples: int = 8192
    top_k: int = 2


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        if self.top_k < self.num_experts:
            topk_values, topk_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
            sparse_gate_probs = torch.zeros_like(gate_probs)
            sparse_gate_probs.scatter_(dim=-1, index=topk_indices, src=topk_values)
            gate_probs = sparse_gate_probs / sparse_gate_probs.sum(dim=-1, keepdim=True)

        # 各Expertの出力を重み付きで合成する dense MoE
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed_output = torch.sum(expert_outputs * gate_probs.unsqueeze(-1), dim=1)
        return mixed_output, gate_probs


def build_synthetic_dataset(cfg: TrainConfig, device: torch.device) -> TensorDataset:
    x = torch.randn(cfg.num_samples, cfg.input_dim, device=device)

    w_teacher = torch.randn(cfg.input_dim, cfg.output_dim, device=device)
    logits_teacher = x @ w_teacher + 0.2 * torch.randn(
        cfg.num_samples, cfg.output_dim, device=device
    )
    y = torch.argmax(logits_teacher, dim=1)

    return TensorDataset(x, y)


def train(model: MoEClassifier, cfg: TrainConfig, device: torch.device) -> None:
    dataset = build_synthetic_dataset(cfg, device)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_count += batch_x.size(0)

        avg_loss = total_loss / total_count
        avg_acc = total_correct / total_count
        print(f"epoch={epoch:02d} loss={avg_loss:.4f} acc={avg_acc:.4f}")


@torch.no_grad()
def infer(model: MoEClassifier, cfg: TrainConfig, device: torch.device) -> None:
    model.eval()
    sample_x = torch.randn(8, cfg.input_dim, device=device)
    logits, gate_probs = model(sample_x)
    pred = torch.argmax(logits, dim=1)

    print("inference predictions:", pred.tolist())
    print("gate probs for first sample:", gate_probs[0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch MoE training and inference on GPU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/moe.pt")
    parser.add_argument("--inference-only", action="store_true")
    return parser.parse_args()


def save_checkpoint(model: MoEClassifier, cfg: TrainConfig, path: str) -> None:
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA対応GPUが見つかりません。GPU環境で実行してください。"
        )
    device = torch.device("cuda")
    print("using device:", device, torch.cuda.get_device_name(0))

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )

    if args.top_k <= 0 or args.top_k > args.num_experts:
        raise ValueError("--top-k は 1 以上かつ --num-experts 以下にしてください。")

    model = MoEClassifier(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
    ).to(device)

    if args.inference_only:
        model = load_model_from_checkpoint(args.checkpoint_path, device)
        infer(model, cfg, device)
        return

    train(model, cfg, device)
    save_checkpoint(model, cfg, args.checkpoint_path)
    loaded_model = load_model_from_checkpoint(args.checkpoint_path, device)
    infer(loaded_model, cfg, device)


if __name__ == "__main__":
    main()
