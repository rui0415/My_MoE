import argparse
import csv
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


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
    dataset: str = "synthetic"
    data_dir: str = "./data"
    max_train_samples: int = 0
    max_test_samples: int = 0


def plot_history(losses: list[float], accs: list[float], path: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    epochs = list(range(1, len(losses) + 1))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, losses, color="#cc5500", marker="o", label="loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="#cc5500")
    ax1.tick_params(axis="y", labelcolor="#cc5500")

    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, color="#007a7a", marker="s", label="accuracy")
    ax2.set_ylabel("accuracy", color="#007a7a")
    ax2.tick_params(axis="y", labelcolor="#007a7a")

    plt.title("MoE Training Curves")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("history plot saved:", path)


def save_history_csv(losses: list[float], accs: list[float], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "acc"])
        for i, (loss, acc) in enumerate(zip(losses, accs), start=1):
            writer.writerow([i, f"{loss:.8f}", f"{acc:.8f}"])
    print("history csv saved:", path)


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


def build_mnist_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    if cfg.max_train_samples > 0:
        train_ds = Subset(train_ds, list(range(min(cfg.max_train_samples, len(train_ds)))))
    if cfg.max_test_samples > 0:
        test_ds = Subset(test_ds, list(range(min(cfg.max_test_samples, len(test_ds)))))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    return train_loader, test_loader


def build_train_loader(cfg: TrainConfig, device: torch.device) -> DataLoader:
    if cfg.dataset == "synthetic":
        dataset = build_synthetic_dataset(cfg, device)
        return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    train_loader, _ = build_mnist_loaders(cfg)
    return train_loader


def preprocess_batch(batch_x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if batch_x.dim() == 4:
        batch_x = batch_x.flatten(start_dim=1)
    return batch_x.to(device)


def train(
    model: MoEClassifier,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    loader = build_train_loader(cfg, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []
    accs: list[float] = []
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch_x, batch_y in loader:
            batch_x = preprocess_batch(batch_x, device)
            batch_y = batch_y.to(device)

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
        losses.append(avg_loss)
        accs.append(avg_acc)
        print(f"epoch={epoch:02d} loss={avg_loss:.4f} acc={avg_acc:.4f}")

    return losses, accs


@torch.no_grad()
def infer(model: MoEClassifier, cfg: TrainConfig, device: torch.device, num_test_samples: int = 8) -> None:
    if num_test_samples <= 0:
        return

    model.eval()
    torch.seed()

    if cfg.dataset == "synthetic":
        sample_x = torch.randn(num_test_samples, cfg.input_dim).to(device)
        sample_y = None
    else:
        _, test_loader = build_mnist_loaders(cfg)
        batch_x_list: list[torch.Tensor] = []
        batch_y_list: list[torch.Tensor] = []
        collected = 0

        for batch_x, batch_y in test_loader:
            batch_x = preprocess_batch(batch_x, device)
            batch_y = batch_y.to(device)
            batch_x_list.append(batch_x)
            batch_y_list.append(batch_y)
            collected += batch_x.size(0)
            if collected >= num_test_samples:
                break

        sample_x = torch.cat(batch_x_list, dim=0)[:num_test_samples]
        sample_y = torch.cat(batch_y_list, dim=0)[:num_test_samples]

    logits, gate_probs = model(sample_x)
    pred = torch.argmax(logits, dim=1)

    print(f"inference predictions ({num_test_samples} samples):", pred.tolist())
    if sample_y is not None:
        inference_acc = (pred == sample_y).float().mean().item()
        print(f"inference accuracy ({num_test_samples} samples): {inference_acc:.4f}")
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
    parser.add_argument("--num-test-samples", type=int, default=8, help="Number of test samples for inference")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "mnist"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--history-csv", type=str, default="artifacts/train_history.csv")
    parser.add_argument("--history-plot", type=str, default="artifacts/train_history.png")
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
        dataset=args.dataset,
        data_dir=args.data_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    if cfg.dataset == "mnist":
        cfg.input_dim = 28 * 28
        cfg.output_dim = 10

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
        infer(model, cfg, device, num_test_samples=args.num_test_samples)
        return

    losses, accs = train(model, cfg, device)
    save_history_csv(losses, accs, args.history_csv)
    plot_history(losses, accs, args.history_plot)
    save_checkpoint(model, cfg, args.checkpoint_path)
    loaded_model = load_model_from_checkpoint(args.checkpoint_path, device)
    infer(loaded_model, cfg, device, num_test_samples=args.num_test_samples)


if __name__ == "__main__":
    main()
