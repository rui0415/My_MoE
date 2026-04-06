import torch

from .config import TrainConfig
from .data import build_mnist_loaders, preprocess_batch
from .model import MoEClassifier
from .visualization import save_digit_routing_heatmap


def collect_mnist_routing_stats(
    model: MoEClassifier,
    cfg: TrainConfig,
    device: torch.device,
    num_test_samples: int,
) -> torch.Tensor:
    _, test_loader = build_mnist_loaders(cfg)
    routing_counts = torch.zeros(cfg.output_dim, model.num_experts, device=device)
    collected = 0

    for batch_x, batch_y in test_loader:
        batch_x = preprocess_batch(batch_x, device)
        batch_y = batch_y.to(device)

        remaining = num_test_samples - collected
        if remaining <= 0:
            break

        if batch_x.size(0) > remaining:
            batch_x = batch_x[:remaining]
            batch_y = batch_y[:remaining]

        _, gate_probs, _ = model(batch_x)
        expert_ids = torch.argmax(gate_probs, dim=1)

        for digit, expert_id in zip(batch_y.tolist(), expert_ids.tolist()):
            routing_counts[digit, expert_id] += 1

        collected += batch_x.size(0)

    return routing_counts


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

    logits, gate_probs, _ = model(sample_x)
    pred = torch.argmax(logits, dim=1)

    print(f"inference predictions ({num_test_samples} samples):", pred.tolist())
    if sample_y is not None:
        inference_acc = (pred == sample_y).float().mean().item()
        print(f"inference accuracy ({num_test_samples} samples): {inference_acc:.4f}")
        routing_counts = collect_mnist_routing_stats(model, cfg, device, num_test_samples)
        save_digit_routing_heatmap(routing_counts, cfg.routing_plot)
    print("gate probs for first sample:", gate_probs[0].tolist())
