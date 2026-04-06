import csv
import os

import torch


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


def save_digit_routing_heatmap(
    routing_counts: torch.Tensor,
    path: str,
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    counts = routing_counts.detach().cpu().float()
    row_sums = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    routing_ratios = counts / row_sums

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(routing_ratios.numpy(), cmap="viridis", aspect="auto")

    ax.set_xlabel("expert")
    ax.set_ylabel("digit")
    ax.set_xticks(range(counts.size(1)))
    ax.set_yticks(range(counts.size(0)))
    ax.set_yticklabels([str(i) for i in range(counts.size(0))])
    ax.set_title("Digit to Expert Routing Ratio")
    fig.colorbar(image, ax=ax, label="routing ratio")

    for digit in range(counts.size(0)):
        for expert in range(counts.size(1)):
            value = routing_ratios[digit, expert].item()
            ax.text(
                expert,
                digit,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("routing heatmap saved:", path)
