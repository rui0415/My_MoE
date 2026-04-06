import torch
import torch.nn as nn

from .config import TrainConfig
from .data import build_train_loader, preprocess_batch
from .model import MoEClassifier


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
            logits, _, balance_loss = model(batch_x)
            task_loss = criterion(logits, batch_y)
            loss = task_loss + cfg.balance_loss_weight * balance_loss
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
