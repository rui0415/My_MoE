from torch.utils.data import DataLoader, Subset, TensorDataset
import torch
from torchvision import datasets, transforms

from .config import TrainConfig


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


def build_cifar10_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_ds = datasets.CIFAR10(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.CIFAR10(
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


def build_classification_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    if cfg.dataset == "mnist":
        return build_mnist_loaders(cfg)
    if cfg.dataset == "cifar10":
        return build_cifar10_loaders(cfg)
    raise ValueError(f"Unsupported dataset for classification loaders: {cfg.dataset}")


def build_train_loader(cfg: TrainConfig, device: torch.device) -> DataLoader:
    if cfg.dataset == "synthetic":
        dataset = build_synthetic_dataset(cfg, device)
        return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    train_loader, _ = build_classification_loaders(cfg)
    return train_loader


def preprocess_batch(batch_x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if batch_x.dim() == 4:
        batch_x = batch_x.flatten(start_dim=1)
    return batch_x.to(device)
