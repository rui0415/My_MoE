from dataclasses import dataclass


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
    routing_plot: str = "artifacts/digit_to_expert_routing.png"
    balance_loss_weight: float = 0.01
