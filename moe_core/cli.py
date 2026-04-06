import argparse

import torch

from .checkpoint import load_model_from_checkpoint, save_checkpoint
from .config import TrainConfig
from .infer import infer
from .model import MoEClassifier
from .train import train
from .visualization import plot_history, save_history_csv


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
    parser.add_argument("--routing-plot", type=str, default="artifacts/digit_to_expert_routing.png")
    parser.add_argument("--balance-loss-weight", type=float, default=0.01)
    return parser.parse_args()


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
        routing_plot=args.routing_plot,
        balance_loss_weight=args.balance_loss_weight,
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
