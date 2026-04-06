import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        if self.top_k < self.num_experts:
            topk_values, topk_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
            sparse_gate_probs = torch.zeros_like(gate_probs)
            sparse_gate_probs.scatter_(dim=-1, index=topk_indices, src=topk_values)
            gate_probs = sparse_gate_probs / sparse_gate_probs.sum(dim=-1, keepdim=True)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed_output = torch.sum(expert_outputs * gate_probs.unsqueeze(-1), dim=1)
        importance = gate_probs.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(importance * importance)
        return mixed_output, gate_probs, balance_loss
