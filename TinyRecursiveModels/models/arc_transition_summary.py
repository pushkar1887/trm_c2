from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ARCColourTransitionSummary(nn.Module):
    """Extract exact visible-demo colour transition evidence."""

    def __init__(
        self,
        color_token_ids: Sequence[int],
        pad_id: int,
        eos_id: int,
        vocab_size: int = 12,
    ) -> None:
        super().__init__()
        assert len(color_token_ids) == 10
        lookup = torch.full((vocab_size,), -1, dtype=torch.long)
        for colour_index, token_id in enumerate(color_token_ids):
            lookup[int(token_id)] = colour_index
        self.register_buffer("colour_lookup", lookup, persistent=False)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)

    def forward(
        self,
        context_inputs: torch.Tensor,
        context_outputs: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert context_inputs.shape == context_outputs.shape
        assert context_inputs.ndim == 3 and context_inputs.shape[-1] == 900
        b, m, _l = context_inputs.shape

        x_colour = self.colour_lookup[context_inputs.long()]
        y_colour = self.colour_lookup[context_outputs.long()]
        x_valid = x_colour >= 0
        y_valid = y_colour >= 0
        demo_valid = context_mask.bool().unsqueeze(-1)

        valid_to_valid = demo_valid & x_valid & y_valid
        changed_valid = valid_to_valid & (x_colour != y_colour)
        safe_x = x_colour.clamp_min(0)
        safe_y = y_colour.clamp_min(0)
        pair_id = safe_x * 10 + safe_y
        pair_onehot = F.one_hot(pair_id, num_classes=100).float()
        transition_hist = (pair_onehot * changed_valid.unsqueeze(-1).float()).sum(dim=2)
        transition_hist = transition_hist / (transition_hist.sum(dim=-1, keepdim=True) + 1.0)

        denominator = demo_valid.float().sum(dim=2).clamp_min(1.0)
        changed_valid_rate = changed_valid.float().sum(dim=2) / denominator
        valid_preserved_rate = ((valid_to_valid & (x_colour == y_colour)).float().sum(dim=2) / denominator)
        added_valid_rate = ((demo_valid & ~x_valid & y_valid).float().sum(dim=2) / denominator)
        removed_valid_rate = ((demo_valid & x_valid & ~y_valid).float().sum(dim=2) / denominator)
        structural = torch.stack(
            [changed_valid_rate, valid_preserved_rate, added_valid_rate, removed_valid_rate],
            dim=-1,
        )
        summary = torch.cat([transition_hist, structural], dim=-1)
        assert summary.shape == (b, m, 104)
        return summary
