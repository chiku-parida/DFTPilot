# src/model.py
from __future__ import annotations
import torch, torch.nn as nn
from typing import Optional

class BandGapRegressor(nn.Module):
    def __init__(self, g_dim=128, t_dim=384, rag_dim=4, hidden=256, levels=("PBE","PBE+U","SCAN","HSE06","GW")):
        super().__init__()
        self.levels = list(levels)
        self.fuse = nn.Sequential(
            nn.Linear(g_dim + t_dim + rag_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU()
        )
        # direct prediction head
        self.head = nn.Linear(hidden, 1)
        # optional Δ heads (predict correction from PBE to each target level)
        self.base = "PBE"
        self.delta_heads = nn.ModuleDict({lvl: nn.Linear(hidden,1) for lvl in self.levels if lvl!=self.base})

    def forward(self, g_emb, t_emb, rag_feats, target_level:str):
        z = self.fuse(torch.cat([g_emb, t_emb, rag_feats], dim=-1))
        y_direct = self.head(z).squeeze(-1)
        if target_level!=self.base and target_level in self.delta_heads:
            delta = self.delta_heads[target_level](z).squeeze(-1)
            return y_direct, delta
        return y_direct, None
    



    
