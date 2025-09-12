# CGNN-based graph encoder for crystal structures
from __future__ import annotations
import torch, torch.nn as nn
from typing import Dict, Any
from pymatgen.core import Structure
import numpy as np

'''
class CrystalGraph(nn.Module):
    def __init__(self, node_dim=64, edge_dim=64, r_cut=5.0, n_rbf=32):
        super().__init__()
        self.emb = nn.Embedding(101, node_dim)
        self.edge_mlp = nn.Sequential(nn.Linear(n_rbf, edge_dim), nn.SiLU(), nn.Linear(edge_dim, edge_dim), nn.SiLU())
        self.convs = nn.ModuleList([nn.Sequential(nn.Linear(node_dim+edge_dim, node_dim), nn.SiLU()) for _ in range(3)])
        self.readout = nn.Sequential(nn.Linear(node_dim, 128), nn.SiLU())
        self.r_cut = r_cut
        self.register_buffer("rbf_centers", torch.linspace(0.0, r_cut, n_rbf))

    def rbf(self, d):
        # Gaussian RBF
        return torch.exp(-(d.unsqueeze(-1)-self.rbf_centers)**2 * 5.0)

    def forward(self, struct:Structure)->torch.Tensor:
        xyz = torch.tensor(struct.cart_coords, dtype=torch.float32)
        Z = torch.tensor([sp.number for sp in struct.species], dtype=torch.long).clamp(max=100)
        N = xyz.shape[0]
        # naive neighbor loop (fine for small N)
        send, recv, dist = [], [], []
        for i in range(N):
            for j in range(N):
                if i==j: continue
                v = struct.cart_coords[j]-struct.cart_coords[i]
                d = np.linalg.norm(v)
                if d<= self.r_cut:
                    send.append(i); recv.append(j); dist.append(d)
        if len(send)==0:  # degenerate
            send=[0]; recv=[0]; dist=[0.0]
        send, recv = torch.tensor(send), torch.tensor(recv)
        dist = torch.tensor(dist, dtype=torch.float32)
        x = self.emb(Z)  # [N,D]
        e = self.edge_mlp(self.rbf(dist))  # [E,edge_dim]

        for conv in self.convs:
            m = torch.zeros_like(x)
            m.index_add_(0, recv, e)  # simple edge pooling
            x = conv(torch.cat([x, m], dim=-1))
        h = self.readout(x)  # [N,128]
        g = h.mean(dim=0)    # graph embedding
        return g  # [128]
'''
class CrystalGraph(nn.Module):
    def __init__(self, node_dim=256, edge_dim=256, r_cut=5.0, n_rbf=32):
        super().__init__()
        self.emb = nn.Embedding(101, node_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_rbf, edge_dim), nn.SiLU(),
            nn.Linear(edge_dim, edge_dim), nn.SiLU()
        )
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Linear(node_dim + edge_dim, node_dim), nn.SiLU())
            for _ in range(3)
        ])
        self.readout = nn.Sequential(nn.Linear(node_dim, 256), nn.SiLU())
        self.r_cut = r_cut
        self.register_buffer("rbf_centers", torch.linspace(0.0, r_cut, n_rbf))

    def rbf(self, d):
        return torch.exp(-(d.unsqueeze(-1) - self.rbf_centers)**2 * 5.0)

    def forward(self, struct: Structure) -> torch.Tensor:
        # >>> ensure all tensors are created on the same device as the module <<<
        dev = self.rbf_centers.device  # same as self.emb.weight.device once model.to(device) is called

        xyz = torch.tensor(struct.cart_coords, dtype=torch.float32, device=dev)
        Z = torch.tensor([sp.number for sp in struct.species], dtype=torch.long, device=dev).clamp(max=100)
        N = xyz.shape[0]

        send, recv, dist = [], [], []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                v = struct.cart_coords[j] - struct.cart_coords[i]
                d = np.linalg.norm(v)
                if d <= self.r_cut:
                    send.append(i); recv.append(j); dist.append(d)
        if len(send) == 0:
            send = [0]; recv = [0]; dist = [0.0]

        send = torch.tensor(send, dtype=torch.long, device=dev)
        recv = torch.tensor(recv, dtype=torch.long, device=dev)
        dist = torch.tensor(dist, dtype=torch.float32, device=dev)

        x = self.emb(Z)  # [N, D]  (Z and emb weights are on the same device now)
        e = self.edge_mlp(self.rbf(dist))  # [E, edge_dim]

        for conv in self.convs:
            m = torch.zeros_like(x, device=dev)
            m.index_add_(0, recv, e)  # accumulate edge features to receivers
            x = conv(torch.cat([x, m], dim=-1))
        h = self.readout(x)   # [N, 128]
        g = h.mean(dim=0)     # [128]
        return g
