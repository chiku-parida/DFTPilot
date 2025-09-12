from __future__ import annotations
import os, json, random, math, argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pymatgen.core import Structure

from data.rag_index_build import RAG                         # <— new RAG that loads .soap.json internally
from utills.text_encoder import lot_prompt, encode_lot, text_dim
from utills.graph_encoder import CrystalGraph
from modules.models import BandGapRegressor
from collections import defaultdict


'''
class BGDataset(Dataset):
    """Groups rows by a per-structure key so different theory levels of the same structure can be sampled."""
    def __init__(self, jsonl: str):
        with open(jsonl, "r") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.by_struct = defaultdict(list)
        for r in self.rows:
            # robust key: (parent folder name) or fallback to calc_dir
            calc_dir = r.get("calc_dir", "")
            key = os.path.basename(calc_dir).split("_")[0] or calc_dir
            self.by_struct[key].append(r)
        self.keys = list(self.by_struct.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        group = self.by_struct[self.keys[idx]]
        r = random.choice(group)  # pick one level-of-theory sample
        return {
            "structure": r["structure"],               # dict (pymatgen format)
            "tgt_level": r["level_of_theory"],         # str
            "extras": r.get("incar", {}),              # dict
            "y": float(r["bandgap_eV"]),               # float
            "group": group,                            # list[dict]
        }


def custom_collate(batch: List[Dict[str, Any]]):
    structs = [Structure.from_dict(b["structure"]) for b in batch]
    tgt_levels = [b["tgt_level"] for b in batch]
    extras = [b["extras"] for b in batch]
    ys = torch.tensor([b["y"] for b in batch], dtype=torch.float32)
    groups = [b["group"] for b in batch]
    return structs, tgt_levels, extras, ys, groups



@torch.no_grad()
def rag_features(rag: RAG, struct: Structure, target_level: str, k: int = 8, device: torch.device | None = None) -> torch.Tensor:
    """Return [mean, std, min, max] of neighbor band gaps, or zeros if none/failure."""
    try:
        nn = rag.query(struct, k=k, level_filter=target_level)
        if not nn:
            return torch.zeros(4, dtype=torch.float32, device=device)
        vals = np.asarray([x["bandgap_eV"] for x in nn], dtype=float)
        vec = np.array(
            [vals.mean(), vals.std() if vals.size > 1 else 0.0, vals.min(), vals.max()],
            dtype=np.float32,
        )
        return torch.from_numpy(vec).to(device) if device else torch.from_numpy(vec)
    except Exception as e:
        print(f"[RAG WARN] using zeros (query failed): {e}")
        return torch.zeros(4, dtype=torch.float32, device=device)



def split_indices(n: int, val_frac: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:] if n_val < n else idx[:]
    return train_idx, val_idx


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)



def train(
    jsonl: str,
    index_path: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    patience: int,
    val_frac: float,
    k_neighbors: int,
    ckpt_best_path: str,
    ckpt_last_path: str,
    seed: int,
):
 
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    dev = torch.device(device if torch.cuda.is_available() else device == "cpu")

    # Data
    ds = BGDataset(jsonl)
    train_idx, val_idx = split_indices(len(ds), val_frac=val_frac, seed=seed)
    dl_train = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate)
    dl_val   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate) if len(val_idx)>0 else None

    # Models
    genc = CrystalGraph().to(dev)
    model = BandGapRegressor(g_dim=128, t_dim=text_dim(), rag_dim=4).to(dev)  # rag_dim=4 for [mean,std,min,max]
    opt = torch.optim.AdamW(list(genc.parameters()) + list(model.parameters()), lr=lr)
    grad_clip = 1.0

    rag = None
    if index_path and os.path.exists(index_path):
        try:
            rag = RAG(index_path)
        except Exception as e:
            print(f"[RAG WARN] failed to load index '{index_path}': {e}. Continuing without RAG.")
            rag = None

    best_val = math.inf
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        genc.train(); model.train()
        train_losses: List[float] = []

        for structs, tgt_levels, extras, ys, _ in dl_train:
            B = len(structs)
            # Graph embeddings
            g_embs = torch.stack([genc(s) for s in structs], dim=0)

            # Text embeddings
            t_embs = torch.stack([encode_lot(lot_prompt(l, e)).to(dev) for l, e in zip(tgt_levels, extras)], dim=0)

            # RAG features
            rag_feats = torch.stack([
                rag_features(rag, s, l, k=k_neighbors, device=dev) if rag is not None
                else torch.zeros(4, dtype=torch.float32, device=dev)
            for s, l in zip(structs, tgt_levels)], dim=0)

            # Forward (per-sample because model expects a level string)
            y_preds = []
            for i in range(B):
                y_pred, y_delta = model(g_embs[i], t_embs[i], rag_feats[i], tgt_levels[i])
                y_hat = y_pred + y_delta if y_delta is not None else y_pred
                y_preds.append(y_hat)
            y_preds = torch.stack(y_preds, dim=0).squeeze(-1)  # [B]

            loss = mse_loss(y_preds, ys.to(dev))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(genc.parameters()) + list(model.parameters()), grad_clip)
            opt.step()

            train_losses.append(float(loss.item()))

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")

        val_mse = float("nan")
        if dl_val is not None:
            genc.eval(); model.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for structs, tgt_levels, extras, ys, _ in dl_val:
                    B = len(structs)
                    g_embs = torch.stack([genc(s) for s in structs], dim=0)
                    t_embs = torch.stack([encode_lot(lot_prompt(l, e)).to(dev) for l, e in zip(tgt_levels, extras)], dim=0)
                    rag_feats = torch.stack([
                        rag_features(rag, s, l, k=k_neighbors, device=dev) if rag is not None
                        else torch.zeros(4, dtype=torch.float32, device=dev)
                    for s, l in zip(structs, tgt_levels)], dim=0)

                    y_preds = []
                    for i in range(B):
                        yp, yd = model(g_embs[i], t_embs[i], rag_feats[i], tgt_levels[i])
                        y_preds.append(yp + yd if yd is not None else yp)
                    y_preds = torch.stack(y_preds, dim=0).squeeze(-1)

                    val_losses.append(float(mse_loss(y_preds, ys.to(dev)).item()))
            val_mse = float(np.mean(val_losses)) if val_losses else float("nan")

        print(f"Epoch {ep:03d} | train MSE={train_mse:.6f}"
              + (f" | val MSE={val_mse:.6f}" if dl_val is not None else ""))

        improved = (val_mse < best_val) if dl_val is not None else (train_mse < best_val)
        metric_to_track = val_mse if dl_val is not None else train_mse
        if improved and not math.isnan(metric_to_track):
            best_val = metric_to_track
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(ckpt_best_path), exist_ok=True)
            torch.save({"genc": genc.state_dict(), "model": model.state_dict()}, ckpt_best_path)
            print(f"[checkpoint] saved BEST -> {ckpt_best_path} (metric={best_val:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[early stopping] no improvement for {patience} epochs. Stopping at epoch {ep}.")
                break

    # Always save the last state too
    os.makedirs(os.path.dirname(ckpt_last_path), exist_ok=True)
    torch.save({"genc": genc.state_dict(), "model": model.state_dict()}, ckpt_last_path)
    print(f"[checkpoint] saved LAST -> {ckpt_last_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BandGap LLM model with RAG.")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to processed JSONL dataset")
    parser.add_argument("--index_path", type=str, required=True, help="Path to RAG index file")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for training")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Validation set fraction")
    parser.add_argument("--k_neighbors", type=int, default=8, help="Number of neighbors for RAG features")
    parser.add_argument("--ckpt_best_path", type=str, default="./checkpoints/bg_llm_best.pt", help="Path to save best checkpoint")
    parser.add_argument("--ckpt_last_path", type=str, default="./checkpoints/bg_llm_last.pt", help="Path to save last checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    train(
        jsonl=args.jsonl,
        index_path=args.index_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        patience=args.patience,
        val_frac=args.val_frac,
        k_neighbors=args.k_neighbors,
        ckpt_best_path=args.ckpt_best_path,
        ckpt_last_path=args.ckpt_last_path,
        seed=args.seed,
    )

'''



class BGDataset(Dataset):
    def __init__(self, jsonl: str):
        with open(jsonl, "r") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.by_struct = defaultdict(list)
        for r in self.rows:
            calc_dir = r.get("calc_dir", "")
            key = os.path.basename(calc_dir).split("_")[0] or calc_dir
            self.by_struct[key].append(r)
        self.keys = list(self.by_struct.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        group = self.by_struct[self.keys[idx]]
        r = random.choice(group)
        return {
            "structure": r["structure"],
            "tgt_level": r["level_of_theory"],
            "extras": r.get("incar", {}),
            "y": float(r["bandgap_eV"]),
            "group": group,
        }


def custom_collate(batch: List[Dict[str, Any]]):
    structs = [Structure.from_dict(b["structure"]) for b in batch]
    tgt_levels = [b["tgt_level"] for b in batch]
    extras = [b["extras"] for b in batch]
    ys = torch.tensor([b["y"] for b in batch], dtype=torch.float32)
    groups = [b["group"] for b in batch]
    return structs, tgt_levels, extras, ys, groups


@torch.no_grad()
def rag_features(rag: RAG, struct: Structure, target_level: str, k: int = 8, device: torch.device | None = None) -> torch.Tensor:
    try:
        nn = rag.query(struct, k=k, level_filter=target_level)
        if not nn:
            return torch.zeros(4, dtype=torch.float32, device=device)
        vals = np.asarray([x["bandgap_eV"] for x in nn], dtype=float)
        vec = np.array([vals.mean(), vals.std() if vals.size > 1 else 0.0, vals.min(), vals.max()], dtype=np.float32)
        return torch.from_numpy(vec).to(device) if device else torch.from_numpy(vec)
    except Exception as e:
        print(f"[RAG WARN] using zeros (query failed): {e}")
        return torch.zeros(4, dtype=torch.float32, device=device)


def split_indices(n: int, val_frac: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:] if n_val < n else idx[:]
    return train_idx, val_idx


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)
def mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred - y_true))

def _safe_dump_json(obj: dict | list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def train(
    jsonl: str,
    index_path: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    patience: int,
    val_frac: float,
    k_neighbors: int,
    ckpt_best_path: str,
    ckpt_last_path: str,
    seed: int,
    log_json_path: str,
):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Data
    ds = BGDataset(jsonl)
    train_idx, val_idx = split_indices(len(ds), val_frac=val_frac, seed=seed)
    dl_train = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate)
    dl_val   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate) if len(val_idx)>0 else None

    # Models
    genc = CrystalGraph().to(dev)
    model = BandGapRegressor(g_dim=256, t_dim=text_dim(), rag_dim=4).to(dev)
    opt = torch.optim.AdamW(list(genc.parameters()) + list(model.parameters()), lr=lr)
    grad_clip = 1.0

    # RAG
    rag = None
    if index_path and os.path.exists(index_path):
        try:
            rag = RAG(index_path)
        except Exception as e:
            print(f"[RAG WARN] failed to load index '{index_path}': {e}. Continuing without RAG.")
            rag = None

    best_val = math.inf
    epochs_no_improve = 0

    # history we will dump to JSON
    history: List[Dict[str, float]] = []

    for ep in range(1, epochs + 1):
        genc.train(); model.train()
        #train_losses: List[float] = []
        train_losses_mse, train_losses_mae = [], []

        for structs, tgt_levels, extras, ys, _ in dl_train:
            B = len(structs)
            g_embs = torch.stack([genc(s) for s in structs], dim=0)
            t_embs = torch.stack([encode_lot(lot_prompt(l, e)).to(dev) for l, e in zip(tgt_levels, extras)], dim=0)
            rag_feats = torch.stack([
                rag_features(rag, s, l, k=k_neighbors, device=dev) if rag is not None
                else torch.zeros(4, dtype=torch.float32, device=dev)
            for s, l in zip(structs, tgt_levels)], dim=0)

            y_preds = []
            for i in range(B):
                y_pred, y_delta = model(g_embs[i], t_embs[i], rag_feats[i], tgt_levels[i])
                y_hat = y_pred + y_delta if y_delta is not None else y_pred
                y_preds.append(y_hat)
            y_preds = torch.stack(y_preds, dim=0).squeeze(-1)

            mse = mse_loss(y_preds, ys.to(dev))
            mae = mae_loss(y_preds, ys.to(dev))
            #loss = mse_loss(y_preds, ys.to(dev))
            opt.zero_grad(set_to_none=True)
            mse.backward()
            torch.nn.utils.clip_grad_norm_(list(genc.parameters()) + list(model.parameters()), grad_clip)
            opt.step()

            train_losses_mse.append(float(mse.item()))
            train_losses_mae.append(float(mae.item()))

        train_mse = float(np.mean(train_losses_mse)) if train_losses_mse else float("nan")
        train_mae = float(np.mean(train_losses_mae)) if train_losses_mae else float("nan")

    
        val_mse = float("nan")
        val_mae = float("nan")
        if dl_val is not None:
            genc.eval(); model.eval()
            #val_losses: List[float] = []
            val_losses_mse, val_losses_mae = [], []
            with torch.no_grad():
                for structs, tgt_levels, extras, ys, _ in dl_val:
                    B = len(structs)
                    g_embs = torch.stack([genc(s) for s in structs], dim=0)
                    t_embs = torch.stack([encode_lot(lot_prompt(l, e)).to(dev) for l, e in zip(tgt_levels, extras)], dim=0)
                    rag_feats = torch.stack([
                        rag_features(rag, s, l, k=k_neighbors, device=dev) if rag is not None
                        else torch.zeros(4, dtype=torch.float32, device=dev)
                    for s, l in zip(structs, tgt_levels)], dim=0)

                    y_preds = []
                    for i in range(B):
                        yp, yd = model(g_embs[i], t_embs[i], rag_feats[i], tgt_levels[i])
                        y_preds.append(yp + yd if yd is not None else yp)
                    y_preds = torch.stack(y_preds, dim=0).squeeze(-1)

                    val_losses_mse.append(float(mse_loss(y_preds, ys.to(dev)).item()))
                    val_losses_mae.append(float(mae_loss(y_preds, ys.to(dev)).item()))
                    #val_losses.append(float(mse_loss(y_preds, ys.to(dev)).item()))
            val_mse = float(np.mean(val_losses_mse)) if val_losses_mse else float("nan")
            val_mae = float(np.mean(val_losses_mae)) if val_losses_mae else float("nan")

        print(f"Epoch {ep:03d} | train MSE={train_mse:.6f} | train MAE={train_mae:.6f}"
              + (f" | val MSE={val_mse:.6f} | val MAE={val_mae:.6f}" if dl_val is not None else ""))

        # log this epoch
        history.append({
            "epoch": ep,
            "train_mse": train_mse,
            "train_mae": train_mae,
            **({"val_mse": val_mse, "val_mae": val_mae} if dl_val is not None else {})
        })
        _safe_dump_json({
            "history": history,
            "has_validation": dl_val is not None,
            "best_metric_so_far": (min([h.get("val_mse", h["train_mse"]) for h in history]) if history else None)
        }, log_json_path)


        improved = (val_mse < best_val) if dl_val is not None else (train_mse < best_val)
        metric_to_track = val_mse if dl_val is not None else train_mse
        if improved and not math.isnan(metric_to_track):
            best_val = metric_to_track
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(ckpt_best_path), exist_ok=True)
            torch.save({"genc": genc.state_dict(), "model": model.state_dict()}, ckpt_best_path)
            print(f"[checkpoint] saved BEST -> {ckpt_best_path} (metric={best_val:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[early stopping] no improvement for {patience} epochs. Stopping at epoch {ep}.")
                break

    # save last
    os.makedirs(os.path.dirname(ckpt_last_path), exist_ok=True)
    torch.save({"genc": genc.state_dict(), "model": model.state_dict()}, ckpt_last_path)
    print(f"[checkpoint] saved LAST -> {ckpt_last_path}")

    # finalize log
    summary = {
        "history": history,
        "has_validation": dl_val is not None,
        "stopped_epoch": history[-1]["epoch"] if history else None,
        "best_metric": min([h.get("val_mse", h["train_mse"]) for h in history]) if history else None,
        "best_checkpoint": ckpt_best_path,
        "last_checkpoint": ckpt_last_path,
    }
    _safe_dump_json(summary, log_json_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BandGap LLM model with RAG.")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to processed JSONL dataset")
    parser.add_argument("--index_path", type=str, required=True, help="Path to RAG index file")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--k_neighbors", type=int, default=8)
    parser.add_argument("--ckpt_best_path", type=str, default="./checkpoints/bg_llm_best.pt")
    parser.add_argument("--ckpt_last_path", type=str, default="./checkpoints/bg_llm_last.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_json_path", type=str, default="./checkpoints/train_log.json",
                        help="Path to dump JSON log of per-epoch metrics")
    args = parser.parse_args()

    train(
        jsonl=args.jsonl,
        index_path=args.index_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        patience=args.patience,
        val_frac=args.val_frac,
        k_neighbors=args.k_neighbors,
        ckpt_best_path=args.ckpt_best_path,
        ckpt_last_path=args.ckpt_last_path,
        seed=args.seed,
        log_json_path=args.log_json_path,
    )
