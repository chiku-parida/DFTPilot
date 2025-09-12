# bandgap_calculator.py
from __future__ import annotations
import os, json
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from pymatgen.core import Structure

from utills.graph_encoder import CrystalGraph
from utills.text_encoder import lot_prompt, encode_lot, text_dim
from modules.models import BandGapRegressor

from data.rag_index_build import RAG



def _read_incar(src: Union[str, Dict[str, Any], None]) -> Dict[str, Any]:
    """Accepts a path to INCAR, raw INCAR text, a dict, or None; returns uppercased dict."""
    if src is None:
        return {}
    if isinstance(src, dict):
        return {str(k).upper(): v for k, v in src.items()}
    if isinstance(src, str):
        text = open(src, "r", errors="ignore").read() if os.path.isfile(src) else src
        d: Dict[str, Any] = {}
        for line in text.splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            d[k.strip().upper()] = v.split("!")[0].strip()
        return d
    raise TypeError("incar must be path, raw text, dict, or None")

def _infer_level_of_theory(incar: Dict[str, Any]) -> str:
    """Heuristic consistent with training."""
    lhf = str(incar.get("LHFCALC", "")).upper()
    if lhf in ("T", ".TRUE.", "TRUE", "1", "YES"):
        hs = incar.get("HFSCREEN", None)
        try:
            hs = float(str(hs).replace(",", " ").split()[0]) if hs is not None else 0.2
        except Exception:
            hs = 0.2
        return f"HSE06(hfscreen={hs})"
    if str(incar.get("METAGGA", "")).upper() in ("SCAN", "RSCAN"):
        return "SCAN"
    if str(incar.get("LDAU", "")).upper() in ("T", ".TRUE.", "TRUE", "1", "YES"):
        return f"PBE+U(U={incar.get('LDAUU','0')})"
    gga = str(incar.get("GGA", "PBE")).upper()
    return gga if gga and gga != "-" else "PBE"

def _load_structure(structure: Union[str, Structure, Dict[str, Any]]) -> Structure:
    """
    Accepts a path to POSCAR/CIF, raw POSCAR/CIF text, a pymatgen.Structure, or a Structure dict.
    """
    if isinstance(structure, Structure):
        return structure
    if isinstance(structure, dict):
        return Structure.from_dict(structure)
    if isinstance(structure, str):
        text = open(structure, "r").read() if os.path.exists(structure) else structure
        try:
            return Structure.from_str(text, fmt="poscar")
        except Exception:
            return Structure.from_str(text, fmt="cif")
    raise TypeError("structure must be path/text, Structure, or Structure-dict")

def _rag_features(rag: Optional[RAG], struct: Structure, target_level: str, k: int, device: torch.device) -> torch.Tensor:
    """
    Use RAG to get simple stats [mean, std, min, max] over neighbor band gaps.
    Returns zeros if RAG is disabled/unavailable or on any failure.
    """
    try:
        if rag is None:
            return torch.zeros(4, dtype=torch.float32, device=device)
        nn = rag.query(struct, k=k, level_filter=target_level)
        if not nn:
            return torch.zeros(4, dtype=torch.float32, device=device)
        vals = np.array([x["bandgap_eV"] for x in nn], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std()) if len(vals) > 1 else 0.0
        return torch.tensor([mean, std, float(vals.min()), float(vals.max())],
                            dtype=torch.float32, device=device)
    except Exception as e:
        # Fail-soft: do not crash prediction if RAG mismatches or any runtime issue occurs
        print(f"[RAG WARN] using zeros: {e}")
        return torch.zeros(4, dtype=torch.float32, device=device)



class BandGapCalculator:
    """
    Calculator that loads the trained model and (optionally) a RAG index.

    Example:
        calc = BandGapCalculator(
            checkpoint="/path/to/bg_llm.pt",
            rag_index="/path/to/rag.index",   # or None to disable RAG
            device="cuda"
        )

        result = calc.predict(
            structure="/path/to/POSCAR",   # or raw text / Structure / dict
            incar="/path/to/INCAR",        # or raw text / dict / None
            target_level=None,             # inferred from INCAR if None
            k_neighbors=5
        )
    """

    def __init__(
        self,
        checkpoint: str,
        rag_index: Optional[str] = None,
        device: Optional[str] = None,
        g_dim: int = 256,
        rag_dim: int = 4,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Models
        self.genc = CrystalGraph().to(self.device)
        self.model = BandGapRegressor(g_dim=g_dim, t_dim=text_dim(), rag_dim=rag_dim).to(self.device)

        # Load checkpoint
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.genc.load_state_dict(ckpt["genc"])
        self.model.load_state_dict(ckpt["model"])
        self.genc.eval()
        self.model.eval()

        # Optional RAG index (loads its own SOAP config and checks dimension)
        self.rag: Optional[RAG] = None
        if rag_index:
            try:
                self.rag = RAG(rag_index)
            except Exception as e:
                # Don't hard-fail the calculator; just warn and continue without RAG.
                print(f"[RAG WARN] Failed to load RAG index '{rag_index}': {e}")
                self.rag = None

    def predict(
        self,
        structure: Union[str, Structure, Dict[str, Any]],
        incar: Union[str, Dict[str, Any], None] = None,
        target_level: Optional[str] = None,
        k_neighbors: int = 5,
    ) -> Dict[str, Any]:
        # Inputs
        struct = _load_structure(structure)
        incar_dict = _read_incar(incar)
        level = target_level or _infer_level_of_theory(incar_dict)

        # Embeddings (ensure everything on the same device)
        with torch.no_grad():
            g_emb = self.genc(struct)  # on self.device (CrystalGraph creates tensors on its device)
            t_emb = encode_lot(lot_prompt(level, incar_dict)).to(self.device)
            r_feats = _rag_features(self.rag, struct, level, k=k_neighbors, device=self.device)

            # Forward
            y_pred, y_delta = self.model(g_emb, t_emb, r_feats, level)
            y_hat = y_pred + y_delta if y_delta is not None else y_pred
            bandgap = float(y_hat.item())

        # Nearest neighbors (optional)
        neighbors = []
        if self.rag is not None:
            try:
                neighbors = self.rag.query(struct, k=k_neighbors, level_filter=level)
            except Exception as e:
                print(f"[RAG WARN] neighbor query failed: {e}")
                neighbors = []

        return {
            "target_level": level,
            "predicted_bandgap_eV": bandgap,
            "neighbors": neighbors,  # [{similarity, bandgap_eV, level_of_theory, calc_dir, is_direct, incar}, ...]
        }







