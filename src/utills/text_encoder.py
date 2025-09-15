
from __future__ import annotations
from typing import Dict
from sentence_transformers import SentenceTransformer
import numpy as np, torch
hf_token="your-hf-token"
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=hf_token)  # small but strong
_DIM = _MODEL.get_sentence_embedding_dimension()

def lot_prompt(target_level:str, extras:Dict[str,str|float|int]|None=None)->str:
    pieces=[f"Predict band gap at level: {target_level}."]
    if extras:
        kv=", ".join(f"{k}={v}" for k,v in extras.items())
        pieces.append(f"Calculation settings: {kv}.")
    return " ".join(pieces)

def encode_lot(text:str)->torch.Tensor:
    v = _MODEL.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return torch.tensor(v, dtype=torch.float32)

def text_dim()->int: return _DIM
