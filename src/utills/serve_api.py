# src/serve.py
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import torch
from pymatgen.core import Structure
from text_encoder import lot_prompt, encode_lot
from graph_encoder import CrystalGraph
from model import BandGapRegressor
from rag_index import RAG, rag_features

app = FastAPI(title="BandGap-LLM")
device="cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load("checkpoints/bg_llm.pt", map_location=device)
genc = CrystalGraph().to(device); genc.load_state_dict(ckpt["genc"]); genc.eval()
model = BandGapRegressor().to(device); model.load_state_dict(ckpt["model"]); model.eval()
rag = RAG("data/processed/rag.index")

class PredictResp(BaseModel):
    target_level: str
    bandgap_eV: float
    neighbors: list

@app.post("/predict", response_model=PredictResp)
async def predict(structure: UploadFile = File(...),
                  target_level: str = Form(...),
                  extras: str = Form(default="")):
    raw = (await structure.read()).decode()
    try:
        s = Structure.from_str(raw, fmt="poscar")
    except Exception:
        s = Structure.from_str(raw, fmt="cif")

    g_emb = genc(s).to(device)
    prompt = lot_prompt(target_level, {})  # parse `extras` into dict if sent
    t_emb = encode_lot(prompt).to(device)
    r_feats = rag_features(rag, s, target_level).to(device)

    with torch.no_grad():
        y_pred, y_delta = model(g_emb, t_emb, r_feats, target_level)
        y = (y_pred + y_delta) if y_delta is not None else y_pred
    neigh = rag.query(s, k=8, level_filter=target_level)
    return PredictResp(target_level=target_level, bandgap_eV=float(y.item()), neighbors=neigh)
