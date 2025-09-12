# app.py
from __future__ import annotations
import json
from typing import Any, Dict, Optional, List

import torch
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pymatgen.core import Structure


from utills.text_encoder import lot_prompt, encode_lot, text_dim
from utills.graph_encoder import CrystalGraph
from modules.models import BandGapRegressor
from data.rag_index_build import RAG

# If you already have a helper rag_features that returns [mean,std,min,max]:
import numpy as np
def rag_features(rag: Optional[RAG], struct: Structure, target_level: str, k: int = 8, device: Optional[torch.device] = None) -> torch.Tensor:
    if rag is None:
        return torch.zeros(4, dtype=torch.float32, device=device)
    try:
        nn = rag.query(struct, k=k, level_filter=target_level)
        if not nn:
            return torch.zeros(4, dtype=torch.float32, device=device)
        vals = np.asarray([float(x["bandgap_eV"]) for x in nn], dtype=float)
        vec = np.array([vals.mean(), vals.std() if vals.size > 1 else 0.0, vals.min(), vals.max()], dtype=np.float32)
        return torch.from_numpy(vec).to(device) if device else torch.from_numpy(vec)
    except Exception as e:
        print(f"[RAG WARN] using zeros: {e}")
        return torch.zeros(4, dtype=torch.float32, device=device)


app = FastAPI(title="BandGap-LLM")

# static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load checkpoint ---
CKPT_PATH = "/home2/llmhackathon25/chkpt_test/bg_llm_last_final_n128.pt"     
RAG_INDEX = "/home2/llmhackathon25/data/train_final_processed/rag.index"  

ckpt = torch.load(CKPT_PATH, map_location=device)
genc = CrystalGraph().to(device); genc.load_state_dict(ckpt["genc"]); genc.eval()
model = BandGapRegressor(g_dim=256, t_dim=text_dim(), rag_dim=4).to(device)
model.load_state_dict(ckpt["model"]); model.eval()

rag: Optional[RAG] = None
try:
    rag = RAG(RAG_INDEX)  # loads .meta.json and .soap.json internally
except Exception as e:
    print(f"[RAG WARN] failed to load RAG index '{RAG_INDEX}': {e}")
    rag = None


class PredictResp(BaseModel):
    target_level: str
    bandgap_eV: float
    neighbors: list



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the browser form."""
    # some common target levels to pick from
    levels = ["PBE", "SCAN", "HSE06(hfscreen=0.2)", "HSE06(hfscreen=0.208)"]
    return templates.TemplateResponse("index.html", {"request": request, "levels": levels})

@app.post("/api/predict", response_model=PredictResp)
async def api_predict(
    structure: UploadFile = File(..., description="POSCAR or CIF"),
    target_level: str = Form(..., description="e.g. PBE / SCAN / HSE06(hfscreen=0.208)"),
    extras: str = Form("", description='Optional JSON dict of INCAR-like settings'),
):
    """JSON API: returns a prediction and neighbors."""
    raw = (await structure.read()).decode(errors="ignore")

    # parse structure
    try:
        s = Structure.from_str(raw, fmt="poscar")
    except Exception:
        try:
            s = Structure.from_str(raw, fmt="cif")
        except Exception as e:
            return JSONResponse({"detail": f"Failed to parse structure: {e}"}, status_code=400)

    # parse extras (optional)
    extras_dict: Dict[str, Any] = {}
    if extras.strip():
        try:
            extras_dict = json.loads(extras)
        except Exception:
            # also support simple "KEY=VAL" lines
            try:
                ex: Dict[str, Any] = {}
                for ln in extras.splitlines():
                    if "=" in ln:
                        k, v = ln.split("=", 1)
                        ex[k.strip().upper()] = v.strip()
                extras_dict = ex
            except Exception:
                extras_dict = {}

    # embeddings
    with torch.no_grad():
        g_emb = genc(s)  # already on device
        t_emb = encode_lot(lot_prompt(target_level, extras_dict)).to(device)
        r_feats = rag_features(rag, s, target_level, k=8, device=device)
        y_pred, y_delta = model(g_emb, t_emb, r_feats, target_level)
        y = (y_pred + y_delta) if y_delta is not None else y_pred
        bandgap = float(y.item())

    neighbors: List[Dict[str, Any]] = []
    if rag is not None:
        try:
            neighbors = rag.query(s, k=8, level_filter=target_level)
        except Exception as e:
            print(f"[RAG WARN] neighbor query failed: {e}")
            neighbors = []

    return PredictResp(target_level=target_level, bandgap_eV=bandgap, neighbors=neighbors)

@app.post("/predict", response_class=HTMLResponse)
async def ui_predict(
    request: Request,
    structure: UploadFile = File(...),
    target_level: str = Form(...),
    extras: str = Form(""),
):
    """Handle form submit and render a result page."""
    api_resp = await api_predict(structure=structure, target_level=target_level, extras=extras)
    if isinstance(api_resp, JSONResponse) and api_resp.status_code != 200:
        # render error
        return templates.TemplateResponse("index.html", {
            "request": request,
            "levels": ["PBE", "SCAN", "HSE06(hfscreen=0.2)", "HSE06(hfscreen=0.208)"],
            "error": api_resp.body.decode()
        })
    data = api_resp.dict()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "levels": ["PBE", "SCAN", "HSE06(hfscreen=0.2)", "HSE06(hfscreen=0.208)"],
        "prediction": data
    })

@app.get("/health")
def health():
    return {"ok": True, "device": str(device), "has_rag": rag is not None}
