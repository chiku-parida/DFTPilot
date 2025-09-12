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
import numpy as np
from pathlib import Path

from utills.text_encoder import lot_prompt, encode_lot, text_dim
from utills.graph_encoder import CrystalGraph
from modules.models import BandGapRegressor
from data.rag_index_build import RAG  # loads .meta.json and .soap.json


def _read_incar_text(text: str) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for ln in text.splitlines():
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        d[k.strip().upper()] = v.split("!")[0].strip()
    return d

def _infer_level_of_theory(incar: Dict[str, Any]) -> str:
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

def _rag_features(rag: Optional[RAG], struct: Structure, level: str, k: int = 8, device: Optional[torch.device] = None) -> torch.Tensor:
    if rag is None:
        return torch.zeros(4, dtype=torch.float32, device=device)
    try:
        nn = rag.query(struct, k=k, level_filter=level)
        if not nn:
            return torch.zeros(4, dtype=torch.float32, device=device)
        vals = np.asarray([float(x["bandgap_eV"]) for x in nn], dtype=float)
        vec = np.array([vals.mean(), vals.std() if vals.size > 1 else 0.0, vals.min(), vals.max()], dtype=np.float32)
        return torch.from_numpy(vec).to(device) if device else torch.from_numpy(vec)
    except Exception as e:
        print(f"[RAG WARN] using zeros: {e}")
        return torch.zeros(4, dtype=torch.float32, device=device)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="BandGap-LLM")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_PATH = "/home2/llmhackathon25/chkpt_test/bg_llm_best_final_n256.pt"     # adjust paths if needed
RAG_INDEX = "/home2/llmhackathon25/data/train_final_processed/rag.index"

ckpt = torch.load(CKPT_PATH, map_location=device)
genc = CrystalGraph().to(device); genc.load_state_dict(ckpt["genc"]); genc.eval()
model = BandGapRegressor(g_dim=256, t_dim=text_dim(), rag_dim=4).to(device)
model.load_state_dict(ckpt["model"]); model.eval()

rag: Optional[RAG] = None
try:
    rag = RAG(RAG_INDEX)
except Exception as e:
    print(f"[RAG WARN] failed to load RAG index '{RAG_INDEX}': {e}")
    rag = None


class PredictResp(BaseModel):
    target_level: str
    inferred_from_incar: bool
    bandgap_eV: float
    neighbors: list


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # include AUTO option
    levels = ["AUTO (infer from INCAR)", "PBE", "SCAN", "HSE06(hfscreen=0.2)", "HSE06(hfscreen=0.208)"]
    return templates.TemplateResponse("index.html", {"request": request, "levels": levels})

@app.post("/api/predict", response_model=PredictResp)
async def api_predict(
    structure: UploadFile = File(..., description="POSCAR or CIF"),
    target_level: str = Form(..., description="Pick a level or AUTO to infer from INCAR"),
    extras: str = Form("", description="Optional JSON dict or KEY=VAL lines"),
    incar: UploadFile | None = File(None, description="Optional INCAR file to infer target level"),
):
    # read structure
    raw_struct = (await structure.read()).decode(errors="ignore")
    try:
        s = Structure.from_str(raw_struct, fmt="poscar")
    except Exception:
        try:
            s = Structure.from_str(raw_struct, fmt="cif")
        except Exception as e:
            return JSONResponse({"detail": f"Failed to parse structure: {e}"}, status_code=400)

    # read/parse extras (optional)
    extras_dict: Dict[str, Any] = {}
    if extras.strip():
        try:
            extras_dict = json.loads(extras)
        except Exception:
            # support KEY=VAL lines
            ex: Dict[str, Any] = {}
            for ln in extras.splitlines():
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    ex[k.strip().upper()] = v.strip()
            extras_dict = ex

    # read/parse INCAR (optional)
    incar_dict: Dict[str, Any] = {}
    if incar is not None:
        try:
            incar_text = (await incar.read()).decode(errors="ignore")
            incar_dict = _read_incar_text(incar_text)
        except Exception as e:
            print(f"[WARN] could not parse INCAR: {e}")

    # decide level
    want_auto = target_level.strip().upper().startswith("AUTO")
    inferred = False
    level = target_level
    if want_auto:
        if incar_dict:
            level = _infer_level_of_theory(incar_dict)
            inferred = True
        else:
            # fallback if no INCAR provided
            level = "PBE"  # safe default
            inferred = False

    # embeddings & predict
    with torch.no_grad():
        g_emb = genc(s)
        # merge extras with incar when building prompt (INCOR not to override extras keys)
        prompt_kwargs = {**incar_dict, **extras_dict}
        t_emb = encode_lot(lot_prompt(level, prompt_kwargs)).to(device)
        r_feats = _rag_features(rag, s, level, k=8, device=device)
        y_pred, y_delta = model(g_emb, t_emb, r_feats, level)
        y = (y_pred + y_delta) if y_delta is not None else y_pred
        bandgap = float(y.item())

    neighbors: List[Dict[str, Any]] = []
    if rag is not None:
        try:
            neighbors = rag.query(s, k=8, level_filter=level)
        except Exception as e:
            print(f"[RAG WARN] neighbor query failed: {e}")

    return PredictResp(
        target_level=level,
        inferred_from_incar=inferred,
        bandgap_eV=bandgap,
        neighbors=neighbors,
    )

@app.post("/predict", response_class=HTMLResponse)
async def ui_predict(
    request: Request,
    structure: UploadFile = File(...),
    target_level: str = Form(...),
    extras: str = Form(""),
    incar: UploadFile | None = File(None),
):
    api_resp = await api_predict(structure=structure, target_level=target_level, extras=extras, incar=incar)
    if isinstance(api_resp, JSONResponse) and api_resp.status_code != 200:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "levels": ["AUTO (infer from INCAR)", "PBE", "SCAN", "HSE06(hfscreen=0.2)", "HSE06(hfscreen=0.208)"],
            "error": api_resp.body.decode()
        })
    data = api_resp.dict()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "levels": ["AUTO (infer from INCAR)", "PBE", "SCAN", "HSE06(hfscreen=0.2)", "HSE06(hfscreen=0.208)"],
        "prediction": data
    })

@app.get("/health")
def health():
    return {"ok": True, "device": str(device), "has_rag": rag is not None}
