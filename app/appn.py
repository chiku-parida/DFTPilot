# app.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pymatgen.core import Structure

# --- your project imports ---
from utills.text_encoder import lot_prompt, encode_lot, text_dim
from utills.graph_encoder import CrystalGraph
from modules.models import BandGapRegressor
from data.rag_index_build import RAG
# reuse calculator helpers to read INCAR and infer level
from calculators.calculator import _infer_level_of_theory, _read_incar


# ---------- helpers ----------
def serialize_structure(s: Structure) -> dict:
    return s.as_dict()

def rag_features(
    rag: Optional[RAG],
    struct: Structure,
    target_level: str,
    k: int = 8,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return [mean, std, min, max] of neighbor bandgaps or zeros if none/failure."""
    if rag is None:
        return torch.zeros(4, dtype=torch.float32, device=device)
    try:
        nn = rag.query(struct, k=k, level_filter=target_level)
        if not nn:
            return torch.zeros(4, dtype=torch.float32, device=device)
        vals = np.asarray([float(x["bandgap_eV"]) for x in nn], dtype=float)
        vec = np.array(
            [vals.mean(), vals.std() if vals.size > 1 else 0.0, vals.min(), vals.max()],
            dtype=np.float32,
        )
        return torch.from_numpy(vec).to(device) if device else torch.from_numpy(vec)
    except Exception as e:
        print(f"[RAG WARN] using zeros: {e}")
        return torch.zeros(4, dtype=torch.float32, device=device)


# ---------- FastAPI setup ----------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="BandGap-LLM")

# only mount static if folder exists (prevents earlier crash)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- load model / RAG ----------
# TODO: set your actual checkpoint + index paths
CKPT_PATH = "/home2/llmhackathon25/chkpt_test/bg_llm_best_final_n256.pt"     # adjust paths if needed
RAG_INDEX = "/home2/llmhackathon25/data/train_final_processed/rag.index"

ckpt = torch.load(CKPT_PATH, map_location=device)
genc = CrystalGraph().to(device)
genc.load_state_dict(ckpt["genc"])
genc.eval()

# Ensure g_dim matches training (commonly 128)
model = BandGapRegressor(g_dim=256, t_dim=text_dim(), rag_dim=4).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

rag: Optional[RAG] = None
try:
    rag = RAG(RAG_INDEX)
except Exception as e:
    print(f"[RAG WARN] failed to load RAG index '{RAG_INDEX}': {e}")
    rag = None


# ---------- response model ----------
class PredictResp(BaseModel):
    target_level: str
    inferred_from_incar: bool
    bandgap_eV: float
    neighbors: list
    structure: Optional[dict] = None
    structure_cif: Optional[str] = None
    structure_poscar: Optional[str] = None  # for POSCAR fallback in viewer
    source: str  # "incar" | "auto_no_incar" | "manual"


# ---------- routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    levels = [
        "AUTO (infer from INCAR)",
        "PBE",
        "SCAN",
        "HSE06(hfscreen=0.2)",
        "HSE06(hfscreen=0.208)",
    ]
    return templates.TemplateResponse("index.html", {"request": request, "levels": levels})


@app.post("/api/predict", response_model=PredictResp)
async def api_predict(
    structure: UploadFile = File(..., description="POSCAR or CIF"),
    incar: Optional[UploadFile] = File(None, description="Optional INCAR file"),
    target_level: str = Form(..., description="Pick a level or AUTO to infer from INCAR"),
    extras: str = Form("", description="Optional JSON dict or KEY=VAL lines"),
):
    # --- parse structure (POSCAR preferred, fallback CIF) ---
    raw = (await structure.read()).decode(errors="ignore")
    try:
        s = Structure.from_str(raw, fmt="poscar")
    except Exception:
        try:
            s = Structure.from_str(raw, fmt="cif")
        except Exception as e:
            return JSONResponse({"detail": f"Failed to parse structure: {e}"}, status_code=400)

    # --- INCAR (optional) ---
    incar_dict: Dict[str, Any] = {}
    lot_from_incar: Optional[str] = None
    if incar is not None:
        try:
            incar_raw = (await incar.read()).decode(errors="ignore")
            incar_dict = _read_incar(incar_raw)  # accepts raw text
            lot_from_incar = _infer_level_of_theory(incar_dict)
        except Exception as e:
            return JSONResponse({"detail": f"Failed to parse INCAR: {e}"}, status_code=400)

    # --- level selection (AUTO -> prefer INCAR; fallback PBE) ---
    want_auto = target_level.strip().upper().startswith("AUTO")
    inferred = False
    if want_auto:
        if lot_from_incar:
            chosen_level = lot_from_incar
            inferred = True
        else:
            chosen_level = "PBE"
            inferred = False
    else:
        chosen_level = target_level

    # --- extras (optional JSON or KEY=VAL lines) ---
    extras_dict: Dict[str, Any] = {}
    if extras.strip():
        try:
            extras_dict = json.loads(extras)
        except Exception:
            # support simple KEY=VAL lines
            for ln in extras.splitlines():
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    extras_dict[k.strip().upper()] = v.strip()

    # Merge for prompt (INCAR first so extras can override)
    prompt_kwargs = {**incar_dict, **extras_dict}

    # --- predict ---
    with torch.no_grad():
        g_emb = genc(s)  # CrystalGraph returns tensor on same device as model
        t_emb = encode_lot(lot_prompt(chosen_level, prompt_kwargs)).to(device)
        r_feats = rag_features(rag, s, chosen_level, k=8, device=device)
        y_pred, y_delta = model(g_emb, t_emb, r_feats, chosen_level)
        y = (y_pred + y_delta) if y_delta is not None else y_pred
        bandgap = float(y.item())

    # --- neighbors (best-effort; also try to attach their structures, INCAR, and formula) ---
    neighbors: List[Dict[str, Any]] = []
    if rag is not None:
        try:
            neighbors = rag.query(s, k=5, level_filter=chosen_level)
            for n in neighbors:
                calc_dir = n.get("calc_dir", "")
                struct_obj = None

                # Try POSCAR or CONTCAR
                for fname in ("POSCAR", "CONTCAR"):
                    p = os.path.join(calc_dir, fname)
                    if os.path.isfile(p):
                        try:
                            struct_obj = Structure.from_file(p)
                            break
                        except Exception:
                            continue

                # If structure found, attach structure dict and formula
                if struct_obj:
                    n["structure"] = serialize_structure(struct_obj)
                    n["formula"] = struct_obj.composition.reduced_formula  # e.g., "LiFePO4"
                else:
                    n["structure"] = None
                    n["formula"] = None

                # Try to read INCAR
                incar_path = os.path.join(calc_dir, "INCAR")
                if os.path.isfile(incar_path):
                    try:
                        with open(incar_path, "r") as f:
                            incar_raw = f.read()
                        n["incar"] = _read_incar(incar_raw)
                        # Simple INCAR similarity: fraction of matching keys (ignoring values)
                        if incar_dict and n["incar"]:
                            n["incar_similarity"] = (incar_dict == n["incar"])
                            diff_keys = {k for k in set(incar_dict) | set(n["incar"])
                                    if incar_dict.get(k) != n["incar"].get(k)}
                            n["incar_diff"] = {
                            k: (incar_dict.get(k), n["incar"].get(k)) for k in diff_keys
                        }

                    except Exception as e:
                        print(f"[RAG WARN] Failed to parse INCAR for neighbor: {e}")
                        n["incar"] = None
                        n["incar_similarity"] = None
                        n["incar_diff"] = {}
                else:
                    n["incar"] = None
                    n["incar_similarity"] = None
                    n["incar_diff"] = {}

        except Exception as e:
            print(f"[RAG WARN] neighbor query failed: {e}")
            neighbors = []

    # --- provide both CIF and POSCAR strings for robust client-side viewing ---
    cif_text = s.to(fmt="cif")
    poscar_text = s.to(fmt="poscar")

    return PredictResp(
        target_level=chosen_level,
        inferred_from_incar=inferred,
        bandgap_eV=bandgap,
        neighbors=neighbors,
        structure=serialize_structure(s),
        structure_cif=cif_text,
        structure_poscar=poscar_text,
        source=("incar" if inferred else ("auto_no_incar" if want_auto else "manual")),
    )


@app.post("/predict", response_class=HTMLResponse)
async def ui_predict(
    request: Request,
    structure: UploadFile = File(...),
    incar: Optional[UploadFile] = File(None),
    target_level: str = Form(...),
    extras: str = Form(""),
):
    api_resp = await api_predict(structure=structure, incar=incar, target_level=target_level, extras=extras)
    if isinstance(api_resp, JSONResponse) and api_resp.status_code != 200:
        # Render error at top of the page
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "levels": [
                    "AUTO (infer from INCAR)",
                    "PBE",
                    "SCAN",
                    "HSE06(hfscreen=0.2)",
                    "HSE06(hfscreen=0.208)",
                ],
                "error": api_resp.body.decode(),
            },
        )
    data = api_resp.dict()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "levels": [
                "AUTO (infer from INCAR)",
                "PBE",
                "SCAN",
                "HSE06(hfscreen=0.2)",
                "HSE06(hfscreen=0.208)",
            ],
            "prediction": data,
        },
    )


@app.get("/health")
def health():
    return {"ok": True, "device": str(device), "has_rag": rag is not None}
