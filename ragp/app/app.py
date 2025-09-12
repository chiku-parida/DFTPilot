import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI
from pydantic import BaseModel, Field

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

# =============================
# Config
# =============================
DEFAULT_DB_PATH = os.getenv("VASP_ASSIST_DB", "/home/charles/Downloads/faiss_db/faiss_db")
DEFAULT_EMB_MODEL = os.getenv("DEFAULT_EMB_MODEL", "all-MiniLM-L6-v2")
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
DEFAULT_GEN_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "32"))
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Fallback model IDs (public)
EMBEDDING_FALLBACKS = [
    lambda cur: cur,
    lambda cur: "sentence-transformers/" + cur if not cur.startswith("sentence-transformers/") else cur,
    lambda cur: "sentence-transformers/all-MiniLM-L6-v2",
    lambda cur: "sentence-transformers/all-MiniLM-L12-v2",
    lambda cur: "sentence-transformers/all-distilroberta-v1",
]

RERANKER_FALLBACKS = [
    lambda cur: cur,
    lambda cur: "cross-encoder/ms-marco-MiniLM-L6-v2",
    lambda cur: "cross-encoder/ms-marco-MiniLM-L12-v2",
]


def load_sentence_transformer(model_id: str, token: Optional[str]):
    """Try loading with token, then without; iterate through fallbacks."""
    last_err = None
    for mk in EMBEDDING_FALLBACKS:
        mid = mk(model_id)
        for tk in [token, None]:
            try:
                return SentenceTransformer(mid, token=tk)
            except Exception as e:
                last_err = e
                if "401" in str(e) or "Invalid credentials" in str(e):
                    # try without token next loop iteration automatically
                    continue
    raise RuntimeError(f"Failed to load embedding model '{model_id}': {last_err}")


def load_cross_encoder(model_id: str, device: str, token: Optional[str]):
    """Try loading CrossEncoder with token args, then without; iterate fallbacks."""
    last_err = None
    tok_args = ( {"token": token} if token else {} )
    for mk in RERANKER_FALLBACKS:
        mid = mk(model_id)
        # try with token
        try:
            return CrossEncoder(mid, device=device, tokenizer_args=tok_args, automodel_args=tok_args)
        except Exception as e:
            last_err = e
            # try without token
            try:
                return CrossEncoder(mid, device=device)
            except Exception as e2:
                last_err = e2
                continue
    raise RuntimeError(f"Failed to load reranker '{model_id}': {last_err}")

# =============================
# Pydantic Schemas (Validated Output)
# =============================
class RetrievedItem(BaseModel):
    formula: str
    spacegroup: str
    xc_family: str
    potcar_id: str = "unknown"
    cosine: Union[float, str, None] = "unknown"

class OutcarQualitative(BaseModel):
    verdict: str = Field(..., description="sufficient | borderline | insufficient | unknown")
    confidence: str = Field(..., description="high | medium | low | unknown")
    one_liner: str = Field(..., description="≤25 words, human-readable takeaway.")
    signals: List[str] = Field(..., description="2–4 cues, e.g., 'SCF 6–9; ΔE<1e-4; fmax≤0.03; ENCUT≈1.3×ENMAX'")

class OutcarNumbers(BaseModel):
    ionic_steps: Union[float, str]
    scf_steps_last: Union[float, str]
    scf_steps_mean: Union[float, str]
    scf_dE_last: Union[float, str]
    scf_reached_ediff: Union[bool, str]
    toten_final: Union[float, str]
    f_max: Union[float, str]
    f_mean: Union[float, str]

class OutcarPreview(BaseModel):
    qualitative: OutcarQualitative
    numbers: OutcarNumbers

class VaspAnalysis(BaseModel):
    sanity_check: str
    convergence_prediction: str
    outcar_preview: OutcarPreview
    actionable_tips: List[str]
    retrieved_summary: List[RetrievedItem]
    retrieved_vs_query: str
    rationale: str

# =============================
# Helper: Render metadata text like in your CLI tool
# =============================

def _kv_line(k: str, v):
    try:
        if isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v)
        elif isinstance(v, dict):
            v = ",".join(f"{kk}:{vv}" for kk, vv in list(v.items())[:8])
        return f"{k}: {v}"
    except Exception:
        return f"{k}: {v}"

def _render_potcar_blocks(blocks, max_blocks: int = 3, max_keys: int = 8) -> str:
    if not isinstance(blocks, list) or not blocks:
        return ""
    lines = ["POTCAR.blocks:"]
    for bi, blk in enumerate(blocks[:max_blocks], 1):
        if not isinstance(blk, dict):
            continue
        items = list(blk.items())[:max_keys]
        blk_str = ", ".join(f"{k}={v if not isinstance(v, list) else ','.join(map(str, v))}" for k, v in items)
        lines.append(f" block{bi}: {blk_str}")
    return "\n".join(lines)

def _render_outcar(oc: dict) -> str:
    if not isinstance(oc, dict) or not oc:
        return ""
    lines = ["OUTCAR:"]
    for k in ("toten_final", "converged", "ionic_steps"):
        if k in oc:
            lines.append(_kv_line(k, oc[k]))
    scf = oc.get("scf") or {}
    if isinstance(scf, dict) and scf:
        lines.append(" scf:")
        for k in ("total_scf_iters", "scf_steps_last", "scf_steps_mean", "scf_dE_last", "scf_reached_ediff"):
            if k in scf:
                lines.append(" " + _kv_line(k, scf[k]))
    forces = oc.get("forces") or {}
    if isinstance(forces, dict) and forces:
        lines.append(" forces:")
        for k in ("f_max", "f_mean", "f_rms"):
            if k in forces:
                lines.append(" " + _kv_line(k, forces[k]))
    return "\n".join(lines)

def _render_oszicar(oz: dict) -> str:
    if not isinstance(oz, dict) or not oz:
        return ""
    lines = ["OSZICAR:"]
    if "ionic_steps" in oz:
        lines.append(_kv_line("ionic_steps", oz["ionic_steps"]))
    scf = oz.get("scf") or {}
    if isinstance(scf, dict) and scf:
        lines.append(" scf:")
        for k in ("total_scf_iters", "scf_steps_last", "scf_steps_mean", "scf_dE_last", "scf_reached_ediff"):
            if k in scf:
                lines.append(" " + _kv_line(k, scf[k]))
    return "\n".join(lines)

def meta_to_full_text(m: dict) -> str:
    lines = []
    if isinstance(m, dict) and m.get("key_text"):
        lines.append(str(m["key_text"]).strip())
    pot = m.get("potcar") if isinstance(m, dict) else None
    if isinstance(pot, dict):
        blk_txt = _render_potcar_blocks(pot.get("blocks"))
        if blk_txt:
            lines.append(blk_txt)
    oc_txt = _render_outcar(m.get("outcar") if isinstance(m, dict) else None)
    if oc_txt:
        lines.append(oc_txt)
    oz_txt = _render_oszicar(m.get("oszicar") if isinstance(m, dict) else None)
    if oz_txt:
        lines.append(oz_txt)
    return "\n".join(lines)

# =============================
# Retrieval utilities
# =============================
_WORD_RE = re.compile(r"\w+", re.UNICODE)

def _simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def _truncate_for_ce(text: str, max_chars: int = 4000) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n...\n" + tail


def _faiss_metric_is_ip(index: faiss.Index) -> Optional[bool]:
    try:
        return getattr(index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT
    except Exception:
        return None


def perform_hybrid_search(
    query_text: str,
    metas: List[Dict[str, Any]],
    index: faiss.Index,
    model: SentenceTransformer,
    reranker: Optional[CrossEncoder] = None,
    k_initial: int = 50,
    k_final: int = 5,
) -> List[Dict[str, Any]]:
    # 1) Lexical BM25
    corpus = [meta_to_full_text(m) for m in metas]
    tokenized_corpus = [_simple_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _simple_tokenize(query_text)
    bm25_scores = bm25.get_scores(tokenized_query)
    lexical_indices = np.argsort(bm25_scores)[-k_initial:][::-1]

    # 2) Semantic FAISS
    emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    ntotal = index.ntotal if hasattr(index, "ntotal") else len(metas)
    k_sem = min(k_initial, ntotal)
    distances, semantic_indices = index.search(emb, k=k_sem)
    semantic_indices = semantic_indices.flatten()

    # Cosine-like scores
    cosine_scores: Dict[int, float] = {}
    if distances is not None and distances.size > 0:
        d = distances.flatten()
        ip = _faiss_metric_is_ip(index)
        for idx_val, raw in zip(semantic_indices, d):
            if idx_val < 0:
                continue
            if ip is True:
                cos = float(raw)
            elif ip is False:
                cos = float(1.0 - 0.5 * raw)
            else:
                cos = float(raw)
            cosine_scores[int(idx_val)] = cos

    # 3) Combine
    combined_indices = np.unique(np.concatenate([lexical_indices, semantic_indices]))
    retrieved_documents = [
        {
            "idx": int(idx),
            "meta": metas[int(idx)],
            "score": 0.0,
            "rank": 0,
            "cosine": cosine_scores.get(int(idx), None),
        }
        for idx in combined_indices
        if idx >= 0
    ]

    # 4) Re-rank
    if reranker is None:
        from torch import cuda
        device = "cuda" if cuda.is_available() else "cpu"
        reranker = CrossEncoder(DEFAULT_RERANKER_MODEL, device=device)
    pairs = [(query_text, _truncate_for_ce(meta_to_full_text(doc["meta"]))) for doc in retrieved_documents]
    reranker_scores = reranker.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)

    for i, doc in enumerate(retrieved_documents):
        doc["score"] = float(reranker_scores[i])

    reranked_docs = sorted(retrieved_documents, key=lambda x: x["score"], reverse=True)
    return reranked_docs[:k_final]

# =============================
# Coercion & Validation
# =============================
NUMERIC_KEYS = [
    "ionic_steps",
    "scf_steps_last",
    "scf_steps_mean",
    "scf_dE_last",
    "toten_final",
    "f_max",
    "f_mean",
]
BOOL_KEYS = ["scf_reached_ediff"]


def _default_qualitative() -> Dict[str, Any]:
    return {
        "verdict": "unknown",
        "confidence": "unknown",
        "one_liner": "Insufficient signals to judge convergence sufficiency.",
        "signals": ["n/a"],
    }


def _ensure_numbers_block(d: Dict[str, Any]) -> Dict[str, Any]:
    numbers = d.get("outcar_preview", {}).get("numbers")
    if "outcar_preview" in d and isinstance(d["outcar_preview"], dict):
        op = d["outcar_preview"]
        if numbers is None:
            numbers = {}
        for k in NUMERIC_KEYS + BOOL_KEYS:
            if k in op and k not in numbers:
                numbers[k] = op.pop(k)
        d["outcar_preview"]["numbers"] = numbers
    return d


def _coerce_outcar_preview(d: Dict[str, Any]) -> Dict[str, Any]:
    if "outcar_preview" not in d or not isinstance(d["outcar_preview"], dict):
        d["outcar_preview"] = {
            "qualitative": _default_qualitative(),
            "numbers": {k: "unknown" for k in NUMERIC_KEYS} | {"scf_reached_ediff": "unknown"},
        }
        return d

    d = _ensure_numbers_block(d)
    op = d["outcar_preview"]

    # Qualitative
    if "qualitative" not in op or not isinstance(op["qualitative"], dict):
        op["qualitative"] = _default_qualitative()
    else:
        q = op["qualitative"]
        q["verdict"] = q.get("verdict", "unknown") or "unknown"
        q["confidence"] = q.get("confidence", "unknown") or "unknown"
        q["one_liner"] = q.get("one_liner", "Insufficient signals to judge convergence sufficiency.") or "Insufficient signals to judge convergence sufficiency."
        sigs = q.get("signals", [])
        if isinstance(sigs, str):
            sigs = [sigs]
        if not isinstance(sigs, list) or len(sigs) == 0:
            sigs = ["n/a"]
        if len(sigs) > 4:
            sigs = sigs[:4]
        q["signals"] = sigs
        op["qualitative"] = q

    # Numbers
    if "numbers" not in op or not isinstance(op["numbers"], dict):
        op["numbers"] = {}
    for k in NUMERIC_KEYS:
        op["numbers"][k] = op["numbers"].get(k, "unknown")
    op["numbers"]["scf_reached_ediff"] = op["numbers"].get("scf_reached_ediff", "unknown")

    d["outcar_preview"] = op
    return d


def _coerce_retrieved_summary(d: Dict[str, Any]) -> Dict[str, Any]:
    rs = d.get("retrieved_summary", [])
    if not isinstance(rs, list):
        rs = []
    cleaned = []
    for it in rs[:3]:
        if not isinstance(it, dict):
            continue
        cleaned.append(
            {
                "formula": str(it.get("formula", "unknown") or "unknown"),
                "spacegroup": str(it.get("spacegroup", "unknown") or "unknown"),
                "xc_family": str(it.get("xc_family", "unknown") or "unknown"),
                "potcar_id": str(it.get("potcar_id", "unknown") or "unknown"),
                "cosine": it.get("cosine", "unknown"),
            }
        )
    d["retrieved_summary"] = cleaned
    return d


def _coerce_actionable_tips(d: Dict[str, Any]) -> Dict[str, Any]:
    tips = d.get("actionable_tips", [])
    if isinstance(tips, str):
        tips = [tips]
    if not isinstance(tips, list):
        tips = []
    tips = [t for t in tips if isinstance(t, str) and t.strip()]
    d["actionable_tips"] = tips
    return d


def _coerce_top_level_strings(d: Dict[str, Any]) -> Dict[str, Any]:
    for k in ["sanity_check", "convergence_prediction", "retrieved_vs_query", "rationale"]:
        v = d.get(k, "unknown")
        if not isinstance(v, str):
            v = str(v)
        d[k] = v
    return d


def _coerce_to_schema(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {
            "sanity_check": "unknown",
            "convergence_prediction": "unknown",
            "outcar_preview": {
                "qualitative": _default_qualitative(),
                "numbers": {k: "unknown" for k in NUMERIC_KEYS} | {"scf_reached_ediff": "unknown"},
            },
            "actionable_tips": [],
            "retrieved_summary": [],
            "retrieved_vs_query": "unknown",
            "rationale": "unknown",
        }
    d = _coerce_outcar_preview(d)
    d = _coerce_retrieved_summary(d)
    d = _coerce_actionable_tips(d)
    d = _coerce_top_level_strings(d)
    return d


def validate_and_normalize_json(result_text: str) -> str:
    if not result_text or not isinstance(result_text, str):
        raise RuntimeError("OpenAI response empty or not a string.")
    try:
        data = json.loads(result_text)
    except Exception as e:
        snippet = result_text[:800]
        raise RuntimeError(f"OpenAI response is not valid JSON: {e}\n--- Begin snippet ---\n{snippet}\n--- End snippet ---")
    data = _coerce_to_schema(data)
    try:
        obj = VaspAnalysis.model_validate(data)  # pydantic v2
        return obj.model_dump_json(indent=2)
    except Exception:
        obj = VaspAnalysis.parse_obj(data)  # pydantic v1
        return obj.json(indent=2)

# =============================
# Prompt scaffolding
# =============================
SCHEMA_TEXT = (
    "{\n"
    "  \"sanity_check\": string,\n"
    "  \"convergence_prediction\": string,\n"
    "  \"outcar_preview\": {\n"
    "    \"qualitative\": {\n"
    "      \"verdict\": \"sufficient | borderline | insufficient | unknown\",\n"
    "      \"confidence\": \"high | medium | low | unknown\",\n"
    "      \"one_liner\": string,\n"
    "      \"signals\": [string, string, string]\n"
    "    },\n"
    "    \"numbers\": {\n"
    "      \"ionic_steps\": number | \"a..b\" | \"unknown\",\n"
    "      \"scf_steps_last\": number | \"a..b\" | \"unknown\",\n"
    "      \"scf_steps_mean\": number | \"a..b\" | \"unknown\",\n"
    "      \"scf_dE_last\": number | \"<=x\" | \">=x\" | \"unknown\" | \"a..b\",\n"
    "      \"scf_reached_ediff\": true | false | \"unknown\",\n"
    "      \"toten_final\": number | \"a..b\" | \"unknown\",\n"
    "      \"f_max\": number | \"a..b\" | \"<=x\" | \"unknown\",\n"
    "      \"f_mean\": number | \"a..b\" | \"unknown\"\n"
    "    }\n"
    "  },\n"
    "  \"actionable_tips\": [string, string, string],\n"
    "  \"retrieved_summary\": [{\"formula\": string, \"spacegroup\": string, \"xc_family\": string | \"unknown\", \"potcar_id\": string | \"unknown\", \"cosine\": number | \"unknown\"}],\n"
    "  \"retrieved_vs_query\": string,\n"
    "  \"rationale\": string\n"
    "}"
)

SYSTEM_MSG = (
    "You are a Vienna Ab initio Simulation Package (VASP) assistant. "
    "Ground every statement only in the provided context. If a value is missing, write 'unknown'. "
    "Return a SINGLE JSON object that matches the schema. Keep strings concise and human-friendly. "
    "In 'rationale', provide a brief, high-level justification (no step-by-step reasoning)."
)

RUBRIC = (
    "- OUTCAR preview rubric → choose one verdict:\n"
    "  • sufficient: scf_reached_ediff=true, scf_steps_last ≤ 12, |ΔE_last| ≤ 1e-4 eV, f_max ≤ 0.05 eV/Å, ENCUT ≥ 1.3×ENMAX (or neighbours stable), k-mesh adequate for metallic, ISMEAR/SIGMA consistent.\n"
    "  • borderline: converged but slow (≈13–20 SCF iters), |ΔE_last| ~1e-4–1e-3 eV, f_max ~0.05–0.10 eV/Å, ENCUT ≈ (1.1–1.3)×ENMAX, or k-mesh sparse for metallic.\n"
    "  • insufficient: scf_reached_ediff=false or oscillation, |ΔE_last| ≥ 1e-3 eV, f_max ≥ 0.10 eV/Å, ENCUT < 1.1×ENMAX, or smearing/ISPIN inconsistent.\n"
    "  • unknown: not enough signal.\n"
    "- Produce outcar_preview.qualitative with one_liner ≤ 25 words and 2–4 terse signals (e.g., 'SCF 6–9; ΔE_last<1e-4; f_max≤0.03; ENCUT≈1.3×ENMAX').\n"
    "- Produce outcar_preview.numbers using neighbours with cosine ≥ 0.70 to compute bounds for each numeric key; if none pass, use widest bound from available neighbours; only use 'unknown' when no signal exists.\n"
    "- Provide exactly 3 concrete actionable_tips; one must recommend ENCUT from ENMAX (e.g., 1.3×max(ENMAX)) with a numeric recommendation and brief justification.\n"
)

# =============================
# FastAPI App
# =============================
app = FastAPI(title="VASP Assistant (FastAPI)")

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VASP Assistant</title>
  <style>
    :root { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial; }
    body { margin: 0; background: #0b1020; color: #e6ebf5; }
    .wrap { max-width: 960px; margin: 40px auto; padding: 24px; background: #101936; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
    h1 { margin-top: 0; font-size: 28px; }
    p.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #9fb3ff; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { background: #0d1430; padding: 16px; border-radius: 12px; border: 1px solid #1f2b57; }
    label { display: block; font-weight: 600; margin-bottom: 6px; }
    input[type="text"], input[type="number"], select { width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #2a3a78; background: #0a1130; color: #e6ebf5; }
    input[type="file"] { padding: 8px; background: #0a1130; color:#9fb3ff; border-radius: 8px; border: 1px dashed #2a3a78; width: 100%; }
    .actions { display: flex; gap: 12px; margin-top: 18px; }
    button { background: #4f7cff; color: white; border: none; padding: 12px 16px; border-radius: 10px; cursor: pointer; font-weight: 700; }
    button:hover { filter: brightness(1.1); }
    .hint { color: #a8b2d1; font-size: 13px; margin-top: 6px; }
    .footer { margin-top: 24px; color: #7f8ab2; font-size: 12px; }
    .code { background: #091027; border: 1px solid #1f2b57; padding: 12px; border-radius: 10px; overflow:auto; white-space: pre-wrap; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>⚛️ VASP Assistant</h1>
    <p class="mono">Upload <strong>INCAR</strong> and <strong>POSCAR</strong>, set your FAISS DB + models, then get a concise convergence assessment and tips.</p>

    <form action="/analyze" method="post" enctype="multipart/form-data">
      <div class="grid">
        <div class="card">
          <label>INCAR file</label>
          <input type="file" name="incar" required />
          <div class="hint">Plain text VASP INCAR.</div>
        </div>
        <div class="card">
          <label>POSCAR file</label>
          <input type="file" name="poscar" required />
          <div class="hint">Plain text VASP POSCAR/CONTCAR.</div>
        </div>
      </div>

      <div class="grid" style="margin-top: 16px;">
        <div class="card">
          <label>FAISS DB Path</label>
          <input type="text" name="db_path" value="__DB__" />
          <div class="hint">Must contain semantic.faiss, ids.json, metas.json</div>
        </div>
        <div class="card">
          <label>Embedding Model</label>
          <input type="text" name="emb_model" value="__EMB__" />
        </div>
      </div>

      <div class="grid" style="margin-top: 16px;">
        <div class="card">
          <label>Re-ranker Model</label>
          <input type="text" name="rerank_model" value="__RR__" />
        </div>
        <div class="card">
          <label>OpenAI Chat Model</label>
          <input type="text" name="gen_model" value="__GEN__" />
          <div class="hint">Requires OPENAI_API_KEY in environment.</div>
        </div>
      </div>

      <div class="grid" style="margin-top: 16px;">
        <div class="card">
          <label>k_initial</label>
          <input type="number" name="k_initial" value="50" />
        </div>
        <div class="card">
          <label>k_final</label>
          <input type="number" name="k_final" value="5" />
        </div>
      </div>

      <div class="actions">
        <button type="submit">Analyze</button>
        <a href="/health" target="_blank"><button type="button">Health check</button></a>
      </div>
      <div class="footer">Tip: set <code>OPENAI_API_KEY</code> and <code>VASP_ASSIST_DB</code> in your environment. Use <code>uvicorn fastapi_vasp_assistant_app:app --reload</code>.</div>
    </form>
  </div>
</body>
</html>
"""


def build_query_text(incar_text: str, poscar_text: str) -> str:
    return (
        "# Query Run Card (composed from user uploads)\n\n"
        "## INCAR\n" + incar_text.strip() + "\n\n" +
        "## POSCAR\n" + poscar_text.strip() + "\n"
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (INDEX_HTML
        .replace("__DB__", DEFAULT_DB_PATH)
        .replace("__EMB__", DEFAULT_EMB_MODEL)
        .replace("__RR__", DEFAULT_RERANKER_MODEL)
        .replace("__GEN__", DEFAULT_GEN_MODEL))
    return HTMLResponse(html)


@app.get("/health")
async def health():
    versions = {}
    try:
        import sentence_transformers as st
        versions["sentence_transformers"] = getattr(st, "__version__", "unknown")
    except Exception:
        versions["sentence_transformers"] = "not installed"
    try:
        import transformers as tr
        versions["transformers"] = getattr(tr, "__version__", "unknown")
    except Exception:
        versions["transformers"] = "not installed"
    try:
        import huggingface_hub as hh
        versions["huggingface_hub"] = getattr(hh, "__version__", "unknown")
    except Exception:
        versions["huggingface_hub"] = "not installed"

    ok = {
        "faiss_db_default": DEFAULT_DB_PATH,
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "has_hf_token": bool(HF_TOKEN),
        "default_models": {
            "embedding": DEFAULT_EMB_MODEL,
            "reranker": DEFAULT_RERANKER_MODEL,
            "generator": DEFAULT_GEN_MODEL,
        },
        "versions": versions,
        "notes": "If you see 401/Invalid credentials on public models, unset HF_TOKEN/HUGGINGFACEHUB_API_TOKEN or login with a valid token."
    }
    return JSONResponse(ok)


@app.post("/analyze")
async def analyze(
    request: Request,
    incar: UploadFile = File(...),
    poscar: UploadFile = File(...),
    db_path: str = Form(DEFAULT_DB_PATH),
    emb_model: str = Form(DEFAULT_EMB_MODEL),
    rerank_model: str = Form(DEFAULT_RERANKER_MODEL),
    gen_model: str = Form(DEFAULT_GEN_MODEL),
    k_initial: int = Form(50),
    k_final: int = Form(5),
):
    # Read uploads
    incar_text = (await incar.read()).decode("utf-8", errors="replace")
    poscar_text = (await poscar.read()).decode("utf-8", errors="replace")
    query_text = build_query_text(incar_text, poscar_text)

    # Load FAISS DB
    base = Path(db_path)
    try:
        index = faiss.read_index(str(base / "semantic.faiss"))
        ids = json.loads((base / "ids.json").read_text())
        metas = json.loads((base / "metas.json").read_text())
    except Exception as e:
        return JSONResponse({"error": f"Failed to load FAISS/metadata from {db_path}: {e}"}, status_code=400)

    if len(ids) != len(metas):
        return JSONResponse({"error": f"ids.json ({len(ids)}) and metas.json ({len(metas)}) length mismatch."}, status_code=400)

    # Models
    try:
        from torch import cuda
        device = "cuda" if cuda.is_available() else "cpu"
        embedder = load_sentence_transformer(emb_model, token=HF_TOKEN)
        reranker = load_cross_encoder(rerank_model, device=device, token=HF_TOKEN)
    except Exception as e:
        hint = (
            "Hint: If this is a 401/Invalid credentials on a PUBLIC model, you likely have an invalid HF token set. "
            "Unset HF_TOKEN/HUGGINGFACEHUB_API_TOKEN or login with a valid token: 'huggingface-cli login'."
        )
        return JSONResponse({"error": f"Failed to load models: {e}", "suggestion": hint}, status_code=400)

    # Retrieval
    try:
        top_results = perform_hybrid_search(
            query_text=query_text,
            metas=metas,
            index=index,
            model=embedder,
            reranker=reranker,
            k_initial=int(k_initial),
            k_final=int(k_final),
        )
    except Exception as e:
        return JSONResponse({"error": f"Hybrid search failed: {e}"}, status_code=500)

    # Build context
    context_pairs = []
    for i, res in enumerate(top_results):
        score = res["score"]
        idx = res.get("idx")
        if idx is None:
            try:
                idx = metas.index(res["meta"])  # fallback
            except Exception:
                idx = 0
        doc_id = ids[int(idx)]
        n_txt = meta_to_full_text(res["meta"])
        cos_disp = f"{res.get('cosine'):.3f}" if isinstance(res.get("cosine"), (float, int)) else "n/a"
        pair = (
            f"Neighbour {i+1} (ID={doc_id}, cross_score={score:.3f}, cosine={cos_disp})\n"
            f"Context:\n{n_txt}"
        )
        context_pairs.append(pair)
    context_block = "\n\n".join(context_pairs)

    # Retrieval summary
    top_cosines = [r.get("cosine") for r in top_results if isinstance(r.get("cosine"), (float, int))]
    top_cosines_sorted = sorted(top_cosines, reverse=True)[:3]
    max_cosine = max(top_cosines_sorted) if top_cosines_sorted else None
    confidence_flag = "unknown" if max_cosine is None else ("low" if max_cosine < 0.70 else "high")
    retrieval_summary = (
        f"cosine_top3={','.join(f'{c:.3f}' for c in top_cosines_sorted)}; "
        f"max_cosine={(f'{max_cosine:.3f}' if max_cosine is not None else 'unknown')}; "
        f"confidence_threshold=0.70; confidence={confidence_flag}"
    )

    # Compact neighbour metadata
    neighbour_meta_items = []
    for res in top_results[:3]:
        m = res.get("meta", {}) if isinstance(res, dict) else {}
        pos = m.get("poscar") if isinstance(m, dict) else None
        formula = None
        spacegroup = None
        if isinstance(pos, dict):
            formula = pos.get("formula") or pos.get("formula_pretty") or m.get("formula")
            spacegroup = pos.get("spacegroup") or m.get("spacegroup")
        else:
            formula = m.get("formula")
            spacegroup = m.get("spacegroup")
        xc_family = (
            m.get("xc_family")
            or (m.get("potcar", {}) if isinstance(m.get("potcar"), dict) else {}).get("xc_family")
            or "unknown"
        )
        cos = res.get("cosine") if isinstance(res.get("cosine"), (float, int)) else None
        neighbour_meta_items.append(
            {
                "formula": formula or "unknown",
                "spacegroup": spacegroup or "unknown",
                "xc_family": xc_family or "unknown",
                "potcar_id": (m.get("potcar", {}) or {}).get("id", "unknown"),
                "cosine": (round(cos, 3) if isinstance(cos, (float, int)) else "unknown"),
            }
        )

    neighbour_meta_lines = []
    for it in neighbour_meta_items:
        neighbour_meta_lines.append(
            f"- cos={it['cosine']} formula={it['formula']} spg={it['spacegroup']} xc={it['xc_family']}"
        )
    neighbour_meta_block = "\n".join(neighbour_meta_lines) if neighbour_meta_lines else "(no neighbour metadata available)"

    # Messages
    system_msg = SYSTEM_MSG
    user_msg = (
        f"Query (treat as run_card text):\n{query_text}\n\n"
        f"Retrieval summary: {retrieval_summary}\n\n"
        f"Neighbour metadata (top 3):\n{neighbour_meta_block}\n\n"
        f"Retrieved neighbour context (summaries):\n{context_block}\n\n"
        "Instructions:\n"
        "- Use ONLY the information above; unknown if missing.\n"
        "- In sanity_check, start with a compact template: 'Query system: FORMULA (SPG); XC=..., ENCUT=..., ISPIN=..., ISMEAR=.../SIGMA=...; Calc=relax|static|md|neb (inferred). Top neighbours cosine: [c1,c2,c3]; confidence: low|high|unknown. DFT settings: ENCUT=..., ISPIN=..., ISMEAR=.../SIGMA=..., KPOINTS mesh|KSPACING=.... POTCAR: family/id & ENMAX stats; encut/enmax ratio. Retrieved: [f1 (spg1, xc1); f2 (spg2, xc2); f3 (spg3, xc3)].'\n"
        "- Infer Calc from INCAR cues (e.g., NSW>0 & IBRION>=0 ⇒ relax; NSW=0 ⇒ static; IBRION=-1 & SMASS ⇒ md; images ⇒ neb). If any value is missing, write 'unknown'.\n"
        + RUBRIC +
        "- Keep convergence_prediction ≤ 2 sentences.\n"
        "- Keep rationale brief (no step-by-step reasoning).\n"
        f"- Schema:\n{SCHEMA_TEXT}\n"
        "- Output only the final JSON object, no extra prose."
    )

    # OpenAI call
    try:
        client = OpenAI()
        resp = client.responses.create(
            model=gen_model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            reasoning={"effort": "high"},
            response_format={"type": "json_object"},
            max_output_tokens=1200,
        )
        result_text = getattr(resp, "output_text", None)
        if not result_text:
            try:
                result_text = resp.output[0].content[0].text
            except Exception:
                result_text = resp.content[0].text
    except Exception as e:
        return JSONResponse({"error": f"OpenAI error: {e}"}, status_code=500)

    # Validate
    try:
        validated = validate_and_normalize_json(result_text)
    except Exception as e:
        return JSONResponse({"error": f"Validation error: {e}", "raw": result_text}, status_code=500)

    # Render nice HTML with collapsible JSON
    pretty = json.dumps(json.loads(validated), indent=2)
    html = f"""
    <html><head><meta charset='utf-8'><title>VASP Assistant — Result</title>
    <style>
      body{{font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background:#0b1020; color:#e6ebf5;}}
      .wrap{{max-width:960px;margin:40px auto;background:#101936;padding:24px;border-radius:16px;border:1px solid #1f2b57}}
      pre{{white-space:pre-wrap;background:#091027;border:1px solid #1f2b57;padding:16px;border-radius:12px;}}
      a.btn{{display:inline-block;margin-top:12px;background:#4f7cff;color:white;padding:10px 14px;border-radius:10px;text-decoration:none;font-weight:700}}
    </style>
    </head><body>
      <div class="wrap">
        <h2>VASP Assistant — Validated Result</h2>
        <pre>{pretty}</pre>
        <a class="btn" href="/">← Back</a>
      </div>
    </body></html>
    """
    return HTMLResponse(html)


# Optional JSON API for programmatic use
@app.post("/api/analyze")
async def api_analyze(
    incar: UploadFile = File(...),
    poscar: UploadFile = File(...),
    db_path: str = Form(DEFAULT_DB_PATH),
    emb_model: str = Form(DEFAULT_EMB_MODEL),
    rerank_model: str = Form(DEFAULT_RERANKER_MODEL),
    gen_model: str = Form(DEFAULT_GEN_MODEL),
    k_initial: int = Form(50),
    k_final: int = Form(5),
):
    incar_text = (await incar.read()).decode("utf-8", errors="replace")
    poscar_text = (await poscar.read()).decode("utf-8", errors="replace")
    query_text = build_query_text(incar_text, poscar_text)

    base = Path(db_path)
    index = faiss.read_index(str(base / "semantic.faiss"))
    ids = json.loads((base / "ids.json").read_text())
    metas = json.loads((base / "metas.json").read_text())
    if len(ids) != len(metas):
        return JSONResponse({"error": "ids/metas length mismatch"}, status_code=400)

    from torch import cuda
device = "cuda" if cuda.is_available() else "cpu"
embedder = load_sentence_transformer(emb_model, token=HF_TOKEN)
reranker = load_cross_encoder(rerank_model, device=device, token=HF_TOKEN)

    top_results = perform_hybrid_search(
        query_text, metas, index, embedder, reranker, int(k_initial), int(k_final)
    )

    context_pairs = []
    for i, res in enumerate(top_results):
        score = res["score"]
        idx = res.get("idx") or 0
        doc_id = ids[int(idx)]
        n_txt = meta_to_full_text(res["meta"])
        cos_disp = f"{res.get('cosine'):.3f}" if isinstance(res.get("cosine"), (float, int)) else "n/a"
        pair = (f"Neighbour {i+1} (ID={doc_id}, cross_score={score:.3f}, cosine={cos_disp})\nContext:\n{n_txt}")
        context_pairs.append(pair)
    context_block = "\n\n".join(context_pairs)

    top_cosines = [r.get("cosine") for r in top_results if isinstance(r.get("cosine"), (float, int))]
    top_cosines_sorted = sorted(top_cosines, reverse=True)[:3]
    max_cosine = max(top_cosines_sorted) if top_cosines_sorted else None
    confidence_flag = "unknown" if max_cosine is None else ("low" if max_cosine < 0.70 else "high")
    retrieval_summary = (
        f"cosine_top3={','.join(f'{c:.3f}' for c in top_cosines_sorted)}; max_cosine={(f'{max_cosine:.3f}' if max_cosine is not None else 'unknown')}; confidence_threshold=0.70; confidence={confidence_flag}"
    )

    neighbour_meta_items = []
    for res in top_results[:3]:
        m = res.get("meta", {}) if isinstance(res, dict) else {}
        pos = m.get("poscar") if isinstance(m, dict) else None
        formula = (pos or {}).get("formula") or (pos or {}).get("formula_pretty") or m.get("formula") or "unknown"
        spacegroup = (pos or {}).get("spacegroup") or m.get("spacegroup") or "unknown"
        xc_family = m.get("xc_family") or (m.get("potcar", {}) if isinstance(m.get("potcar"), dict) else {}).get("xc_family") or "unknown"
        cos = res.get("cosine") if isinstance(res.get("cosine"), (float, int)) else None
        neighbour_meta_items.append({
            "formula": formula,
            "spacegroup": spacegroup,
            "xc_family": xc_family,
            "potcar_id": (m.get("potcar", {}) or {}).get("id", "unknown"),
            "cosine": (round(cos, 3) if isinstance(cos, (float, int)) else "unknown"),
        })

    neighbour_meta_lines = [f"- cos={it['cosine']} formula={it['formula']} spg={it['spacegroup']} xc={it['xc_family']}" for it in neighbour_meta_items]
    neighbour_meta_block = "\n".join(neighbour_meta_lines) if neighbour_meta_lines else "(no neighbour metadata available)"

    system_msg = SYSTEM_MSG
    user_msg = (
        f"Query (treat as run_card text):\n{query_text}\n\n"
        f"Retrieval summary: {retrieval_summary}\n\n"
        f"Neighbour metadata (top 3):\n{neighbour_meta_block}\n\n"
        f"Retrieved neighbour context (summaries):\n{context_block}\n\n"
        "Instructions:\n"
        "- Use ONLY the information above; unknown if missing.\n"
        "- In sanity_check, start with a compact template: 'Query system: FORMULA (SPG); XC=..., ENCUT=..., ISPIN=..., ISMEAR=.../SIGMA=...; Calc=relax|static|md|neb (inferred). Top neighbours cosine: [c1,c2,c3]; confidence: low|high|unknown. DFT settings: ENCUT=..., ISPIN=..., ISMEAR=.../SIGMA=..., KPOINTS mesh|KSPACING=.... POTCAR: family/id & ENMAX stats; encut/enmax ratio. Retrieved: [f1 (spg1, xc1); f2 (spg2, xc2); f3 (spg3, xc3)].'\n"
        "- Infer Calc from INCAR cues (e.g., NSW>0 & IBRION>=0 ⇒ relax; NSW=0 ⇒ static; IBRION=-1 & SMASS ⇒ md; images ⇒ neb). If any value is missing, write 'unknown'.\n"
        + RUBRIC +
        "- Keep convergence_prediction ≤ 2 sentences.\n"
        "- Keep rationale brief (no step-by-step reasoning).\n"
        f"- Schema:\n{SCHEMA_TEXT}\n"
        "- Output only the final JSON object, no extra prose."
    )

    client = OpenAI()
    resp = client.responses.create(
        model=gen_model,
        input=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        reasoning={"effort": "high"},
        response_format={"type": "json_object"},
        max_output_tokens=1200,
    )
    result_text = getattr(resp, "output_text", None) or resp.output[0].content[0].text

    validated = validate_and_normalize_json(result_text)
    return JSONResponse(json.loads(validated))


# Run with: uvicorn fastapi_vasp_assistant_app:app --reload


# Favicon (prevent 404 in logs)
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(content="", status_code=204)
