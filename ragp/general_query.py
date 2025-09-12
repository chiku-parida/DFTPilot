import io
import math
from typing import Dict, Any, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- FastAPI setup ----------
app = FastAPI(title="VASP Uploader (POSCAR + INCAR)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten if you deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file


# ---------- Helpers: INCAR ----------
def parse_incar(text: str) -> Dict[str, Any]:
    """
    Very tolerant INCAR parser: KEY = value  (comments after ! or #)
    Returns a dict with string/float/int/bool where sensible.
    """
    out: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "!", ";")):
            continue
        # strip trailing comments
        for c in ("!", "#", ";"):
            if c in line:
                line = line.split(c, 1)[0].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().upper()
        val = val.strip()

        # Coerce common types
        def to_scalar(s: str):
            s = s.strip()
            sl = s.lower()
            if sl in ("true", ".true.", "t", "yes", "y"):
                return True
            if sl in ("false", ".false.", "f", "no", "n"):
                return False
            # int?
            try:
                if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                    return int(s)
            except Exception:
                pass
            # float?
            try:
                return float(s.replace("d", "e"))  # allow Fortran 1d-3
            except Exception:
                return s  # string fallback

        # vector-like (e.g., MAGMOM = 2*0 4*1.0 or "1 2 3")
        if any(ch in val for ch in [",", " "]) and "*" not in val:
            parts = [p for p in val.replace(",", " ").split() if p]
            coerced = [to_scalar(p) for p in parts]
            out[key] = coerced
        else:
            out[key] = to_scalar(val)
    return out


# ---------- Helpers: POSCAR ----------
def _vector_from_line(line: str) -> np.ndarray:
    parts = [float(x.replace("D", "E")) for x in line.split()]
    if len(parts) < 3:
        raise ValueError("Lattice vector line must have 3 numbers.")
    return np.array(parts[:3], dtype=float)


def parse_poscar(text: str) -> Dict[str, Any]:
    """
    Supports VASP 4 & 5 style.
      line0: title
      line1: scale (float). If <0, it encodes target cell volume in Å^3
      line2-4: lattice rows
      line5: element symbols (VASP5) OR counts (VASP4)
      line6: counts if VASP5
      optional: 'Selective dynamics'
      next: 'Direct'/'Cartesian'
      then coordinates
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 7:
        raise ValueError("POSCAR too short.")

    title = lines[0]
    scale = float(lines[1].split()[0].replace("D", "E"))

    a = _vector_from_line(lines[2])
    b = _vector_from_line(lines[3])
    c = _vector_from_line(lines[4])

    # Detect VASP5 vs VASP4
    l5 = lines[5].split()
    vasp5 = all(token.isalpha() for token in l5)

    if vasp5:
        symbols = l5
        counts = [int(x) for x in lines[6].split()]
        coord_start = 7
    else:
        symbols = []  # unknown symbols (VASP4)
        counts = [int(x) for x in lines[5].split()]
        coord_start = 6

    # Optional "Selective dynamics"
    coord_mode_line = lines[coord_start].lower()
    selective = False
    if coord_mode_line.startswith("s"):
        selective = True
        coord_start += 1
        coord_mode_line = lines[coord_start].lower()

    # Direct/Cartesian
    if coord_mode_line.startswith("d"):
        coord_mode = "Direct"
    elif coord_mode_line.startswith("c") or coord_mode_line.startswith("k"):  # some folks use 'Cartesian'/'KPOINTS'
        coord_mode = "Cartesian"
    else:
        coord_mode = "Unknown"

    # Compute scaled lattice
    L = np.vstack([a, b, c])  # rows
    vol0 = float(np.dot(a, np.cross(b, c)))
    if abs(scale) < 1e-30:
        scale = 1.0
    if scale > 0:
        Ls = L * scale
    else:
        # negative scale encodes desired volume (Å^3)
        target_vol = abs(scale)
        factor = (target_vol / vol0) ** (1.0 / 3.0)
        Ls = L * factor

    a_s, b_s, c_s = Ls
    # Lattice metrics
    def norm(v): return float(np.linalg.norm(v))
    def angle(u, v):
        cosang = float(np.dot(u, v) / (norm(u) * norm(v)))
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))

    a_len, b_len, c_len = norm(a_s), norm(b_s), norm(c_s)
    alpha = angle(b_s, c_s)
    beta = angle(a_s, c_s)
    gamma = angle(a_s, b_s)
    vol = float(np.dot(a_s, np.cross(b_s, c_s)))

    natoms = sum(counts)
    # Composition string
    if symbols and len(symbols) == len(counts):
        comp = "".join(f"{el}{cnt if cnt != 1 else ''}" for el, cnt in zip(symbols, counts))
        species = [{"element": el, "count": cnt} for el, cnt in zip(symbols, counts)]
    else:
        comp = f"{natoms} atoms (symbols unknown)"
        species = [{"element": "X", "count": c} for c in counts]

    return {
        "title": title,
        "scale": scale,
        "lattice": Ls.tolist(),          # 3x3
        "a": round(a_len, 6),
        "b": round(b_len, 6),
        "c": round(c_len, 6),
        "alpha": round(alpha, 6),
        "beta": round(beta, 6),
        "gamma": round(gamma, 6),
        "volume": round(vol, 6),
        "coord_mode": coord_mode,
        "selective_dynamics": selective,
        "species": species,
        "natoms": natoms,
        "composition": comp,
    }


# ---------- Heuristic overview (no OUTCAR) ----------
def qualitative_overview(incar: Dict[str, Any], poscar_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gentle, human-readable check of whether the INCAR looks reasonable,
    *without* OUTCAR. Uses a few common rules of thumb.
    """
    encut = incar.get("ENCUT")
    ediff = incar.get("EDIFF")
    ispin = incar.get("ISPIN")
    isym = incar.get("ISYM")
    algo = str(incar.get("ALGO", "")).upper()
    ismear = incar.get("ISMEAR")
    sigma = incar.get("SIGMA")
    lreal = incar.get("LREAL")

    signals: List[str] = []

    # ENCUT heuristic (no POTCAR ENMAX available)
    if isinstance(encut, (int, float)):
        if encut >= 520:
            signals.append(f"ENCUT={encut} eV (robust default for PAW-PBE)")
        elif encut >= 400:
            signals.append(f"ENCUT={encut} eV (okay; consider ≥520 eV for safety)")
        else:
            signals.append(f"ENCUT={encut} eV (low; risk of basis-set error)")
    else:
        signals.append("ENCUT not set (uses POTCAR defaults)")

    # EDIFF
    if isinstance(ediff, (int, float)):
        if ediff <= 1e-5:
            signals.append(f"EDIFF={ediff:g} (tight)")
        elif ediff <= 1e-4:
            signals.append(f"EDIFF={ediff:g} (typical)")
        else:
            signals.append(f"EDIFF={ediff:g} (loose)")
    else:
        signals.append("EDIFF unknown")

    # Smearing
    if ismear is not None:
        try:
            ismear_i = int(ismear)
        except Exception:
            ismear_i = None
        if ismear_i is not None:
            if ismear_i > 0:
                signals.append(f"ISMEAR={ismear_i} (metallic assumption)")
            elif ismear_i == 0:
                signals.append("ISMEAR=0 (Gaussian smearing)")
            elif ismear_i == -5:
                signals.append("ISMEAR=-5 (tetrahedron + Blöchl; good for static DOS)")
            elif ismear_i == -1:
                signals.append("ISMEAR=-1 (Fermi smearing; safe default)")
    if sigma is not None and isinstance(sigma, (int, float)):
        signals.append(f"SIGMA={sigma:g}")

    # ISPIN
    if ispin is not None:
        try:
            ispin_i = int(ispin)
        except Exception:
            ispin_i = None
        if ispin_i == 2:
            signals.append("ISPIN=2 (spin-polarized)")
        elif ispin_i == 1:
            signals.append("ISPIN=1 (non-magnetic)")
    # ALGO
    if algo:
        signals.append(f"ALGO={algo}")
    # ISYM
    if isym is not None:
        signals.append(f"ISYM={isym}")

    # Verdict
    # Very rough: favor "sufficient" if ENCUT ≥ 520 and EDIFF ≤ 1e-5
    # Else "borderline" if ENCUT ≥ 400 and EDIFF ≤ 1e-4
    # Else "insufficient"
    verdict = "unknown"
    confidence = "low"

    if isinstance(encut, (int, float)) and isinstance(ediff, (int, float)):
        if encut >= 520 and ediff <= 1e-5:
            verdict, confidence = "sufficient", "medium"
        elif encut >= 400 and ediff <= 1e-4:
            verdict, confidence = "borderline", "low"
        else:
            verdict, confidence = "insufficient", "low"
    elif isinstance(encut, (int, float)) and encut >= 520:
        verdict, confidence = "borderline", "low"

    one_liner = {
        "sufficient": "INCAR looks tight for SCF; good starting point.",
        "borderline": "Likely ok, but tighten ENCUT/EDIFF or k-mesh if sensitive.",
        "insufficient": "Loose settings; expect convergence/sensitivity issues.",
        "unknown": "Not enough info; set explicit ENCUT and EDIFF.",
    }[verdict]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "one_liner": one_liner,
        "signals": signals[:4] or ["No clear signals"],
        "composition": poscar_meta.get("composition"),
        "natoms": poscar_meta.get("natoms"),
        "cell": {
            "a": poscar_meta.get("a"),
            "b": poscar_meta.get("b"),
            "c": poscar_meta.get("c"),
            "alpha": poscar_meta.get("alpha"),
            "beta": poscar_meta.get("beta"),
            "gamma": poscar_meta.get("gamma"),
            "volume": poscar_meta.get("volume"),
        },
    }


# ---------- HTML UI ----------
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VASP Upload • POSCAR + INCAR</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root { color-scheme: light dark; }
  body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0; }
  .wrap { max-width: 950px; margin: 0 auto; padding: 24px; }
  .card { background: rgba(255,255,255,0.05); border: 1px solid rgba(0,0,0,0.08);
          border-radius: 16px; padding: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }
  h1 { font-size: 24px; margin: 0 0 12px; }
  p.lead { color: #777; margin-top: 0; }
  label { display: block; font-weight: 600; margin: 12px 0 6px; }
  input[type=file] { display:block; width:100%; padding:10px; background:rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.15); border-radius:12px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; }
  .col { flex:1 1 300px; }
  button { background:#2563eb; color:white; border:none; padding:12px 16px; border-radius:12px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .result { margin-top: 20px; padding: 16px; border-radius: 12px; background: rgba(37,99,235,0.08); border: 1px solid rgba(37,99,235,0.25);}
  pre { white-space: pre-wrap; word-wrap: break-word; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .kv { background: rgba(0,0,0,0.04); border-radius: 10px; padding: 10px; }
  .kv h3 { margin: 0 0 6px; font-size: 14px; }
  .muted { color: #666; }
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>VASP Upload</h1>
    <p class="lead">Upload your <strong>POSCAR</strong> and <strong>INCAR</strong>. We’ll parse them and give a concise overview.</p>
    <form id="upload-form">
      <div class="row">
        <div class="col">
          <label for="poscar">POSCAR</label>
          <input id="poscar" name="poscar" type="file" required />
        </div>
        <div class="col">
          <label for="incar">INCAR</label>
          <input id="incar" name="incar" type="file" required />
        </div>
      </div>
      <div style="margin-top:16px;">
        <button id="submit-btn" type="submit">Analyze</button>
      </div>
    </form>
    <div id="result" class="result" style="display:none;">
      <div id="verdict"></div>
      <div class="grid" style="margin-top:12px;">
        <div class="kv"><h3>Composition</h3><div id="composition"></div></div>
        <div class="kv"><h3>Atoms</h3><div id="natoms"></div></div>
        <div class="kv"><h3>Cell (Å)</h3><div id="cell"></div></div>
        <div class="kv"><h3>Signals</h3><div id="signals"></div></div>
      </div>
      <div class="kv" style="margin-top:12px;">
        <h3>INCAR (parsed)</h3>
        <pre id="incar-json" class="muted"></pre>
      </div>
      <div class="kv" style="margin-top:12px;">
        <h3>POSCAR (summary)</h3>
        <pre id="poscar-json" class="muted"></pre>
      </div>
    </div>
  </div>
</div>
<script>
  const form = document.getElementById('upload-form');
  const resultBox = document.getElementById('result');
  const verdictBox = document.getElementById('verdict');
  const compBox = document.getElementById('composition');
  const natomsBox = document.getElementById('natoms');
  const cellBox = document.getElementById('cell');
  const signalsBox = document.getElementById('signals');
  const incarJson = document.getElementById('incar-json');
  const poscarJson = document.getElementById('poscar-json');
  const submitBtn = document.getElementById('submit-btn');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    submitBtn.disabled = true;

    const fd = new FormData(form);
    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        body: fd
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      resultBox.style.display = 'block';
      verdictBox.innerHTML = `<strong>${data.overview.verdict.toUpperCase()}</strong> — ${data.overview.one_liner} <span class="muted">(confidence: ${data.overview.confidence})</span>`;
      compBox.textContent = data.overview.composition || '—';
      natomsBox.textContent = data.overview.natoms ?? '—';
      const c = data.overview.cell || {};
      cellBox.textContent = `a=${c.a}, b=${c.b}, c=${c.c}, α=${c.alpha}, β=${c.beta}, γ=${c.gamma}, V=${c.volume}`;
      signalsBox.innerHTML = (data.overview.signals || []).map(s => `• ${s}`).join('<br/>');

      incarJson.textContent = JSON.stringify(data.incar, null, 2);
      poscarJson.textContent = JSON.stringify(data.poscar, null, 2);
    } catch (err) {
      alert(err.message);
    } finally {
      submitBtn.disabled = false;
    }
  });
</script>
</body>
</html>
"""


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def index(_: Request):
    return HTMLResponse(PAGE)


@app.post("/api/analyze")
async def api_analyze(
    poscar: UploadFile = File(..., description="Upload POSCAR"),
    incar: UploadFile = File(..., description="Upload INCAR"),
):
    # Size checks
    poscar_bytes = await poscar.read()
    incar_bytes = await incar.read()

    if len(poscar_bytes) > MAX_BYTES or len(incar_bytes) > MAX_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Each file must be ≤ {MAX_BYTES//(1024*1024)} MB."},
        )

    try:
        poscar_text = poscar_bytes.decode("utf-8", errors="replace")
        incar_text = incar_bytes.decode("utf-8", errors="replace")
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Failed to decode files as UTF-8."})

    # Parse
    try:
        poscar_meta = parse_poscar(poscar_text)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"POSCAR parse error: {e}"})

    try:
        incar_dict = parse_incar(incar_text)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"INCAR parse error: {e}"})

    overview = qualitative_overview(incar_dict, poscar_meta)

    return {
        "ok": True,
        "overview": overview,
        "poscar": {
            "title": poscar_meta["title"],
            "composition": poscar_meta["composition"],
            "natoms": poscar_meta["natoms"],
            "cell": {
                "a": poscar_meta["a"], "b": poscar_meta["b"], "c": poscar_meta["c"],
                "alpha": poscar_meta["alpha"], "beta": poscar_meta["beta"], "gamma": poscar_meta["gamma"],
                "volume": poscar_meta["volume"],
            },
            "coord_mode": poscar_meta["coord_mode"],
            "selective_dynamics": poscar_meta["selective_dynamics"],
            "species": poscar_meta["species"],
        },
        "incar": incar_dict,
    }
