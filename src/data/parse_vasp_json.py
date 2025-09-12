
from __future__ import annotations
import os, io, gzip, json
from typing import Dict, Any, Optional, Tuple
import numpy as np

from pymatgen.io.vasp import Vasprun
from pymatgen.core import Structure


KEYS = ["LHFCALC","HFSCREEN","AEXX","ALGO","LDAU","LDAUL","LDAUU","LDAUJ",
        "GGA","METAGGA","ISMEAR","SIGMA","ENCUT","KPOINTS"]


def _exists_any(path_noext: str, exts: tuple[str,...]) -> Optional[str]:
    """Return the first existing path among `path_noext + ext`."""
    for e in exts:
        p = path_noext + e
        if os.path.exists(p):
            return p
    return None

def _read_text(path: str) -> str:
    """Read text from plain or .gz path."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", errors="ignore") as f:
            return f.read()
    with open(path, "r", errors="ignore") as f:
        return f.read()

def _safe_float(x, default=None):
    try:
        return float(str(x).replace(",", " ").split()[0])
    except Exception:
        return default


def read_incar_any(dirpath: str) -> Dict[str, Any]:
    """
    Read INCAR or INCAR.gz from dir, return subset of keys (uppercased).
    """
    p = _exists_any(os.path.join(dirpath, "INCAR"), ("", ".gz"))
    d: Dict[str, Any] = {}
    if not p:
        return d
    for line in _read_text(p).splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().upper()
        v = v.split("!")[0].strip()
        if k in KEYS:
            d[k] = v
    return d

def level_of_theory(incar: Dict[str, Any]) -> str:
    # Hybrid (HSE06-like)
    if str(incar.get("LHFCALC", "")).upper() in {"T",".TRUE.","TRUE","1","YES"}:
        hs = _safe_float(incar.get("HFSCREEN", 0.2), 0.2)
        return f"HSE06(hfscreen={hs})"
    # Meta-GGA
    if str(incar.get("METAGGA", "")).upper() in {"SCAN","RSCAN"}:
        return "SCAN"
    # DFT+U
    if str(incar.get("LDAU", "")).upper() in {"T",".TRUE.","TRUE","1","YES"}:
        U = str(incar.get("LDAUU", "0")).replace(",", " ").strip()
        return f"PBE+U(U={U})"
    # Default PBE (or reported GGA)
    gga = str(incar.get("GGA", "PBE")).upper()
    return gga if gga and gga != "-" else "PBE"


def load_structure_any(dirpath: str) -> Optional[Structure]:
    """
    Try POSCAR(.gz) then CIF(.gz). Returns a Structure or None.
    """
    # POSCAR / CONTCAR paths
    for base in ("POSCAR", "CONTCAR"):
        p = _exists_any(os.path.join(dirpath, base), ("", ".gz"))
        if p:
            txt = _read_text(p)
            try:
                return Structure.from_str(txt, fmt="poscar")
            except Exception:
                pass
    # CIF
    for name in ("structure.cif", "STRUCTURE.cif", "POSCAR.cif", "CIF", "cif"):
        p = _exists_any(os.path.join(dirpath, name), ("", ".gz"))
        if p:
            txt = _read_text(p)
            try:
                return Structure.from_str(txt, fmt="cif")
            except Exception:
                pass
    return None


def bandgap_from_vasprun_any(dirpath: str) -> Optional[Tuple[float, float, float, bool, Structure]]:
    """
    Try vasprun.xml or vasprun.xml.gz in dirpath. Returns (gap, cbm, vbm, is_direct, structure).
    """
    p = _exists_any(os.path.join(dirpath, "vasprun.xml"), ("", ".gz"))
    if not p:
        return None
    # pymatgen's Vasprun can open .gz directly
    v = Vasprun(p, parse_potcar_file=False)
    gap, cbm, vbm, is_direct = v.eigenvalue_band_properties
    struct: Structure = v.final_structure or v.initial_structure
    return float(gap), float(cbm), float(vbm), bool(is_direct), struct

def bandgap_from_eigenval_any(dirpath: str) -> Optional[Tuple[float, float, float, bool]]:
    """
    Compute bandgap from EIGENVAL(.gz) occupancies if vasprun is absent.
    Returns (gap, vbm, cbm, is_direct) or None on failure.
    """
    p = _exists_any(os.path.join(dirpath, "EIGENVAL"), ("", ".gz"))
    if not p:
        return None
    lines = _read_text(p).splitlines()
    # find header line containing ne, nk, nb (after first ~10 lines)
    ne = nk = nb = None
    header_end = None
    for i in range(4, min(12, len(lines))):
        toks = lines[i].split()
        ints = []
        for t in toks:
            try:
                ints.append(int(t))
            except Exception:
                pass
        if len(ints) >= 3:
            ne, nk, nb = ints[:3]
            header_end = i
            break
    if nk is None or nb is None or header_end is None:
        return None

    pos = header_end + 1
    # detect spin by band row columns
    # skip k-point header
    if pos < len(lines):
        pos += 1
    first_band = lines[pos].split()
    cols = len(first_band)
    spin2 = (cols >= 5)  # (idx, Eup, occ_up, Edn, occ_dn)
    max_occ = 1.0 if spin2 else 2.0

    E = np.zeros((nk, nb), float)
    O = np.zeros((nk, nb), float)
    for k in range(nk):
        # skip k header line
        if pos < len(lines) and len(lines[pos].split()) <= 3:
            pos += 1
        for b in range(nb):
            toks = lines[pos].split()
            if spin2:
                Eup, occu, Edn, occd = map(float, (toks[1], toks[2], toks[3], toks[4]))
                E[k, b] = min(Eup, Edn)  # conservative aggregation
                O[k, b] = occu + occd
            else:
                e, o = map(float, (toks[1], toks[2]))
                E[k, b] = e; O[k, b] = o
            pos += 1
        while pos < len(lines) and lines[pos].strip() == "":
            pos += 1

    occ_thr = 0.5 * max_occ
    vbm_k = np.full(nk, -np.inf)
    cbm_k = np.full(nk, +np.inf)
    for k in range(nk):
        for b in range(nb):
            if O[k, b] > occ_thr:
                vbm_k[k] = max(vbm_k[k], E[k, b])
            else:
                cbm_k[k] = min(cbm_k[k], E[k, b])

    vbm = float(vbm_k.max())
    cbm = float(cbm_k.min())
    gap = max(0.0, cbm - vbm)
    direct_gap = float(np.min(cbm_k - vbm_k))
    is_direct = abs(direct_gap - gap) < 1e-3
    return gap, vbm, cbm, is_direct


def parse_calc_dir(dirpath: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse a single calculation directory.
    Priority: vasprun -> (gap, cbm, vbm, is_direct, structure)
    Fallback: EIGENVAL + POSCAR/CIF for structure.
    """
    # Try vasprun(.gz)
    try:
        vres = bandgap_from_vasprun_any(dirpath)
    except Exception as e:
        vres = None

    incar = read_incar_any(dirpath)

    if vres is not None:
        gap, cbm, vbm, is_direct, struct = vres
        lot = level_of_theory(incar)
        return {
            "structure": struct.as_dict(),
            "bandgap_eV": gap,
            "cbm_eV": cbm,
            "vbm_eV": vbm,
            "is_direct": is_direct,
            "level_of_theory": lot,
            "incar": incar,
            "calc_dir": os.path.abspath(dirpath),
        }

    #Fallback to EIGENVAL(.gz) + structure file
    try:
        eres = bandgap_from_eigenval_any(dirpath)
    except Exception:
        eres = None

    if eres is not None:
        gap, vbm, cbm, is_direct = eres
        struct = load_structure_any(dirpath)
        if struct is None:
            return None
        lot = level_of_theory(incar)
        return {
            "structure": struct.as_dict(),
            "bandgap_eV": float(gap),
            "cbm_eV": float(cbm),
            "vbm_eV": float(vbm),
            "is_direct": bool(is_direct),
            "level_of_theory": lot,
            "incar": incar,
            "calc_dir": os.path.abspath(dirpath),
        }

    return None

def build_jsonl_recursive(root: str, out_jsonl: str) -> int:
    """
    Walk `root` recursively, parse all folders with VASP outputs, and write JSONL dataset.
    Recognizes both normal and compressed (.gz) files.
    """
    n = 0
    with open(out_jsonl, "w") as w:
        for dirpath, dirnames, filenames in os.walk(root):
            # Make filenames lowercase for consistent matching
            lower_names = [f.lower() for f in filenames]

            # Debug: show which folder is being scanned
            print(f"[SCAN] {dirpath} contains {filenames}")

            # Check if this directory has any relevant VASP files
            if not any(
                name.startswith(("vasprun.xml", "eigenval", "poscar", "contcar", "structure"))
                for name in lower_names
            ) and not any(name.endswith(".gz") for name in lower_names):
                continue  # skip dirs with no relevant files

            # Parse the calculation
            rec = parse_calc_dir(dirpath)
            if rec:
                w.write(json.dumps(rec) + "\n")
                n += 1
            else:
                print(f"[SKIP] Could not parse {dirpath}")

    print(f"[DONE] Wrote {n} records to {out_jsonl}")
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build JSONL of band gaps from VASP folders (supports nested dirs, .gz files)")
    ap.add_argument("root", help="Root folder containing VASP calculations (nested allowed)")
    ap.add_argument("-o", "--out", default="all.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    rows = build_jsonl_recursive(args.root, args.out)
    print(f"Wrote {rows} records to {args.out}")
