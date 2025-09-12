# make_rag_index.py
from __future__ import annotations
import json, sys
from typing import List, Dict, Any, Iterable, Tuple, Optional

import faiss
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from collections import OrderedDict


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def collect_species(jsonl: str) -> List[str]:
    """Collect a stable, sorted set of species symbols across the dataset."""
    seen: "OrderedDict[str, None]" = OrderedDict()
    for r in iter_jsonl(jsonl):
        try:
            s = Structure.from_dict(r["structure"])
        except Exception:
            continue
        for site in s.sites:
            sym = site.specie.symbol
            if sym not in seen:
                seen[sym] = None
    # deterministic order by atomic number
    return sorted(seen.keys(), key=lambda x: Element(x).Z)

def struct_to_ase(struct: Structure) -> Atoms:
    return Atoms(
        numbers=[site.specie.number for site in struct.sites],
        positions=struct.cart_coords,
        cell=struct.lattice.matrix,
        pbc=True,
    )


def build_index(
    jsonl: str,
    out_path: str,
    r_cut: float = 5.0,
    n_max: int = 4,
    l_max: int = 6,
    sigma: float = 0.5,
    batch_log_every: int = 100,
) -> Tuple[int, int]:
    """
    Build a FAISS IP index (cosine after L2-normalization) over SOAP descriptors.
    Saves:
      - out_path
      - out_path + ".meta.json"
      - out_path + ".soap.json"   (for exact query-time reconstruction)
    Returns: (n_indexed, n_failed)
    """
    species = collect_species(jsonl)
    if not species:
        raise RuntimeError("No species found in dataset; cannot build SOAP basis.")

    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic=True,
        sparse=False,
        average="off",
    )

    dim = soap.get_number_of_features()
    index = faiss.IndexFlatIP(dim)

    meta: List[Dict[str, Any]] = []
    n_ok = n_fail = 0

    for i, r in enumerate(iter_jsonl(jsonl), 1):
        try:
            struct = Structure.from_dict(r["structure"])
            atoms = struct_to_ase(struct)

            v = soap.create([atoms])[0].astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(v)
            index.add(v)

            meta.append({
                "bandgap_eV": float(r.get("bandgap_eV", np.nan)),
                "level_of_theory": r.get("level_of_theory", ""),
                "calc_dir": r.get("calc_dir", ""),
                "is_direct": bool(r.get("is_direct", False)),
                "incar": r.get("incar", {}),
            })
            n_ok += 1
        except Exception:
            n_fail += 1

        if (i % batch_log_every) == 0:
            print(f"[build_index] processed={i} indexed={n_ok} failed={n_fail}", file=sys.stderr)

    faiss.write_index(index, out_path)
    with open(out_path + ".meta.json", "w") as w:
        json.dump(meta, w)

    # Persist SOAP config for query-time reconstruction
    with open(out_path + ".soap.json", "w") as w:
        json.dump({
            "species": species,
            "r_cut": r_cut,
            "n_max": n_max,
            "l_max": l_max,
            "sigma": sigma,
            "periodic": True,
            "sparse": False,
            "average": "off",
        }, w)

    print(f"[build_index] DONE -> {out_path} (+ .meta.json, .soap.json). Indexed={n_ok}, failed={n_fail}, dim={dim}")
    return n_ok, n_fail


class RAG:
    """
    Loads FAISS index, metadata, and the exact SOAP config used for building.
    Ensures the query descriptor dimensionality matches the index.
    """
    def __init__(self, index_path: str):
        self.index = faiss.read_index(index_path)
        with open(index_path + ".meta.json") as f:
            self.meta = json.load(f)
        with open(index_path + ".soap.json") as f:
            cfg = json.load(f)

        # Recreate SOAP exactly as during building
        self.soap = SOAP(**cfg)
        dim = self.soap.get_number_of_features()
        if dim != self.index.d:
            raise ValueError(
                f"SOAP dim {dim} != index dim {self.index.d}. "
                f"Mismatched SOAP config/species. Loaded cfg: {cfg}"
            )

    def query(self, struct: Structure, k: int = 8, level_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        atoms = struct_to_ase(struct)
        q = self.soap.create([atoms])[0].astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, max(k * 3, k))
        out: List[Dict[str, Any]] = []
        for d, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            m = self.meta[idx]
            if level_filter and m["level_of_theory"] != level_filter:
                continue
            out.append({"similarity": float(d), **m})
            if len(out) >= k:
                break
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build FAISS RAG index from JSONL dataset with SOAP descriptors.")
    ap.add_argument("jsonl", help="Path to processed all.jsonl")
    ap.add_argument("out", help="Output index path, e.g. rag.index")
    ap.add_argument("--rcut", type=float, default=5.0)
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    build_index(
        jsonl=args.jsonl,
        out_path=args.out,
        r_cut=args.rcut,
        n_max=args.nmax,
        l_max=args.lmax,
        sigma=args.sigma,
        batch_log_every=args.log_every,
    )
