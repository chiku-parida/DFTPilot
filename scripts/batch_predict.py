from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
import argparse
import torch

from calculators.calculator import BandGapCalculator

def predict_from_record(calc: BandGapCalculator, rec: Dict[str, Any], k_neighbors: int = 5) -> Dict[str, Any]:
    """Predict bandgap for a single JSONL record from your dataset."""
    struct_dict = rec["structure"]                 # structure dict (pymatgen format)
    incar = rec.get("incar") or {}                 # may be {}
    target_level = None if incar else rec.get("level_of_theory", None)
    return calc.predict(
        structure=struct_dict,
        incar=incar,
        target_level=target_level,
        k_neighbors=k_neighbors,
    )

def main():
    ap = argparse.ArgumentParser(description="Batch bandgap prediction for JSONL (one JSON object per line).")
    ap.add_argument("--input", required=True, help="Path to input JSONL file (one record per line).")
    ap.add_argument("--output", required=True, help="Path to output JSONL file.")
    ap.add_argument("--checkpoint", required=True, help="Path to trained checkpoint .pt")
    ap.add_argument("--rag-index", required=True, help="Path to RAG FAISS index (without .meta.json)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda","cpu"])
    ap.add_argument("--k", type=int, default=5, help="RAG neighbors (k)")
    args = ap.parse_args()

    calc = BandGapCalculator(
        checkpoint=args.checkpoint,
        rag_index=args.rag_index,
        device=args.device,
    )

    in_path = Path(args.input)
    out_path = Path(args.output)
    n_ok = n_fail = 0

    with in_path.open("r") as fin, out_path.open("w") as fout:
        for idx, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                res = predict_from_record(calc, rec, k_neighbors=args.k)
                # keep original fields + add predictions
                rec_out = {
                    **rec,
                    "predicted_bandgap_eV": res["predicted_bandgap_eV"],
                    "neighbors": res["neighbors"],
                    "target_level_used": res["target_level"],
                }
                fout.write(json.dumps(rec_out) + "\n")
                n_ok += 1
            except Exception as e:
                print(f"[WARN] record {idx} failed: {e}")
                n_fail += 1
            if idx % 50 == 0:
                print(f"[INFO] processed {idx} (ok={n_ok}, fail={n_fail})")

    print(f"[DONE] wrote {n_ok} results to {out_path} (failures: {n_fail})")

if __name__ == "__main__":
    main()
