# Parse VASP calculation results
from __future__ import annotations
import os, re, json
from typing import Dict, Any, Optional
from pymatgen.io.vasp import Vasprun
from pymatgen.core import Structure

KEYS = ["LHFCALC","HFSCREEN","AEXX","ALGO","LDAU","LDAUL","LDAUU","LDAUJ","GGA","METAGGA","ISMEAR","SIGMA","ENCUT","KPOINTS"]

def read_incar(path:str)->Dict[str,Any]:
    d={}
    if not os.path.exists(path): return d
    for line in open(path, errors="ignore"):
        if "=" not in line: continue
        k,v = line.split("=",1); k=k.strip().upper(); v=v.split("!")[0].strip()
        if k in KEYS: d[k]=v
    return d

def level_of_theory(incar:Dict[str,Any], calc_dir:str)->str:
    # crude but effective mapper
    if incar.get("LHFCALC","").upper() in ("T",".TRUE.","TRUE","1","YES"):
        hs = float(incar.get("HFSCREEN", "0.2").replace(","," ").split()[0]) if "HFSCREEN" in incar else 0.2
        return f"HSE06(hfscreen={hs})"
    if incar.get("METAGGA","").upper() in ("SCAN","RSCAN"):
        return "SCAN"
    if incar.get("LDAU","").upper() in ("T",".TRUE.","TRUE","1","YES"):
        # summarize U
        U = incar.get("LDAUU","0").replace(","," ").strip()
        return f"PBE+U(U={U})"
    gga = incar.get("GGA","PBE").upper()
    return gga if gga!="-" else "PBE"

def parse_calc(calc_dir:str)->Optional[Dict[str,Any]]:
    vr = os.path.join(calc_dir,"vasprun.xml")
    poscar = os.path.join(calc_dir,"POSCAR")
    incar = read_incar(os.path.join(calc_dir,"INCAR"))
    if not os.path.exists(vr): return None
    try:
        v = Vasprun(vr, parse_potcar_file=False)
        bg = v.eigenvalue_band_properties  # (gap, cbm, vbm, is_direct)
        gap = float(bg[0]); is_direct = bool(bg[3])
        # Some runs can be metallic (gap=0)
        struct: Structure = v.final_structure or v.initial_structure
        lot = level_of_theory(incar, calc_dir)
        return dict(
            structure=struct.as_dict(),
            bandgap_eV=gap,
            is_direct=is_direct,
            level_of_theory=lot,
            incar=incar,
            calc_dir=os.path.abspath(calc_dir),
        )
    except Exception as e:
        print("[warn]", calc_dir, e)
        return None
