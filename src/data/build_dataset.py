# src/build_dataset.py
import os, glob, json
from utills.parse_vasp import parse_calc

def build_jsonl(raw_root:str, out_jsonl:str):
    n=0
    with open(out_jsonl,"w") as w:
        for d in sorted(glob.glob(os.path.join(raw_root,"*/"))):
            rec = parse_calc(d)
            if rec:
                w.write(json.dumps(rec)+"\n"); n+=1
    print("wrote", n, "records to", out_jsonl)

if __name__=="__main__":
    build_jsonl("/home2/llmhackathon25/data_primitive/train_raw/hybrid/CdO_hexagonal/vasp_std/Cd_EaH_0/","/home2/llmhackathon25/bg_llm/src/data/processed_data/all.jsonl")
