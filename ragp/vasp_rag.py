import os
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
import traceback


# Setup

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # ensure your key is set in environment

FAISS_INDEX_FILE = "faiss.index"
TEXT_STORE_FILE = "texts.npy"

# Maximum characters to send for embedding (≈ safe for 8k tokens)
MAX_CHARS = 20000



# Embedding

def embed_text(text: str) -> np.ndarray:
    text = text[:MAX_CHARS]  # truncate to avoid exceeding model max tokens
    try:
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return np.array(resp.data[0].embedding, dtype="float32")
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {e}")



# FAISS helpers

def build_faiss_index(texts):
    embeddings = [embed_text(t) for t in texts]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(TEXT_STORE_FILE, np.array(texts))
    return index, texts


def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(TEXT_STORE_FILE):
        return None, None
    index = faiss.read_index(FAISS_INDEX_FILE)
    texts = np.load(TEXT_STORE_FILE, allow_pickle=True).tolist()
    return index, texts


def retrieve_similar(query_text, index, texts, k=3):
    try:
        query_emb = embed_text(query_text).reshape(1, -1)
        D, I = index.search(query_emb, k)
        results = []
        for idx in I[0]:
            if idx < len(texts):
                results.append(str(texts[idx]))  # always return string
        return results
    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {e}")



# INCAR generation with comma-separated text output

def prettify_incar(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return ", ".join(lines)


def generate_incar(query_text, retrieved):
    try:
        context = "\n".join(retrieved)

        prompt = f"""
You are an expert in VASP INCAR preparation.
Based on the POSCAR/INCAR context below, suggest the correct INCAR parameters.

Context:
{context}

Query:
{query_text}

Return your suggestion as plain text, one key=value per line.
Do NOT return JSON, dictionaries, or Python syntax.
Only return lines like:
ENCUT = 520
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-5
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for VASP input preparation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        raw_incar = resp.choices[0].message.content.strip()
        return prettify_incar(raw_incar)

    except Exception as e:
        return {"error": f"INCAR suggestion failed: {e}", "traceback": traceback.format_exc()}



# OUTCAR preview with total energy and final forces

def get_outcar_preview(outcar_text: str) -> str:
    """
    Extracts total energy and final forces from OUTCAR.
    Returns a concise preview.
    """
    lines = outcar_text.strip().splitlines()
    
    # Find last total energy (TOTEN)
    toten_lines = [l for l in lines if "free  energy   TOTEN" in l]
    total_energy = toten_lines[-1].split()[-2] if toten_lines else "N/A"

    # Extract last forces block (VASP TOTAL-FORCE section)
    forces = []
    start_idx = None
    for i, line in enumerate(lines):
        if "TOTAL-FORCE" in line or ("POSITION" in line and "TOTAL-FORCE" in line):
            start_idx = i + 2  # skip header lines
    if start_idx is not None:
        for l in lines[start_idx:]:
            if not l.strip():
                break
            parts = l.split()
            if len(parts) >= 4:
                atom_idx, fx, fy, fz = parts[:4]
                forces.append(f"Atom {atom_idx}: fx={fx}, fy={fy}, fz={fz}")

    preview = f"Total Energy (eV): {total_energy}\n"
    preview += "Final Forces:\n" + ("\n".join(forces) if forces else "N/A")
    return preview



async def home():
    return {"message": "VASP Assistant API is running. Use /suggest_incar endpoint."}


@app.post("/suggest_incar")
async def suggest_incar(
    poscar_file: UploadFile = None,
    incar_file: UploadFile = None,
    outcar_file: UploadFile = None,
    notes: str = Form(None),
):
    try:
        # Read uploaded files
        poscar_text = (await poscar_file.read()).decode("utf-8") if poscar_file else ""
        incar_text = (await incar_file.read()).decode("utf-8") if incar_file else ""
        outcar_text = (await outcar_file.read()).decode("utf-8") if outcar_file else ""

        # OUTCAR preview
        outcar_preview = get_outcar_preview(outcar_text) if outcar_text else ""

        # Combine inputs and truncate
        query_text = "\n".join([notes or "", poscar_text, incar_text, outcar_text])
        query_text = query_text[:MAX_CHARS]

        # Load or build FAISS index
        index, texts = load_faiss_index()
        if index is None:
            train_texts = [
                "Standard INCAR for relaxation: ENCUT=520, ISMEAR=0, SIGMA=0.05, EDIFF=1E-5",
                "INCAR for static calculation: NSW=0, IBRION=-1, LREAL=Auto, PREC=Accurate",
                "Spin-polarized INCAR: ISPIN=2, MAGMOM=5*1.0"
            ]
            index, texts = build_faiss_index(train_texts)

        # Retrieve similar contexts
        retrieved = retrieve_similar(query_text, index, texts)

        # Generate INCAR suggestion
        incar_suggestion = generate_incar(query_text, retrieved)

        return {
            "suggested_incar": incar_suggestion,
            "retrieved_contexts": retrieved,
            "outcar_preview": outcar_preview
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()}
        )