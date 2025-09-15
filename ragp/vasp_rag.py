import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from flask import Flask, request, jsonify, render_template_string



# --- Configuration ---
OPENAI_API_KEY = "your_openai_api_key"
HF_TOKEN = os.getenv("your_hugging_face_token")  # optional
DB_DIRECTORY = Path("path/to/ragp/faiss_db/")


app = Flask(__name__)


try:
    FAISS_INDEX = faiss.read_index(str(DB_DIRECTORY / "vasp_index.faiss"))
    with open(DB_DIRECTORY / "vasp_metadata.json", "r") as f:
        METADATA = json.load(f)
    print("FAISS index loaded successfully.")
    EMBEDDING_MODEL = SentenceTransformer(
        'all-MiniLM-L6-v2', token=HF_TOKEN
    ) if HF_TOKEN else SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading FAISS index or metadata: {e}")
    FAISS_INDEX = None
    METADATA = []
    EMBEDDING_MODEL = None



def assess_convergence(incar_text: str, outcar_text: str) -> str:
    """Give qualitative convergence preview instead of numeric forces."""
    import re
    def find(param):
        m = re.search(rf"\b{param}\s*=\s*([^\s#]+)", incar_text, flags=re.I)
        return m.group(1) if m else None

    encut = find("ENCUT")
    ediff = find("EDIFF")
    ediffg = find("EDIFFG")
    algo = find("ALGO")

    reached_elec = re.search(r"reached required accuracy.*stopping electronic", outcar_text, flags=re.I)
    reached_ion = re.search(r"reached required accuracy.*stopping structural", outcar_text, flags=re.I)

    verdict = "Likely converged ✅" if reached_elec and reached_ion else \
              "Potential issues ⚠️" if reached_elec else \
              "Unlikely to converge ❌"

    md = ["### Verdict", verdict, "", "### Why"]
    if encut: md.append(f"- ENCUT = {encut}")
    if ediff: md.append(f"- EDIFF = {ediff}")
    if ediffg: md.append(f"- EDIFFG = {ediffg}")
    if algo: md.append(f"- ALGO = {algo}")
    if reached_elec: md.append("- Electronic convergence achieved")
    if reached_ion: md.append("- Ionic convergence achieved")

    md.append("")
    md.append("### Quick Fixes")
    if not reached_elec:
        md.append("- Adjust EDIFF slightly to improve SCF convergence")
    if not reached_ion:
        md.append("- Check relaxation settings and EDIFFG")
    if not encut or (encut.isdigit() and int(encut) < 450):
        md.append("- Increase ENCUT by 10–20%")
    if not algo or algo.upper() in {"FAST", "VERYFAST"}:
        md.append("- Use ALGO=Normal for stability")

    return "\n".join(md)


def generate_rag_prompt(user_query: str, retrieved_contexts: list) -> Dict[str, Any]:
    """Structured prompt for concise Markdown output."""
    context = "\n---\n".join(retrieved_contexts)
    system_prompt = (
        "You are an expert VASP assistant. ONLY use provided contexts.\n"
        "Output must be concise Markdown:\n"
        "## INCAR suggestion\n(fenced code block with minimal settings)\n"
        "## Rationale\n(3–5 concise bullets)\n"
        "## Expected convergence\n(qualitative notes)\n"
        "Do NOT invent missing values."
    )
    user_prompt = (
        f"Analyze this VASP case:\n{user_query}\n\n"
        f"Similar contexts:\n{context}\n"
        "Respond using the exact Markdown structure above."
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}

def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VASP RAG Assistant</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.7/dist/purify.min.js"></script>
</head>
<body class="bg-gray-50 text-gray-800">
<div class="container mx-auto p-4">
  <h1 class="text-3xl text-center font-bold text-teal-700 mb-4">VASP RAG Assistant</h1>
  <main class="grid grid-cols-1 md:grid-cols-2 gap-4">
    <form id="upload-form" class="space-y-3 bg-white p-4 rounded shadow">
      <input type="file" name="incar" class="block w-full text-sm border p-2" />
      <input type="file" name="poscar" class="block w-full text-sm border p-2" />
      <textarea name="notes" placeholder="Notes (e.g., spin-polarized)" class="block w-full text-sm border p-2"></textarea>
      <button id="submit-btn" class="bg-teal-600 text-white px-4 py-2 rounded w-full">Get Suggestions</button>
      <div id="loading" class="hidden text-center mt-2">Analyzing...</div>
    </form>
    <div class="bg-white p-4 rounded shadow">
      <div id="results-display">Suggestions will appear here.</div>
    </div>
  </main>
</div>
<script>
document.getElementById('upload-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const btn = document.getElementById('submit-btn');
  const loading = document.getElementById('loading');
  const results = document.getElementById('results-display');
  btn.disabled = true; loading.classList.remove('hidden'); results.innerHTML = '';
  try {
    const resp = await fetch('/suggest', {method:'POST', body:formData});
    const data = await resp.json();
    const html = DOMPurify.sanitize(marked.parse(data.suggestions || ''));
    results.innerHTML = html;
  } catch(err){
    results.textContent = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
    loading.classList.add('hidden');
  }
});
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/suggest", methods=["POST"])
def suggest():
    if FAISS_INDEX is None or EMBEDDING_MODEL is None:
        return jsonify({"error": "FAISS or embedding model not loaded"}), 500
    if not OPENAI_API_KEY:
        return jsonify({"error": "Missing OPENAI_API_KEY"}), 500

    try:
        incar_text = request.files.get("incar").read().decode("utf-8", errors="replace") if request.files.get("incar") else ""
        poscar_text = request.files.get("poscar").read().decode("utf-8", errors="replace") if request.files.get("poscar") else ""
        notes = request.form.get("notes", "")
        query_text = f"Notes:\\n{notes}\\n---\\nINCAR:\\n{incar_text}\\n---\\nPOSCAR:\\n{poscar_text}"

        # Embed and search
        query_embedding = EMBEDDING_MODEL.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        k = min(5, FAISS_INDEX.ntotal)
        distances, indices = FAISS_INDEX.search(query_embedding, k)

        retrieved_contexts = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(METADATA): 
                continue
            doc = METADATA[idx]['data']
            ctx = (
                f"INCAR:\\n{doc.get('incar','')[:800]}\\n---\\n"
                f"POSCAR:\\n{doc.get('poscar','')[:800]}\\n---\\n"
                f"Convergence Assessment:\\n{assess_convergence(doc.get('incar',''), doc.get('outcar',''))}"
            )
            retrieved_contexts.append(ctx)

        prompts = generate_rag_prompt(query_text, retrieved_contexts)
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":prompts["system_prompt"]},
                {"role":"user","content":prompts["user_prompt"]}
            ],
            temperature=0.2
        )
        suggestions = response.choices[0].message.content
        return jsonify({
            "suggestions": suggestions,
            "retrieved_contexts": [escape_html(c) for c in retrieved_contexts]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
