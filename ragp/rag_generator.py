import os
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configuration
# This script automatically finds the directory where it is located
# and searches all its subdirectories for VASP data.
data_directory = Path("path/to/data/train_final_raw")#__file__).parent
db_directory = Path("./faiss_db")
hf_token = "your-hf-token"  # Replace with your actual Hugging Face token
# Ensure the database directory exists
db_directory.mkdir(exist_ok=True)

# VASP Data Processing
def parse_vasp_data(directory: Path):
    """
    Parses VASP data from a single directory (e.g., one calculation).
    Combines INCAR, POSCAR, and OUTCAR data into a single text block.
    """
    data = {}
    incar_path = directory / "INCAR"
    poscar_path = directory / "POSCAR"
    outcar_path = directory / "OUTCAR"
    
    # Check for file existence before reading
    if incar_path.exists():
        data['incar'] = incar_path.read_text()
    if poscar_path.exists():
        data['poscar'] = poscar_path.read_text()
    if outcar_path.exists():
        data['outcar'] = outcar_path.read_text()
        
    return data

def combine_data_to_text(data: dict) -> str:
    """
    Combines parsed VASP data into a single text block for embedding.
    """
    text_parts = []
    if 'poscar' in data:
        text_parts.append("POSCAR:\n" + data['poscar'])
    if 'incar' in data:
        text_parts.append("INCAR:\n" + data['incar'])
    if 'outcar' in data:
        text_parts.append("OUTCAR:\n" + data['outcar'])
        
    return "\n---\n".join(text_parts)

# Embedding and Indexing 
def build_faiss_index(all_data: list):
    """
    Builds a FAISS index from the provided VASP data.
    """
    # Use a local, open-source model for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2', use_auth_token=hf_token)
    
    texts = [combine_data_to_text(d['data']) for d in all_data]
    
    print(f"Creating embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    print("Building FAISS index...")
    embeddings_np = np.array(embeddings, dtype='float32')
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)
    
    # Save the index and metadata
    faiss.write_index(index, str(db_directory / "vasp_index.faiss"))
    
    metadata = [{'id': d['id'], 'data': d['data']} for d in all_data]
    with open(db_directory / "vasp_metadata.json", "w") as f:
        json.dump(metadata, f)
        
    print(f"Successfully created index with {index.ntotal} documents.")


if __name__ == "__main__":
    if not data_directory.exists():
        print(f"Error: Data directory not found at {data_directory}")
    else:
        print("Starting index creation...")
        
        all_data = []
        
        # Use rglob to find all POSCAR files recursively
        for poscar_path in data_directory.rglob("POSCAR"):
            # The parent of the POSCAR file is the VASP calculation directory
            subdir = poscar_path.parent
            data = parse_vasp_data(subdir)
            if data and ('poscar' in data or 'incar' in data or 'outcar' in data):
                # Use the path relative to the script location as the ID
                relative_path = subdir.relative_to(data_directory)
                all_data.append({'id': str(relative_path), 'data': data})
                print(f"Found calculation directory: {relative_path}")
        
        if all_data:
            build_faiss_index(all_data)
        else:
            print("No VASP data found in the specified directory.")