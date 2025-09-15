This is a specific instruction for input parameter suggestions and calculation overview using RAG and OpenAI API. This part is independent of the main VASPilot codebase which is our multi-fidelity bandgap prediction code.

# **Note:** There is a bug in this part of the code and it will be fixed ASAP.
# Create a conda environment and install dependencies
```bash
conda create -n ragp python=3.10 
conda activate ragp
pip install fastapi uvicorn openai faiss-cpu python-multipart sentence-transformers Flask six
```

# Prepare RAG index for a new DFT data

### Change the data directory in the rag_generator.py
```bash
python rag_generator.py
```


# Now run the Flask to interact with the RAG system on web interface
```bash
cd /path/to/ragp
python vasp_rag.py
```

## Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Note:
1. Make sure to set your OpenAI API key in the environment variable `OPENAI_API_KEY`.
2. Add your API key in the `vasp_rag.py`.

