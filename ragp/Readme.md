This is a specific instruction for input parameter suggestions and calculation overview using RAG and OpenAI API.

# Create a conda environment and install dependencies
```bash
conda create -n ragp python=3.10 -y
conda activate ragp
pip install fastapi uvicorn openai faiss-cpu
```
# Now run the FASTAPI server through uvicorn to interact with the RAG system on web interface
```bash
cd /path/to/ragp
uvicorn vasp_rag:app --reload
```

## Open your browser and go to [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Note:
1. Make sure to set your OpenAI API key in the environment variable `OPENAI_API_KEY`.
2. Add your API key in the `vasp_rag.py`.

