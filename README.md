# VASPilot

A multi-fidelity bandgap predictor powered by LLM.


# Installation

It is recommended to first create and activate a virtual environment.

```
$ conda create -n vaspilot python=3.11
$ conda activate vaspilot
```

Then install the package with all dependencies:

```
$ pip install .[all]
```
Or install the package in editable mode with development dependencies:

```
$ pip install -e .[dev]
```

# Data Preparation

Data preparation involves creating a JSONL file and a RAG index file from VASP calculation folder. You can use the provided script to process your raw data.


### Prepare JSONL file
```
$ !python3 /data/parse_vasp_json.py /path/to/train_raw -o path/to/processed_data/train.jsonl

``` 

### Build RAG index file using SOAP fingerprints and FAISS

```
$ python /data/rag_index_build.py path/to/processed_data/train.jsonl path/to/processed_data/rag.index

``` 



# Training

```
$ python /scripts/train.py \
    --jsonl /path/to/processed_data/train.jsonl \
    --index_path /path/to/processed_data/rag.index \
    --epochs 501 \
    --lr 1e-3 \
    --batch_size 4 \
    --device cuda \
    --patience 100 \
    --val_frac 0.1 \
    --k_neighbors 8 \
    --ckpt_best_path /path/to/checkpoint/bg_llm_best.pt \
    --ckpt_last_path /path/to/checkpoint/bg_llm_last.pt \
    --log_json_path /path/to/log/bg_llm_log.json \
    --seed 42
```


# Prediction

```
$ python /scripts/batch_predict.py \
  --input /home2/llmhackathon25/data/train_final_processed/test.jsonl \
  --output /home2/llmhackathon25/data/test_final/test_with_predictions_n256.jsonl \
  --checkpoint /home2/llmhackathon25/chkpt_test/bg_llm_best_final_n128.pt \
  --rag-index /home2/llmhackathon25/data/train_final_processed/rag.index \
  --device cuda \
  --k 8
``` 
# App:

```
$ cd app/
$ uvicorn app:app --reload --port 8000
```
Then open your browser and go to [`http://127.0.0.1:8000`](http://127.0.0.1:8000)

Now you can upload a POSCAR file and select the fidelity level to get the bandgap prediction.





# Contributors:
- Chiku Parida (cparida.ai@gmail.com)
- Savya Agarwala (savya10@gmail.com)
