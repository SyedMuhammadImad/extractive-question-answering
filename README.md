# Extractive question answering

Pretrained DistilBERT extractive QA, official development examples, multi-reference exact match and token-multiset F1. A local Streamlit interface is included.

The earlier notebook was repaired into a reusable module plus the same setup → experiment → results notebook flow. Original source files remain on the laptop. See `VERIFICATION.json` for completed checks and `metrics.json` for fresh results when available.

```sh
python -m venv .venv
python -m pip install -r requirements.txt
python qa.py --data dev-v1.1.json --limit 100 --output metrics.json
```

Supply your dataset files at the indicated paths. Raw corpora, model binaries, credentials, pictures and videos are excluded. Pretrained model downloads happen locally. Evaluation sizes and dataset limitations are explicit in the results; small runs are functional evidence, not a broad benchmark.

Run `streamlit run app.py` for the local interface or `pytest -q` for metric/input checks. Download the official SQuAD v1.1 development file from https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json.
