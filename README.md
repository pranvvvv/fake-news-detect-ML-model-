# Fake News Detection Using BERT Sentence Embeddings + Classical ML Models

Project that produces an end-to-end pipeline for fake news detection using
sentence-transformers embeddings and classical machine learning models.

Project structure
```
project/
│── data/
│── output/
│── src/
│   ├── preprocessing.py
│   ├── embedder.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── utils.py
│── main.py
│── requirements.txt
│── README.md
```

Quick start

1. Install dependencies (recommend a virtualenv or Colab):

```bash
pip install -r requirements.txt
```

2. Place your dataset at `./data/news.csv` or a folder containing `Fake.csv` and `True.csv`.

3. Run the pipeline:

```bash
python main.py --data ./data/news.csv --output ./output
```

Outputs
- `output/embeddings.pkl` — saved embeddings for train set
- `output/logreg_grid.pkl`, `output/svc_grid.pkl`, `output/nb_tfidf_grid.pkl` — trained models
- `output/metrics_table.csv` — metrics table with accuracy/precision/recall/f1/roc_auc
- `output/roc_comparison.png`, `output/confusion_matrix_best.png`, `output/accuracy_f1_comparison.png`

Notes
- Reproducibility: seed fixed to 42 in the code.
- To run on Google Colab, upload the project and run the `pip install -r requirements.txt` step first.

Testing

Run the pipeline with:

```bash
python main.py --data ./data/news.csv
```

**APPENDIX A — REPRODUCIBILITY CHECKLIST & GIT REPO STRUCTURE**

Required files to include in your GitHub repo:

```
/data/news.csv                         # dataset (if licensing allows) or link in README
/src/preprocessing.py
/src/embedder.py
/src/train.py
/src/evaluate.py
/src/visualize.py
/main.py
/requirements.txt
/README.md
/output/metrics_table.csv
/output/roc_comparison.png
/output/confusion_matrix_best.png
/output/accuracy_f1_comparison.png
```

Quick reproduction instructions (to paste into README):

```
Create a Python environment: python -m venv venv && source venv/bin/activate

Install dependencies: pip install -r requirements.txt

Place dataset in ./data/news.csv with columns text,label (label values FAKE/REAL or 0/1)

Run: python main.py --data ./data/news.csv --output ./output

Generated artifacts will appear in ./output. Use insert_results_into_word.py to auto-insert PNGs and CSV into the Word doc.
```

Notes:
- If your system is Windows, activate the venv with: `venv\Scripts\Activate.ps1` (PowerShell) or `venv\Scripts\activate` (cmd).
- The pipeline may download a sentence-transformers model and require PyTorch; on CPU this can be slow. For faster embedding generation use a GPU or Colab.

