# ISE-Coursework

Tool Building Project for the Intelligent Software Engineering module at the University of Birmingham.

This repository contains a reproducible experimental pipeline for bug report classification,
comparing a baseline (`Naive Bayes + TF-IDF`) against a proposed model
(`Logistic Regression + TF-IDF`).

## 1. Environment setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Repository structure

- `datasets/*.csv`: labelled datasets per project (`pytorch`, `tensorflow`, `keras`, `incubator-mxnet`, `caffe`)
- `baseline_nb.py`: baseline experiment runner (CLI)
- `classification.py`: proposed model experiment runner (CLI)
- `combine_results.py`: merges baseline/proposed summaries into report tables
- `stats_test.py`: Wilcoxon signed-rank test on positive-class F1
- `run_pipeline.py`: one-command replication pipeline
- `analyze_results.py`: computes per-project deltas and overall win summary
- `experiment_utils.py`: shared data preparation, preprocessing, and evaluation logic
- `results/`: generated summaries, per-run arrays, and statistical outputs

## 3. Run experiments

### 3.1 Single project (default: `pytorch`)

```bash
.venv/bin/python baseline_nb.py --project pytorch --repeats 30
.venv/bin/python classification.py --project pytorch --repeats 30
.venv/bin/python combine_results.py --project pytorch
.venv/bin/python stats_test.py --project pytorch
```

### 3.2 Full replication across all projects

```bash
.venv/bin/python run_pipeline.py --projects pytorch tensorflow keras incubator-mxnet caffe --repeats 30
```

When all 5 projects are executed, this also generates:

- `results/all_projects_combined_results.csv`
- `results/all_projects_wilcoxon_test.csv`

Then generate report-ready win/delta summaries:

```bash
.venv/bin/python analyze_results.py
```

## 4. Key outputs

Per project `<project>`:

- `results/<project>_NB.csv`
- `results/<project>_LR.csv`
- `results/<project>_combined_results.csv`
- `results/<project>_wilcoxon_test.csv`
- `results/<project>_NB_pos_f1.npy`
- `results/<project>_LR_pos_f1.npy`

Backward-compatible files are still written for `pytorch`:

- `results/baseline_f1_pos.npy`
- `results/proposed_f1_pos.npy`

## 5. Reproducibility and marking support

The repository root includes the required coursework artifacts:

- `requirements.pdf`
- `manual.pdf`
- `replication.pdf`

Source text versions are provided as:

- `requirements_source.txt`
- `manual_source.txt`
- `replication_source.txt`

Regenerate PDFs with:

```bash
.venv/bin/python generate_pdfs.py
```
