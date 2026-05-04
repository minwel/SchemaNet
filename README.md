# SchemaNet

Research codebase for program clustering and classification.
## Features
- AST subtree extraction and multi-stage clustering.
- Encoders: cent (graph centrality) + IO (dynamic trace based).
- Classification with CodeBERT + schema + dynamic features.
- Caching of clustering results for faster re-runs.

## Requirements
- Python 3.9+ recommended.
- GPU optional (uses CUDA if available).
- Key dependencies include: `torch`, `transformers`, `pandas`, `scikit-learn`, `networkx`, `openpyxl`, `dependency-injector`, `tqdm`, `overrides`.


## Data Layout
Expected paths (default):
- Source code folder: `data/test/{code_from}/`
- Label file (Excel): `data/test/{code_from}.xlsx`
- Test cases: `data/testcase/`

The Excel file must contain at least:
- `fname`: filename in the source code folder
- `code`: raw source code
- `label`: class label (int)

Example:
```
data/
	test/
		3004/
			1.py
			2.py
		3004.xlsx
	testcase/
		testcase_3004.xml
```

## Quick Start
Run the full pipeline (clustering + classification):
```bash
python SchemaNet.py -pr 3004 -c 6
```

Key CLI options:
- `-pr / --prob`: problem id (used in data paths)
- `-c / --cls_num`: number of classes
- `-pt / --patience`: early stopping patience
- `-lr / --learning_rate`: learning rate
- `-sp / --split_ratio`: train/test split ratio (default 0.6)

Notes:
- The first run will create clustering caches under `tmp/`.
- CodeBERT will download the model from HuggingFace unless cached locally.


