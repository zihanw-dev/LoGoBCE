# LoGoBCE

LoGoBCE is a local-global framework for linear B-cell epitope prediction. This repository provides the open dataset split, trained model parameters, and inference scripts.

## Repository Layout

```text
LoGoBCE_open_source/
|-- README.md
|-- requirements.txt
|-- code/
|   |-- CVAE.py
|   |-- ESM2.py
|   |-- cvae_parameter.pth
|   `-- LoGoBCE_parameter.pth
|-- data/
|   |-- LoGoBCE_train.csv
|   `-- LoGoBCE_independent_test.csv
`-- scripts/
    `-- build_open_dataset.py
```

## Data

The open LoGoBCE dataset is split into:

- `data/LoGoBCE_train.csv`
- `data/LoGoBCE_independent_test.csv`

Each file contains three columns:

- `ID`: UniProt accession.
- `Sequence`: antigen amino-acid sequence.
- `Protein_family`: UniProt protein family annotation. Empty UniProt values are recorded as `Unknown`.

The independent test IDs are the fixed test set used by the LoGoBCE training script. The CVAE pre-training corpus is not included.

## Environment

The released parameters were produced with two inference/training environments. The commands below document the environment used for the CVAE stage and the package versions used for the LoGoBCE stage.

### CVAE Environment

```bash
mamba create -n esm2 python=3.10 pytorch=1.12.1 torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda activate esm2
mamba install pandas ipykernel tqdm
pip install --force-reinstall transformers==4.25.1 sentence-transformers==2.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --force-reinstall huggingface-hub==0.19.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download facebook/esm2_t12_35M_UR50D --local-dir ./esm2_local
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./minilm_local
```

The ESM-2 model used by the scripts is `facebook/esm2_t12_35M_UR50D`.

### LoGoBCE Environment

LoGoBCE residue-level inference was run in a conda environment named `wzh_fix` with:

- OS/platform: Linux x86_64, CentOS 7.9, glibc 2.17
- Conda: 23.9.0
- Python: 3.8.18
- Visible CUDA virtual package: `__cuda=12.4`

Core package versions:

```text
numpy                  1.26.4
pandas                 2.3.2
torch                  1.12.1
torchaudio             0.12.1
torchvision            0.15.2a0
transformers           4.25.1
sentence-transformers  2.2.2
huggingface-hub        0.19.4
tokenizers             0.13.3
safetensors            0.6.2
tqdm                   4.67.1
scikit-learn           1.7.2
scipy                  1.15.3
requests               2.32.5
```

For a compact inference setup, install PyTorch through conda/mamba first, then install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Inference

Run CVAE inference first to generate the global latent features:

```bash
cd code
python CVAE.py \
  --input-csv ../data/LoGoBCE_independent_test.csv \
  --output-tsv latent_space_embeddings.tsv \
  --model-path cvae_parameter.pth \
  --esm-model ./esm2_local \
  --text-model ./minilm_local
```

Then run LoGoBCE residue-level prediction:

```bash
python ESM2.py \
  --input-csv ../data/LoGoBCE_independent_test.csv \
  --latent-tsv latent_space_embeddings.tsv \
  --model-path LoGoBCE_parameter.pth \
  --esm-model ./esm2_local \
  --output-dir predictions
```

The output directory contains one CSV per protein and a combined file named `LoGoBCE_predictions.csv`.

## Rebuilding The Open Dataset

The dataset CSV files can be regenerated from the original local sequence files with:

```bash
cd scripts
python build_open_dataset.py --sequence-dir ../../data/sequence --output-dir ../data
```

The script queries the UniProt REST API for `Protein families` and writes `Unknown` when no protein-family value is returned.
