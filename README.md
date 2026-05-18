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
|   |-- LoGoBCE_parameter.pth
|   `-- cvae_parameter.pth        # download from GitHub Releases
|-- data/
|   |-- LoGoBCE_train.csv
|   |-- LoGoBCE_independent_test.csv
|   |-- LoGoBCE_train_response_frequency.csv
|   `-- LoGoBCE_independent_test_response_frequency.csv
`-- scripts/
    `-- build_open_dataset.py
```

## Data

The open LoGoBCE dataset is split into:

- `data/LoGoBCE_train.csv`
- `data/LoGoBCE_independent_test.csv`
- `data/LoGoBCE_train_response_frequency.csv`
- `data/LoGoBCE_independent_test_response_frequency.csv`

The sequence-level train/test files contain three columns:

- `ID`: UniProt accession.
- `Sequence`: antigen amino-acid sequence.
- `Protein_family`: UniProt protein family annotation. Empty UniProt values are recorded as `Unknown`.

The response-frequency files contain the original residue-level labels:

- `ID`: UniProt accession.
- `Position`: one-based residue position.
- `Amino_acid`: amino acid at the residue position.
- `Lower_bound`: lower bound from the original epitope curve.
- `Upper_bound`: upper bound from the original epitope curve.
- `Response_frequency`: observed response frequency. Missing values are left blank.

The independent test IDs are the fixed test set used by the LoGoBCE training script. The CVAE pre-training corpus is not included.

## Model Weights

The LoGoBCE residue-level model weight file `code/LoGoBCE_parameter.pth` is included in this repository.

The pre-trained CVAE weight file `cvae_parameter.pth` is large, so it is distributed as a GitHub Release asset instead of being tracked directly in the repository. Download it from:

```text
https://github.com/zihanw-dev/LoGoBCE/releases/tag/v1.0.0
```

After downloading, place the file at:

```text
code/cvae_parameter.pth
```

For command-line download, use:

```bash
wget -O code/cvae_parameter.pth https://github.com/zihanw-dev/LoGoBCE/releases/download/v1.0.0/cvae_parameter.pth
```

## Environment

The workflow uses two lightweight conda environments: one for generating CVAE global embeddings and one for LoGoBCE residue-level inference. The commands below use public package indexes and the standard Hugging Face endpoint.

### CVAE Environment

```bash
mamba create -n cvae python=3.10 pytorch=1.12.1 torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda activate cvae
mamba install pandas ipykernel tqdm
pip install transformers==4.25.1 sentence-transformers==2.2.2 huggingface-hub==0.19.4
huggingface-cli download facebook/esm2_t12_35M_UR50D --local-dir ./esm2_local
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./minilm_local
```

The ESM-2 model used by the scripts is `facebook/esm2_t12_35M_UR50D`.

### LoGoBCE Environment

```bash
mamba create -n logobce python=3.8.18 pytorch=1.12.1 torchvision torchaudio -c pytorch
conda activate logobce
pip install -r requirements.txt
```

The original LoGoBCE residue-level experiments were run on Linux x86_64 with Conda 23.9.0, Python 3.8.18, and the following core package versions:

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

If you use a CUDA-enabled GPU, install the PyTorch build that matches your local CUDA driver before running `pip install -r requirements.txt`.


## Inference

Run CVAE inference first to generate the global latent features:

```bash
cd code
# Make sure cvae_parameter.pth has been downloaded from the v1.0.0 release.
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
