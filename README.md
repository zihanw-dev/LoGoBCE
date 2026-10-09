# LoGoBCE

LoGoBCE is a local-global framework for linear B-cell epitope prediction. This repository provides the open dataset split, trained model parameters, and inference scripts.

## Repository Layout

```text
LoGoBCE_open_source/
|-- README.md
|-- environment.yml            # recommended single GPU environment
|-- requirements.txt           # fully pinned pip dependencies
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

The recommended environment for running the released LoGoBCE models uses Python 3.10, pip 24.2, `torch==1.12.1+cu116`, and `torchvision==0.13.1+cu116`. Linux x86_64 with an NVIDIA GPU and a driver compatible with CUDA 11.6 is recommended. CVAE global embedding generation and LoGoBCE residue-level inference use the same `logobce` conda environment.

Run the following commands from the repository root. `environment.yml` creates the environment and installs all pinned Python dependencies. Its package versions and CUDA 11.6 PyTorch index match `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate logobce
python -m pip check
```

All Python packages are installed together with exact PyTorch and torchvision versions so that transitive dependencies cannot replace the CUDA-enabled PyTorch build. PyPI is the primary package source, and the additional official PyTorch index supplies the CUDA 11.6 wheels. Installing these wheels through pip also avoids linking to newer conda CUDA libraries on clusters with an older system GLIBC.

Verify the runtime before inference:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The CUDA 11.6 wheels require an NVIDIA driver compatible with CUDA 11.x. The system CUDA toolkit does not need to match the wheels exactly because they include their user-space CUDA runtime.

Download the two Hugging Face models into `code/` so that their paths match the inference commands below:

```bash
huggingface-cli download facebook/esm2_t12_35M_UR50D --local-dir ./code/esm2_local
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./code/minilm_local
```

For offline inference, download both models beforehand and use their local directories with `--esm-model` and `--text-model` as shown below.

## Inference

Keep the `logobce` environment active for both inference steps. Run CVAE inference first to generate the global latent features:

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
