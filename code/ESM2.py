import argparse
import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PositionalEncoding:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def get_encoding(self, position: int) -> np.ndarray:
        encoding = torch.zeros(self.dimension)
        div_term = torch.exp(torch.arange(0, self.dimension, 2).float() * (-np.log(10000.0) / self.dimension))
        encoding[0::2] = torch.sin(position * div_term)
        encoding[1::2] = torch.cos(position * div_term)
        return encoding.numpy()


class CNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int = 256, dropout_rate: float = 0.6):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.fc_region = nn.Linear(hidden_dim, 1)
        self.fc_node = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv_stack(x).permute(0, 2, 1)
        region_features = features.mean(dim=1)
        p_region = self.fc_region(region_features).squeeze(-1)
        p_nodes = self.fc_node(features).squeeze(-1)
        return p_region, p_nodes


def parse_vector(value: str) -> np.ndarray:
    try:
        parsed = ast.literal_eval(str(value))
        return np.asarray(parsed, dtype=np.float32)
    except (ValueError, SyntaxError):
        cleaned = str(value).strip("[] \n\t").replace(",", " ")
        return np.fromstring(cleaned, sep=" ", dtype=np.float32)


def load_latent_embeddings(path: Path) -> tuple[dict[str, np.ndarray], int]:
    data = pd.read_csv(path, sep="\t")
    embeddings = {}
    for _, row in data.iterrows():
        vector = parse_vector(row["Latent_Space"])
        if vector.size > 0:
            embeddings[str(row["Entry"])] = vector

    if not embeddings:
        raise ValueError(f"No usable latent embeddings were found in {path}")

    latent_dim = len(next(iter(embeddings.values())))
    return embeddings, latent_dim


def get_esm2_embeddings(sequence: str, model: EsmModel, tokenizer: EsmTokenizer, device: torch.device, max_len: int, overlap: int) -> np.ndarray:
    model.eval()
    sequence = str(sequence)
    seq_len = len(sequence)
    esm_dim = model.config.hidden_size
    if seq_len == 0:
        return np.zeros((0, esm_dim), dtype=np.float32)

    stride = max_len - overlap
    summed = torch.zeros((seq_len, esm_dim), device=device, dtype=torch.float32)
    counts = torch.zeros(seq_len, device=device, dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, seq_len, stride):
            chunk = sequence[start:start + max_len]
            inputs = tokenizer(chunk, return_tensors="pt", add_special_tokens=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            chunk_embeddings = outputs.last_hidden_state[0, 1:-1, :]
            chunk_len = min(len(chunk), chunk_embeddings.shape[0])
            summed[start:start + chunk_len] += chunk_embeddings[:chunk_len]
            counts[start:start + chunk_len] += 1

    return (summed / counts.clamp(min=1.0).unsqueeze(-1)).cpu().numpy()


def build_windows(esm_embeddings: np.ndarray, latent_vector: np.ndarray, window_size: int, pe_dim: int = 10) -> np.ndarray:
    seq_len, esm_dim = esm_embeddings.shape
    padding = window_size // 2
    padded = np.vstack([
        np.zeros((padding, esm_dim), dtype=np.float32),
        esm_embeddings,
        np.zeros((padding, esm_dim), dtype=np.float32),
    ])

    encoder = PositionalEncoding(pe_dim)
    windows = []
    for index in range(seq_len):
        window = padded[index:index + window_size]
        coordinates = np.tile(latent_vector, (window_size, 1))
        start_encoding = np.tile(encoder.get_encoding(index + 1), (window_size, 1))
        end_encoding = np.tile(encoder.get_encoding(index + window_size), (window_size, 1))
        windows.append(np.concatenate([window, coordinates, start_encoding, end_encoding], axis=1))

    return np.asarray(windows, dtype=np.float32)


def predict_sequence(model: CNNModel, windows: np.ndarray, device: torch.device, batch_size: int, window_size: int) -> np.ndarray:
    model.eval()
    middle = window_size // 2
    predictions = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[start:start + batch_size], dtype=torch.float32, device=device)
            batch = batch.permute(0, 2, 1)
            _, node_scores = model(batch)
            predictions.extend(node_scores[:, middle].cpu().numpy())

    return np.asarray(predictions, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoGoBCE residue-level inference.")
    parser.add_argument("--input-csv", default="../data/LoGoBCE_independent_test.csv", help="CSV with ID, Sequence, and Protein_family columns.")
    parser.add_argument("--latent-tsv", default="latent_space_embeddings.tsv", help="CVAE latent embedding TSV produced by CVAE.py.")
    parser.add_argument("--model-path", default="LoGoBCE_parameter.pth", help="Trained LoGoBCE parameter file.")
    parser.add_argument("--output-dir", default="predictions", help="Directory for per-sequence and combined prediction CSV files.")
    parser.add_argument("--esm-model", default="facebook/esm2_t12_35M_UR50D", help="ESM-2 model name or local path.")
    parser.add_argument("--window-size", type=int, default=11)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--esm-max-len", type=int, default=1022)
    parser.add_argument("--esm-overlap", type=int, default=511)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    input_data = pd.read_csv(args.input_csv)
    required_columns = {"ID", "Sequence", "Protein_family"}
    missing = required_columns - set(input_data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    latent_embeddings, latent_dim = load_latent_embeddings(Path(args.latent_tsv))

    tokenizer = EsmTokenizer.from_pretrained(args.esm_model)
    esm_model = EsmModel.from_pretrained(args.esm_model).to(device)
    esm_dim = esm_model.config.hidden_size

    input_size = esm_dim + latent_dim + 20
    model = CNNModel(input_size=input_size, hidden_dim=args.hidden_dim, dropout_rate=args.dropout).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []

    for _, row in tqdm(input_data.iterrows(), total=len(input_data), desc="Running LoGoBCE inference"):
        entry = str(row["ID"])
        sequence = str(row["Sequence"])
        if entry not in latent_embeddings:
            raise KeyError(f"No latent embedding found for {entry}. Run CVAE.py on the same input CSV first.")

        esm_embeddings = get_esm2_embeddings(
            sequence,
            esm_model,
            tokenizer,
            device,
            max_len=args.esm_max_len,
            overlap=args.esm_overlap,
        )
        windows = build_windows(esm_embeddings, latent_embeddings[entry], args.window_size)
        scores = predict_sequence(model, windows, device, args.batch_size, args.window_size)

        per_sequence = pd.DataFrame({
            "ID": entry,
            "Position": np.arange(1, len(sequence) + 1),
            "Amino_acid": list(sequence),
            "Predicted_response_frequency": scores,
        })
        per_sequence.to_csv(output_dir / f"{entry}.csv", index=False)
        combined_rows.append(per_sequence)

    if combined_rows:
        pd.concat(combined_rows, ignore_index=True).to_csv(output_dir / "LoGoBCE_predictions.csv", index=False)


if __name__ == "__main__":
    main()
