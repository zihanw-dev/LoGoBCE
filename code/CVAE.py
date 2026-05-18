import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProteinFamilyDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.entries = dataframe["ID"].astype(str).tolist()
        self.sequences = dataframe["Sequence"].fillna("").astype(str).tolist()
        self.families = dataframe["Protein_family"].fillna("Unknown").astype(str).tolist()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {
            "entry": self.entries[index],
            "sequence": self.sequences[index],
            "family": self.families[index],
        }


def make_collate_fn(text_embedder: SentenceTransformer):
    def collate_fn(batch: list[dict[str, str]]) -> dict[str, object]:
        families = [item["family"] for item in batch]
        with torch.no_grad():
            family_embeddings = text_embedder.encode(
                families,
                convert_to_tensor=True,
                show_progress_bar=False,
            ).float()
        return {
            "entries": [item["entry"] for item in batch],
            "sequences": [item["sequence"] for item in batch],
            "family_embeddings": family_embeddings,
        }

    return collate_fn


class ConditionalVAE(nn.Module):
    def __init__(self, esm_model_name: str, text_embedding_dim: int, latent_dim: int, hidden_dim: int, max_chunk_len: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        self.esm_encoder = EsmModel.from_pretrained(esm_model_name)
        self.max_chunk_len = max_chunk_len

        for parameter in self.esm_encoder.parameters():
            parameter.requires_grad = False

        esm_embedding_dim = self.esm_encoder.config.hidden_size
        self.esm_embedding_dim = esm_embedding_dim

        self.vae_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim + esm_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_dim + esm_embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, text_embedding_dim),
        )

    def encode_sequence(self, sequences: list[str], device: torch.device) -> torch.Tensor:
        cls_embeddings = []
        cls_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else 0
        sep_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else 2

        with torch.no_grad():
            for sequence in sequences:
                token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
                chunks = [token_ids[i:i + self.max_chunk_len] for i in range(0, len(token_ids), self.max_chunk_len)]
                if not chunks:
                    chunks = [[]]

                chunk_embeddings = []
                for chunk in chunks:
                    input_ids = torch.tensor([cls_id] + chunk + [sep_id], dtype=torch.long, device=device).unsqueeze(0)
                    outputs = self.esm_encoder(input_ids=input_ids)
                    chunk_embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze(0))

                cls_embeddings.append(torch.stack(chunk_embeddings, dim=0).mean(dim=0))

        return torch.stack(cls_embeddings, dim=0)

    def get_latent_mu(self, sequences: list[str], family_embeddings: torch.Tensor, device: torch.device) -> torch.Tensor:
        condition = self.encode_sequence(sequences, device)
        encoder_input = torch.cat([family_embeddings, condition], dim=1)
        hidden = self.vae_encoder(encoder_input)
        return self.fc_mu(hidden)

    def forward(self, sequences: list[str], family_embeddings: torch.Tensor, device: torch.device):
        condition = self.encode_sequence(sequences, device)
        encoder_input = torch.cat([family_embeddings, condition], dim=1)
        hidden = self.vae_encoder(encoder_input)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        decoder_input = torch.cat([z, condition], dim=1)
        reconstructed = self.vae_decoder(decoder_input)
        return reconstructed, mu, logvar


def generate_latent_embeddings(model: ConditionalVAE, loader: DataLoader, device: torch.device, output_path: Path) -> None:
    model.eval()
    entries = []
    latent_vectors = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating CVAE latent embeddings"):
            family_embeddings = batch["family_embeddings"].to(device)
            mu = model.get_latent_mu(batch["sequences"], family_embeddings, device)
            entries.extend(batch["entries"])
            latent_vectors.extend(mu.cpu().numpy())

    output = pd.DataFrame({
        "Entry": entries,
        "Latent_Space": [list(vector) for vector in latent_vectors],
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LoGoBCE global latent embeddings with the trained CVAE.")
    parser.add_argument("--input-csv", default="../data/LoGoBCE_independent_test.csv", help="CSV with ID, Sequence, and Protein_family columns.")
    parser.add_argument("--output-tsv", default="latent_space_embeddings.tsv", help="Output TSV used by ESM2.py.")
    parser.add_argument("--model-path", default="cvae_parameter.pth", help="Trained CVAE parameter file.")
    parser.add_argument("--esm-model", default="facebook/esm2_t12_35M_UR50D", help="ESM-2 model name or local path.")
    parser.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name or local path.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--text-embedding-dim", type=int, default=384)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--max-chunk-len", type=int, default=1022)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    data = pd.read_csv(args.input_csv)
    required_columns = {"ID", "Sequence", "Protein_family"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    text_embedder = SentenceTransformer(args.text_model)
    dataset = ProteinFamilyDataset(data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate_fn(text_embedder))

    model = ConditionalVAE(
        esm_model_name=args.esm_model,
        text_embedding_dim=args.text_embedding_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_chunk_len=args.max_chunk_len,
    ).to(device)
    model.esm_encoder.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    generate_latent_embeddings(model, loader, device, Path(args.output_tsv))


if __name__ == "__main__":
    main()
