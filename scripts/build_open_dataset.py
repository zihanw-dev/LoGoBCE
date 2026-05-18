import argparse
import csv
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


INDEPENDENT_TEST_IDS = {
    "P03452", "P04977", "Q8JUX5", "Q8I0U8", "P03206", "Q19U29", "P03230", "P0C6W2", "Q9DUE3", "Q6NK15",
    "P01556", "P15917", "P23504", "Q81871", "P0DPI1", "A0A4Z2DS21", "A0A3Q0KGW0", "Q8IJ56", "P01857",
    "P03107", "M4QPE2", "P03314", "P30215", "P17763", "P03191", "P02647", "P05164", "A0A3Q0KCT5", "Q46409",
    "P35579", "A0A0N7J131", "P10324", "P13794", "P09841", "P43839", "P22776", "P0C6Y5", "A0A4Z2D4E3",
    "P07202", "P12977", "K9N796",
}


def read_sequences(sequence_dir: Path) -> dict[str, str]:
    records = {}
    for file_path in sorted(sequence_dir.glob("*.txt")):
        records[file_path.stem] = file_path.read_text(encoding="utf-8").strip()
    return records


def fetch_uniprot_families(accessions: list[str], chunk_size: int = 100) -> dict[str, str]:
    families: dict[str, str] = {}
    endpoint = "https://rest.uniprot.org/uniprotkb/accessions"

    for start in range(0, len(accessions), chunk_size):
        chunk = accessions[start:start + chunk_size]
        query = urlencode({
            "accessions": ",".join(chunk),
            "fields": "accession,sequence,protein_families",
            "format": "tsv",
        })
        with urlopen(f"{endpoint}?{query}", timeout=60) as response:
            text = response.read().decode("utf-8")

        reader = csv.DictReader(text.splitlines(), delimiter="\t")
        for row in reader:
            accession = row.get("Entry", "").strip()
            family = row.get("Protein families", "").strip()
            if accession:
                families[accession] = family or "Unknown"

        time.sleep(0.2)

    return families


def write_split(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Sequence", "Protein_family"])
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the open LoGoBCE train/test CSV files.")
    parser.add_argument("--sequence-dir", default="../../data/sequence", help="Directory containing one FASTA-free TXT sequence per UniProt ID.")
    parser.add_argument("--output-dir", default="../data", help="Directory for the generated CSV files.")
    args = parser.parse_args()

    sequence_dir = Path(args.sequence_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    sequences = read_sequences(sequence_dir)
    if not sequences:
        raise SystemExit(f"No sequence TXT files found in {sequence_dir}")

    families = fetch_uniprot_families(sorted(sequences))

    train_records = []
    test_records = []
    for accession, sequence in sorted(sequences.items()):
        row = {
            "ID": accession,
            "Sequence": sequence,
            "Protein_family": families.get(accession, "Unknown"),
        }
        if accession in INDEPENDENT_TEST_IDS:
            test_records.append(row)
        else:
            train_records.append(row)

    write_split(train_records, output_dir / "LoGoBCE_train.csv")
    write_split(test_records, output_dir / "LoGoBCE_independent_test.csv")

    print(f"Training records: {len(train_records)}")
    print(f"Independent test records: {len(test_records)}")
    print(f"Protein families marked Unknown: {sum(1 for row in train_records + test_records if row['Protein_family'] == 'Unknown')}")


if __name__ == "__main__":
    main()
