import os
import time
import pandas as pd
import torch

# ESMFold via HuggingFace transformers -- chosen over AlphaFold because it
# needs no MSA/database search step, making it the only one of the two
# that's realistically self-contained on CPU-only hardware. It will still
# be SLOW on CPU (this is a large model) -- test timing on one short
# peptide before running a full batch.
#
# First run will download the model weights (multi-GB) from HuggingFace --
# make sure you have space and a working internet connection for that
# one-time download; afterward it's cached locally.
_MODEL = None
_TOKENIZER = None


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    from transformers import AutoTokenizer, EsmForProteinFolding

    print("[Predict_Structure] Loading ESMFold (facebook/esmfold_v1) -- "
          "first run downloads multi-GB weights, this can take a while...")
    _TOKENIZER = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    _MODEL = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    _MODEL = _MODEL.float()  # CPU inference: stick to float32, not half precision
    _MODEL.eval()
    return _MODEL, _TOKENIZER


def _predict_pdb(sequence: str) -> str:
    """Runs ESMFold on a single sequence, returns PDB-format text."""
    model, _ = _load_model()
    with torch.no_grad():
        # transformers' ESMFold wrapper provides this convenience method
        # that returns ready-to-use PDB text directly -- no separate
        # CIF-to-PDB conversion step needed (unlike the old simplefold path).
        pdb_text = model.infer_pdb(sequence)
    return pdb_text


def Predict_Structure(input_data: dict) -> str:
    folder_path = input_data.get("folder_path", None)
    if not folder_path:
        raise ValueError("folder_path is required in input_data")

    csv_path = os.path.join(folder_path, "generated_sequences.csv")
    df = pd.read_csv(csv_path)

    if "structure_filter" in df.columns:
        df = df.drop(columns=["structure_filter"])

    filter_keys = [key for key in df.columns if "filter" in key]

    structure_dir = os.path.join(folder_path, "structure_output")
    os.makedirs(structure_dir, exist_ok=True)
    fasta_dir = os.path.join(folder_path, "fasta_input")
    os.makedirs(fasta_dir, exist_ok=True)

    pdb_paths = []
    fasta_paths = []
    num_predicted = 0

    for index, row in df.iterrows():
        already_done = (
            "predicted_structure_path" in df.columns
            and ".pdb" in str(row.get("predicted_structure_path", ""))
        )
        if already_done:
            pdb_paths.append(str(row["predicted_structure_path"]))
            fasta_paths.append(str(row.get("fasta_path", "")))
            continue

        if not all(row[key] == 1 for key in filter_keys):
            pdb_paths.append("")
            fasta_paths.append("")
            continue

        sequence = row["sequence"].upper()
        print(f"[Predict_Structure] folding sequence {index+1}: {sequence} "
              f"(CPU inference -- can be slow, please wait)...")

        start = time.time()
        pdb_text = _predict_pdb(sequence)
        elapsed = time.time() - start
        print(f"[Predict_Structure] done in {elapsed:.1f}s")

        fasta_path = os.path.join(fasta_dir, f"seq_{index}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">seq_{index}\n{sequence}\n")

        pdb_path = os.path.join(structure_dir, f"seq_{index}.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_text)

        pdb_paths.append(pdb_path)
        fasta_paths.append(fasta_path)
        num_predicted += 1

    df["predicted_structure_path"] = pdb_paths
    df["fasta_path"] = fasta_paths
    df.to_csv(csv_path, index=False)

    return f"Structure prediction completed for {num_predicted} sequences. Predicted structures saved in {structure_dir}."
