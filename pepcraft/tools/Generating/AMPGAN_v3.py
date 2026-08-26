
"""
Lightweight stand-in for the real AMPGAN_v3 GAN generator.

Keeps the exact same call contract as the original:
    AMPGAN_v3(input_data: dict) -> str

    input_data keys (unchanged from original):
        min_length         (int)
        max_length         (int)
        num_generations    (int)
        species_of_interest(str)  -- kept for interface parity / logging only,
                                     the stub does not condition on it
        folder_path        (str)  -- required

Writes/appends to folder_path/generated_sequences.csv with a 'sequence'
column, exactly like the original, so every downstream tool (filters,
verifiers) works unmodified.

No torch, no lightning, no trained checkpoint required.
"""

import os
import random
import pandas as pd

# 20 standard L-amino acids (uppercase). If you need D-amino acid sequences
# for the Damino_Filter to pass, lowercase letters represent D-amino acids
# in this pipeline's convention (see Damino_Filter.py / Lamino_Filter.py).
L_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _random_sequence(min_length: int, max_length: int, use_d_amino: bool = False) -> str:
    length = random.randint(min_length, max_length)
    seq = "".join(random.choice(L_AMINO_ACIDS) for _ in range(length))
    return seq.lower() if use_d_amino else seq


def AMPGAN_v3(input_data: dict) -> str:
    min_length = input_data.get("min_length", 10)
    max_length = input_data.get("max_length", 100)
    folder_path = input_data.get("folder_path", None)
    species_of_interest = input_data.get("species_of_interest", "ecoli")
    num_samples = input_data.get("num_generations", 4)
    # Optional extra flag (not in the original schema, defaults off) so you
    # can request D-amino sequences straight from the stub if you want them.
    use_d_amino = bool(input_data.get("use_d_amino", False))

    if not folder_path:
        raise ValueError("folder_path is required in input_data")

    os.makedirs(folder_path, exist_ok=True)
    csv_path = os.path.join(folder_path, "generated_sequences.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        num_original = len(df)
    else:
        df = None
        num_original = 0

    sequences = set()
    while len(sequences) < num_samples:
        sequences.add(_random_sequence(min_length, max_length, use_d_amino))
    sequences = list(sequences)[:num_samples]

    temp_df = pd.DataFrame({"sequence": sequences})

    if num_original > 0:
        for col in df.columns:
            if col != "sequence":
                temp_df[col] = "new"
        df = pd.concat([df, temp_df], ignore_index=True)
    else:
        df = temp_df

    df.to_csv(csv_path, index=False)

    return (
        f"Successfully generated {len(sequences)} new sequences using the lightweight "
        f"stub generator (min_length={min_length}, max_length={max_length}, "
        f"species_of_interest={species_of_interest} [not conditioned on by the stub]). "
        f"Appended to existing file: {csv_path}. "
        f"(Previous total: {num_original} | New total: {len(df)}). "
        f"Missing filter columns for the new entries were safely filled with NA.\n"
    )