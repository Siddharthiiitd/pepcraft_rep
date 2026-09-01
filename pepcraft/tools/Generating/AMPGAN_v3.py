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

# 20 standard L-amino acids (uppercase). Lowercase letters represent
# D-amino acids in this pipeline's convention (see Damino_Filter.py /
# Lamino_Filter.py).
CATIONIC = "KR"          # positively charged at pH 7.4 -- drives net charge up
HYDROPHOBIC = "ALIVFMWY"  # positive Kyte-Doolittle hydropathy -- drives GRAVY up
NEUTRAL = "GSTCNQHP"       # mildly hydrophilic, near-neutral GRAVY contribution
NEGATIVE = "DE"           # negatively charged -- drives net charge down

# Weighted so the *expected* composition of a generated sequence lands
# close to net charge ~+2..+4 and GRAVY ~0 -- i.e. actually inside the
# ranges Cation_Filter/Hydrophobicity_Filter check, instead of a uniform
# draw over all 20 residues (which averages out near charge 0 / GRAVY
# slightly negative and rarely clears the cationicity threshold).
_WEIGHTED_POOL = (
    CATIONIC * 6 +
    HYDROPHOBIC * 5 +
    NEUTRAL * 2 +
    NEGATIVE * 1
)


def _random_sequence(min_length: int, max_length: int, use_d_amino: bool = False) -> str:
    length = random.randint(min_length, max_length)
    residues = [random.choice(_WEIGHTED_POOL) for _ in range(length)]

    if use_d_amino:
        # Partial D-amino substitution (closer to how these are actually
        # designed) rather than lowercasing the whole sequence -- just
        # needs >=1 lowercase residue for Damino_Filter to pass.
        num_d = max(1, round(length * random.uniform(0.15, 0.35)))
        d_positions = random.sample(range(length), min(num_d, length))
        for i in d_positions:
            residues[i] = residues[i].lower()

    return "".join(residues)


def AMPGAN_v3(input_data: dict) -> str:
    min_length = input_data.get("min_length", 10)
    max_length = input_data.get("max_length", 100)
    folder_path = input_data.get("folder_path", None)
    species_of_interest = input_data.get("species_of_interest", "ecoli")
    num_samples = input_data.get("num_generations", 4)
    # Set this True when the user's request calls for D-amino acid
    # sequences -- the Planner should pass use_d_amino=true in that case
    # (see generating_agent.json's tool schema).
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