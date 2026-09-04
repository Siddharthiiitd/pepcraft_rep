"""
Lightweight, pure-Python local sequence search -- no BLAST+ install needed.

Used by both Verify_SwissProt.py and Verify_DBAASP.py to search a locally
downloaded CSV of known sequences (SwissProt or DBAASP) for the closest
match to a query peptide.

How it works (two-stage, mimics what BLAST does internally at a much
smaller scale):
  1. K-MER PREFILTER -- build an index once (cached in memory) mapping each
     short substring ("k-mer") to which rows contain it. For a query, count
     shared k-mers against every row almost instantly (set operations), and
     keep only the top N candidates. This avoids running a slow alignment
     against every single row in a large database.
  2. LOCAL ALIGNMENT REFINE -- only the top N candidates from step 1 get a
     real Biopython local alignment (Smith-Waterman-style), from which we
     compute a genuine %identity score.

This is NOT as fast or as rigorous as real BLAST, but for a database in the
thousands-to-hundreds-of-thousands-of-rows range and short query peptides
(10-20 residues), it's tractable on CPU with no external binary.
"""
import os
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices

# Cache: csv_path -> {"rows": [...], "kmer_index": {...}}
# Built once per process, reused across every query in a run.
_INDEX_CACHE = {}

K = 4  # k-mer length for the prefilter stage
TOP_N_CANDIDATES = 25  # how many k-mer-similar rows get a real alignment


def _build_kmer_index(sequences):
    index = {}
    for row_idx, seq in enumerate(sequences):
        seq = seq.upper()
        seen = set()
        for i in range(len(seq) - K + 1):
            kmer = seq[i:i + K]
            if kmer in seen:
                continue
            seen.add(kmer)
            index.setdefault(kmer, []).append(row_idx)
    return index


def load_local_database(csv_path: str, sequence_col: str = "sequence", id_col: str = None):
    """Loads (and caches) a local CSV database + its k-mer index.
    id_col: which column to treat as the display name/accession -- if None,
    uses the row index."""
    if csv_path in _INDEX_CACHE:
        return _INDEX_CACHE[csv_path]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Local database CSV not found: {csv_path}\n"
            f"Download it first and point the relevant env var at this path."
        )

    df = pd.read_csv(csv_path)
    if sequence_col not in df.columns:
        raise ValueError(
            f"Expected a '{sequence_col}' column in {csv_path}, found: {list(df.columns)}"
        )

    df = df.dropna(subset=[sequence_col]).reset_index(drop=True)
    sequences = df[sequence_col].astype(str).tolist()

    print(f"[local_match] Building k-mer index for {csv_path} ({len(sequences)} rows) -- one-time cost for this run...")
    kmer_index = _build_kmer_index(sequences)

    entry = {"df": df, "sequences": sequences, "kmer_index": kmer_index, "id_col": id_col}
    _INDEX_CACHE[csv_path] = entry
    return entry


def _prefilter_candidates(query: str, kmer_index: dict, num_sequences: int, top_n: int):
    query = query.upper()
    scores = {}
    for i in range(len(query) - K + 1):
        kmer = query[i:i + K]
        for row_idx in kmer_index.get(kmer, []):
            scores[row_idx] = scores.get(row_idx, 0) + 1

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [row_idx for row_idx, _ in ranked[:top_n]]


def find_best_match(query_sequence: str, csv_path: str, sequence_col: str = "sequence", id_col: str = None):
    """Returns the single best local-alignment match for query_sequence
    against the CSV database at csv_path, or None if the database is empty
    or nothing shares any k-mer with the query at all."""
    db = load_local_database(csv_path, sequence_col=sequence_col, id_col=id_col)
    query = query_sequence.upper()

    candidate_indices = _prefilter_candidates(query, db["kmer_index"], len(db["sequences"]), TOP_N_CANDIDATES)
    if not candidate_indices:
        return None

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    best = None
    for row_idx in candidate_indices:
        target = db["sequences"][row_idx].upper()
        try:
            alignment = aligner.align(query, target)[0]
        except Exception:
            continue

        aligned_query, aligned_target = str(alignment[0]), str(alignment[1])
        matches = sum(1 for a, b in zip(aligned_query, aligned_target) if a == b and a != "-")
        aligned_len = max(len(aligned_query), 1)
        identity_pct = round((matches / aligned_len) * 100, 2)

        if best is None or alignment.score > best["_raw_score"]:
            row = db["df"].iloc[row_idx]
            name = row[db["id_col"]] if db["id_col"] and db["id_col"] in row else f"row_{row_idx}"
            best = {
                "name": str(name),
                "identity_pct": identity_pct,
                "alignment_score": alignment.score,
                "matched_sequence": target,
                "row": row.to_dict(),
                "_raw_score": alignment.score,
            }

    if best:
        best.pop("_raw_score", None)
    return best
