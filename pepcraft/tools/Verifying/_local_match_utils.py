"""
Lightweight local sequence search -- no BLAST+ binary required.

Supports two distinct uses in this pipeline:

  1. NOVELTY CHECK (full SwissProt, ~570k proteins of every function)
     "Does this peptide resemble ANY known protein?"
  2. AMP SIMILARITY CHECK (DBAASP, ~1.8k antimicrobial peptides)
     "Does this peptide resemble a known antimicrobial peptide?"

These answer different questions and should use different databases.
Searching an AMP-only subset of SwissProt collapses (1) into (2) and gives
you two copies of the same evidence.

Design notes for large databases:
  - K=5 k-mers (3.2M possible) rather than K=4 (160k). At 570k proteins a
    4-mer index stops discriminating -- nearly every protein shares 4-mers
    with any query.
  - k-mer hit counts are LENGTH-NORMALIZED so long proteins don't win the
    prefilter purely by being long.
  - The built index is CACHED TO DISK (pickle) next to the CSV, so the
    expensive build happens once ever, not once per pipeline run.
  - Alignments report query COVERAGE alongside %identity. On short peptides
    a local (Smith-Waterman) alignment can report 100% identity from a tiny
    perfectly-matching sub-region, which is misleading on its own.
"""
import os
import pickle
import hashlib
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices

_INDEX_CACHE = {}

# K is per-database, not global: a large protein database needs longer
# k-mers to discriminate, while a small database of SHORT peptides needs
# short k-mers or nothing matches at all.
DEFAULT_K = 5          # full SwissProt (~570k proteins) -- novelty check
SMALL_DB_K = 4         # DBAASP / AMP sets (~2k short peptides)
SMALL_DB_THRESHOLD = 50000   # rows below this are treated as a "small db"
TOP_N_CANDIDATES = 50
MIN_COVERAGE_PCT = 50.0  # alignments covering less of the query than this are reported but flagged


def _cache_path(csv_path, k):
    h = hashlib.md5(os.path.abspath(csv_path).encode()).hexdigest()[:8]
    return os.path.join(os.path.dirname(os.path.abspath(csv_path)), f".kmer_index_{h}_k{k}.pkl")


def _build_kmer_index(sequences, K):
    index = {}
    for row_idx, seq in enumerate(sequences):
        seq = seq.upper()
        for kmer in {seq[i:i + K] for i in range(len(seq) - K + 1)}:
            index.setdefault(kmer, []).append(row_idx)
    return index


def load_local_database(csv_path: str, sequence_col: str = "sequence", id_col: str = None):
    if csv_path in _INDEX_CACHE:
        return _INDEX_CACHE[csv_path]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Local database CSV not found: {csv_path}\n"
            f"Download it and/or point the relevant env var at the right path."
        )

    df = pd.read_csv(csv_path)
    if sequence_col not in df.columns:
        raise ValueError(f"Expected a '{sequence_col}' column in {csv_path}, found: {list(df.columns)}")

    df = df.dropna(subset=[sequence_col]).reset_index(drop=True)
    sequences = df[sequence_col].astype(str).tolist()
    lengths = [len(s) for s in sequences]

    K = SMALL_DB_K if len(sequences) < SMALL_DB_THRESHOLD else DEFAULT_K

    cache_file = _cache_path(csv_path, K)
    csv_mtime = os.path.getmtime(csv_path)
    kmer_index = None

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            if cached.get("mtime") == csv_mtime and cached.get("k") == K and cached.get("n") == len(sequences):
                kmer_index = cached["index"]
                print(f"[local_match] Loaded cached k-mer index for {os.path.basename(csv_path)} ({len(sequences)} rows).")
        except Exception:
            kmer_index = None

    if kmer_index is None:
        print(f"[local_match] Building k-mer index for {os.path.basename(csv_path)} "
              f"({len(sequences)} rows) -- ONE-TIME cost, cached to disk afterwards...")
        kmer_index = _build_kmer_index(sequences, K)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"mtime": csv_mtime, "k": K, "n": len(sequences), "index": kmer_index}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[local_match] Index cached to {os.path.basename(cache_file)} -- future runs load instantly.")
        except Exception as e:
            print(f"[local_match] (could not write index cache: {e})")

    entry = {"df": df, "sequences": sequences, "lengths": lengths,
             "kmer_index": kmer_index, "id_col": id_col, "k": K}
    _INDEX_CACHE[csv_path] = entry
    return entry


def _prefilter_candidates(query, kmer_index, lengths, top_n, K):
    query = query.upper()
    counts = {}
    for kmer in {query[i:i + K] for i in range(len(query) - K + 1)}:
        for row_idx in kmer_index.get(kmer, []):
            counts[row_idx] = counts.get(row_idx, 0) + 1

    if not counts:
        return []

    # Length-normalize: a 600aa protein shouldn't outrank a real homolog
    # just because its size makes incidental k-mer collisions likely.
    scored = [(idx, hits / (lengths[idx] ** 0.5)) for idx, hits in counts.items()]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [idx for idx, _ in scored[:top_n]]


def find_best_match(query_sequence: str, csv_path: str, sequence_col: str = "sequence", id_col: str = None):
    """Best local-alignment match for query_sequence in the CSV database.
    Returns dict with name, identity_pct, coverage_pct, alignment_score,
    matched_sequence, row -- or None if nothing shares any k-mer."""
    db = load_local_database(csv_path, sequence_col=sequence_col, id_col=id_col)
    query = query_sequence.upper()

    candidates = _prefilter_candidates(query, db["kmer_index"], db["lengths"], TOP_N_CANDIDATES, db["k"])
    if not candidates:
        return None

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    best = None
    for row_idx in candidates:
        target = db["sequences"][row_idx].upper()
        try:
            alignment = aligner.align(query, target)[0]
        except Exception:
            continue

        aligned_query, aligned_target = str(alignment[0]), str(alignment[1])
        matches = sum(1 for a, b in zip(aligned_query, aligned_target) if a == b and a != "-")
        aligned_len = max(len(aligned_query.replace("-", "")), 1)
        identity_pct = round((matches / max(len(aligned_query), 1)) * 100, 2)
        coverage_pct = round((aligned_len / max(len(query), 1)) * 100, 2)

        if best is None or alignment.score > best["_raw"]:
            row = db["df"].iloc[row_idx]
            name = row[db["id_col"]] if db["id_col"] and db["id_col"] in row else f"row_{row_idx}"
            best = {
                "name": str(name),
                "identity_pct": identity_pct,
                "coverage_pct": coverage_pct,
                "low_coverage": coverage_pct < MIN_COVERAGE_PCT,
                "alignment_score": alignment.score,
                "matched_sequence": target,
                "row": row.to_dict(),
                "_raw": alignment.score,
            }

    if best:
        best.pop("_raw", None)
    return best