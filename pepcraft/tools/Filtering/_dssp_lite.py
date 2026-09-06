"""
Pure-Python secondary structure assignment -- a drop-in replacement for the
external `mkdssp` binary.

Implements the Kabsch & Sander (1983) DSSP algorithm directly from PDB
backbone geometry, emitting the same single-letter codes the original
pipeline expects:

    H = alpha helix (4-turn)      G = 3-10 helix (3-turn)
    I = pi helix (5-turn)         E = extended beta strand
    B = isolated beta bridge      T = hydrogen-bonded turn
    S = bend                      - = coil / none

Why this exists: Structure_Filter.py originally shelled out to `mkdssp`,
which is a separate non-Python executable. On locked-down machines (e.g.
Windows with Application Control policies) installing that binary may be
blocked outright, producing
    "Error: [WinError 2] The system cannot find the file specified"
and silently failing every structure classification. This module needs
nothing but Biopython + numpy, which the pipeline already depends on.

Accuracy note: this reproduces DSSP's core H-bond and pattern logic, not
every edge case of the reference C implementation (e.g. it does not handle
chain breaks, insertion codes, or PPII assignment specially). For short
single-chain peptides -- which is all this pipeline folds -- it should
agree with mkdssp closely. Treat it as a faithful reimplementation for
this use case, not a certified byte-identical clone.
"""
import numpy as np
from Bio.PDB import PDBParser

# Kabsch-Sander electrostatic H-bond energy constants
Q1Q2 = 0.084 * 332.0  # kcal/mol * Angstrom
HBOND_ENERGY_CUTOFF = -0.5  # kcal/mol; more negative = stronger bond
MIN_DISTANCE = 0.5  # guard against division blowups on degenerate geometry


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _extract_backbone(chain):
    """Returns a list of per-residue backbone atom dicts, skipping any
    residue missing a required backbone atom."""
    residues = []
    for res in chain:
        if not all(a in res for a in ("N", "CA", "C", "O")):
            continue
        residues.append({
            "res": res,
            "N": res["N"].get_coord().astype(float),
            "CA": res["CA"].get_coord().astype(float),
            "C": res["C"].get_coord().astype(float),
            "O": res["O"].get_coord().astype(float),
            "H": None,
        })

    # Place amide hydrogens: 1.0 A from N, along the direction opposite the
    # preceding residue's C=O vector (DSSP's standard approximation).
    for i in range(1, len(residues)):
        prev_C = residues[i - 1]["C"]
        prev_O = residues[i - 1]["O"]
        residues[i]["H"] = residues[i]["N"] + _unit(prev_C - prev_O) * 1.0

    return residues


def _hbond_energy(donor, acceptor):
    """Energy of the H-bond donated by `donor` (its N-H) to `acceptor`
    (its C=O). Returns 0.0 if the donor has no placed hydrogen."""
    if donor["H"] is None:
        return 0.0

    N, H = donor["N"], donor["H"]
    C, O = acceptor["C"], acceptor["O"]

    d_ON = max(np.linalg.norm(O - N), MIN_DISTANCE)
    d_CH = max(np.linalg.norm(C - H), MIN_DISTANCE)
    d_OH = max(np.linalg.norm(O - H), MIN_DISTANCE)
    d_CN = max(np.linalg.norm(C - N), MIN_DISTANCE)

    return Q1Q2 * (1.0 / d_ON + 1.0 / d_CH - 1.0 / d_OH - 1.0 / d_CN)


def _build_hbond_map(residues):
    """hbond[i][j] is True when residue j donates an H-bond to residue i's
    C=O (i.e. CO(i) -> NH(j))."""
    n = len(residues)
    hb = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < 2:  # DSSP ignores i,i and adjacent
                continue
            if _hbond_energy(residues[j], residues[i]) < HBOND_ENERGY_CUTOFF:
                hb[i][j] = True
    return hb


def _bend_angles(residues):
    """S (bend) where the CA(i-2)->CA(i)->CA(i+2) kink exceeds 70 degrees."""
    n = len(residues)
    bends = [False] * n
    for i in range(2, n - 2):
        v1 = residues[i]["CA"] - residues[i - 2]["CA"]
        v2 = residues[i + 2]["CA"] - residues[i]["CA"]
        cosang = np.clip(np.dot(_unit(v1), _unit(v2)), -1.0, 1.0)
        if np.degrees(np.arccos(cosang)) > 70.0:
            bends[i] = True
    return bends


def assign_secondary_structure(pdb_path, chain_id=None):
    """Parses a PDB file and returns a list of DSSP-style single-letter
    secondary structure codes, one per backbone-complete residue."""
    structure = PDBParser(QUIET=True).get_structure("peptide", pdb_path)
    model = next(structure.get_models())

    chain = None
    for c in model.get_chains():
        if chain_id is None or c.id == chain_id:
            chain = c
            break
    if chain is None:
        return []

    residues = _extract_backbone(chain)
    n = len(residues)
    if n < 4:
        return ["-"] * n

    hb = _build_hbond_map(residues)
    ss = ["-"] * n

    # --- n-turns: residue i has an n-turn if CO(i) -> NH(i+n) ---
    turns = {3: [False] * n, 4: [False] * n, 5: [False] * n}
    for nt in (3, 4, 5):
        for i in range(n - nt):
            if hb[i][i + nt]:
                turns[nt][i] = True

    # --- helices: two CONSECUTIVE n-turns define a helix ---
    # Assign 4-helix (H) first, then 3-10 (G) and pi (I) only where free,
    # matching DSSP's precedence.
    for nt, code in ((4, "H"), (3, "G"), (5, "I")):
        for i in range(n - nt - 1):
            if turns[nt][i] and turns[nt][i + 1]:
                for k in range(i + 1, min(i + nt + 1, n)):
                    if code == "H" or ss[k] == "-":
                        ss[k] = code

    # --- beta bridges ---
    # Parallel:     hb[i-1][j] and hb[j][i+1]   OR  hb[j-1][i] and hb[i][j+1]
    # Antiparallel: hb[i][j]  and hb[j][i]      OR  hb[i-1][j+1] and hb[j-1][i+1]
    bridge_partners = {i: [] for i in range(n)}
    for i in range(1, n - 1):
        for j in range(i + 3, n - 1):  # bridges need separation
            parallel = (hb[i - 1][j] and hb[j][i + 1]) or (hb[j - 1][i] and hb[i][j + 1])
            antiparallel = (hb[i][j] and hb[j][i]) or (hb[i - 1][j + 1] and hb[j - 1][i + 1])
            if parallel or antiparallel:
                bridge_partners[i].append(j)
                bridge_partners[j].append(i)

    # A bridge residue is E when it sits in a ladder (its neighbour also
    # bridges), otherwise an isolated bridge B.
    for i in range(n):
        if ss[i] in ("H", "G", "I") or not bridge_partners[i]:
            continue
        in_ladder = any(
            bridge_partners.get(i + d) and
            any(abs(pj - pi) <= 1 for pi in bridge_partners[i] for pj in bridge_partners[i + d])
            for d in (-1, 1) if 0 <= i + d < n
        )
        ss[i] = "E" if in_ladder else "B"

    # --- turns (T): any residue covered by an n-turn but not already assigned ---
    for nt in (3, 4, 5):
        for i in range(n - nt):
            if turns[nt][i]:
                for k in range(i + 1, min(i + nt, n)):
                    if ss[k] == "-":
                        ss[k] = "T"

    # --- bends (S): lowest priority ---
    for i, is_bend in enumerate(_bend_angles(residues)):
        if is_bend and ss[i] == "-":
            ss[i] = "S"

    return ss
