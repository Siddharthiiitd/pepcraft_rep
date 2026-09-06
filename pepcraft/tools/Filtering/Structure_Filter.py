import os
import sys
import pandas as pd
import json
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dssp_lite import assign_secondary_structure

# The original pipeline shelled out to the external `mkdssp` binary here.
# That binary is often unavailable (and may be blocked outright by
# corporate Application Control policies on Windows), which silently
# failed EVERY structure classification with
#   "Error: [WinError 2] The system cannot find the file specified"
# and therefore failed every candidate at the structure filter.
#
# We now compute secondary structure in pure Python (_dssp_lite, a direct
# Kabsch-Sander implementation) -- no external executable needed. Set
# USE_EXTERNAL_MKDSSP=1 in the environment to prefer the real mkdssp
# binary instead, falling back to the pure-Python path if it is missing.
USE_EXTERNAL_MKDSSP = os.environ.get("USE_EXTERNAL_MKDSSP", "0") == "1"


def _get_ss_codes(model, structure_path):
    """Returns a list of DSSP single-letter codes for the structure."""
    if USE_EXTERNAL_MKDSSP:
        try:
            from Bio.PDB.DSSP import DSSP
            dssp = DSSP(model, structure_path, dssp="mkdssp")
            return [dssp[key][2] for key in dssp.keys()]
        except Exception as e:
            print(f"[Structure_Filter] mkdssp unavailable ({e}); using built-in pure-Python DSSP.")
    return assign_secondary_structure(structure_path)

map_dssp_to_3class = {1: "Mixed (Alpha/Beta)", 2: "Alpha-Helical", 3: "Beta-Hairpin / Turn-Rich Beta", 4: "Extended Beta-Strand", 5: "Structured Turns / Bends", 6: "Unstructured / Random Coil"}

def classify_peptide(counts, total_residues):
    # 1. Group the DSSP raw counts into biological categories
    # Helices (Alpha, 3-10, Pi)
    helix_count = counts.get('H', 0) + counts.get('G', 0) + counts.get('I', 0)
    # Beta structures (Extended strand, isolated bridge)
    beta_count  = counts.get('E', 0) + counts.get('B', 0)
    # Structured turns and bends
    turn_count  = counts.get('T', 0) + counts.get('S', 0)
    
    # Calculate percentages
    helix_pct = (helix_count / total_residues) * 100
    beta_pct  = (beta_count / total_residues) * 100
    turn_pct  = (turn_count / total_residues) * 100

    # 2. The 6-Class Decision Tree
    
    # Class 1: Mixed (Significant amounts of both helix and sheet)
    if helix_pct >= 15.0 and beta_pct >= 15.0:
        return 1
        
    # Class 2: Alpha-Helical (Dominant helix)
    elif helix_pct >= 40.0:
        return 2
        
    # Class 3: Beta-Hairpin (Beta strands connected by structured turns)
    elif beta_pct >= 20.0 and turn_pct >= 10.0:
        return 3
        
    # Class 4: Extended Beta-Strand (Beta sheet without many turns)
    elif beta_pct >= 30.0:
        return 4
        
    # Class 5: Structured Turns (Not helical or sheet, but heavily folded)
    elif turn_pct >= 30.0:
        return 5
        
    # Class 6: Unstructured (Mostly '-' codes, meaning random coil)
    else:
        return 6

def Structure_Filter(input_data: dict) -> str:
    folder_path = input_data.get("folder_path", None)
    if folder_path:
        csv_path = os.path.join(folder_path, "generated_sequences.csv")
    else:
        raise ValueError("folder_path is required in input_data")
    structure = input_data.get("structure", None)
    df = pd.read_csv(csv_path)
    if 'predicted_structure_path' not in df.columns:
        return "Error: 'predicted_structure_path' column not found in CSV. Please run the structure prediction tool to generate structural data before applying the structure filter."
    elif "new" in df['predicted_structure_path'].values:
        return "Error: new sequences detected without predicted structures. Please run the structure prediction tool to generate structural data for the new sequences before applying the structure filter."
    # remove structure_filter if it exists
    if 'structure_filter' in df.columns:
        df.drop(columns=['structure_filter'], inplace=True)    
    keys = []
    for key in df.keys():
        if "filter" in key:
            keys.append(key)

    labels = []
    for index, row in df.iterrows():
        if all(row[key] == 1 for key in keys):
            structure_path = row.get("predicted_structure_path", None)
            if structure_path:
                try:
                    parser_pdb = PDBParser(QUIET=True)
                    pdb_structure = parser_pdb.get_structure("peptide", structure_path)
                    model = pdb_structure[0]

                    # Ensure Chain ID is not blank (DSSP hates blank chains)
                    for chain in model.get_chains():
                        if chain.id == " " or chain.id == "":
                            chain.id = "A"

                    all_sec_structures = _get_ss_codes(model, structure_path)
                    total_residues = len(all_sec_structures)
                    if total_residues == 0:
                        df.at[index, 'structure_classification'] = "Error: no backbone residues parsed from structure"
                        continue
                    counts = Counter(all_sec_structures)

                    classification_label = classify_peptide(counts, total_residues)
                    df.at[index, 'structure_classification'] = map_dssp_to_3class[classification_label]
                    labels.append(classification_label)
                except Exception as e:
                    df.at[index, 'structure_classification'] = f"Error: {e}"
            else:
                df.at[index, 'structure_classification'] = "No valid structure path"
        else:
            df.at[index, 'structure_classification'] = "Did not pass filters"
    if structure:
        structure = [int(s) for s in structure.split(";")]
        df['structure_filter'] = df['structure_classification'].apply(
            lambda x: 1 if x in map_dssp_to_3class.values() and 
            (list(map_dssp_to_3class.values()).index(x) + 1) in structure else 0
        )
    df.to_csv(csv_path, index=False)

    # report each class distribution
    class_counts = Counter(labels)
    report = f"Structural classification completed for {len(labels)} sequences. The classifications are saved in the 'structure_classification' column of the original CSV file."
    class_report = "\n".join([f"Class {cls} ({map_dssp_to_3class[cls]}): {count} sequences" for cls, count in class_counts.items()])
    if structure:
        class_report += f"\nThe structure filter was applied with the following classes: {', '.join([map_dssp_to_3class[int(s)] for s in structure])}. \n"
        # number of sequences that passed the structure filter
        passed_filter_count = df['structure_filter'].sum()
        class_report += f"Among the {len(df)} generated sequences, {passed_filter_count} sequences passed the structure filter."
    
    reported = [col for col in df.columns if "filter" in col]
    reported_done = []
    for col in reported:
        if col in df.columns:
            check = df[col].apply(lambda x: True if (x is not None) and (x != "") and (x!="new") else False)
        if check.sum() == len(check):
            reported_done.append(col)
    filter_text = f"Completed Filtering : {', '.join(reported_done)}"
    
    return report + "\n\n" + class_report + "\n" + filter_text