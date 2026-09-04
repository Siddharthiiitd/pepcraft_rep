import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_match_utils import find_best_match

# NOVELTY CHECK -- this must point at FULL SwissProt (all ~570k reviewed
# proteins of every function), NOT an antibacterial-filtered subset.
#
# The question this tool answers is "does this peptide resemble ANY known
# protein?". If you point it at an AMP-only subset, it instead answers
# "does this resemble a known AMP?" -- which is what Verify_DBAASP already
# tells you, so you'd get the same evidence twice and no novelty signal.
#
# Get it from UniProt (verify the exact current URL on uniprot.org -- I
# can't confirm it live):
#   https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true&format=tsv&fields=accession,id,protein_name,organism_name,sequence
# Save as CSV with at least "accession" and "sequence" columns.
#
# First run builds a k-mer index (~5 min at full-SwissProt scale) and caches
# it to disk next to the CSV; later runs load it in about a minute.
SWISSPROT_CSV_PATH = os.environ.get(
    "SWISSPROT_CSV_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "swissprot.csv"),
)


def Verify_SwissProt(input_data: dict) -> str:
    folder_path = input_data.get("folder_path", None)
    if folder_path:
        csv_path = os.path.join(folder_path, "generated_sequences.csv")
    else:
        raise ValueError("folder_path is required in input_data")

    df = pd.read_csv(csv_path)
    filter_columns = [col for col in df.columns if "filter" in col]

    results_text = ""
    swissprot_report = []

    for index, row in df.iterrows():
        try:
            if 'swissprot_report' in df.columns and row.get('swissprot_report') and row['swissprot_report'] != "new":
                swissprot_report.append(row['swissprot_report'])
                continue

            if all(row[key] == 1 for key in filter_columns):
                sequence = row["sequence"].upper()
                print(f"[Verify_SwissProt] local search for sequence {index+1}: {sequence}...")

                match = find_best_match(sequence, SWISSPROT_CSV_PATH, sequence_col="sequence", id_col="accession")

                if match is None:
                    entry_text = (f"{sequence}\n[1] NOVEL -- no similar protein found anywhere "
                                  f"in SwissProt.\n")
                else:
                    caveat = (" (LOW COVERAGE -- only a short sub-region aligned, "
                              "treat this hit as weak evidence)" if match.get("low_coverage") else "")
                    entry_text = (
                        f"{sequence}\n[1] Closest SwissProt protein: {match['name']} | "
                        f"Identity: {match['identity_pct']}% | Query coverage: {match['coverage_pct']}%"
                        f" | alignment score: {match['alignment_score']}{caveat}\n"
                    )

                swissprot_report.append(entry_text)
                results_text += entry_text
            else:
                swissprot_report.append("")  # didn't pass filters

        except Exception as e:
            error_msg = f"Error during local SwissProt search: {e}\n"
            swissprot_report.append(error_msg)
            results_text += error_msg

    if len(swissprot_report) == 0:
        return "No sequences passed all filters."

    df['swissprot_report'] = swissprot_report
    df.to_csv(csv_path, index=False)

    reported = [col for col in df.columns if "report" in col]
    reported_done = []
    for col in reported:
        if col in df.columns:
            check = df[col].apply(lambda x: True if (x is not None) and (x != "") and (x != "new") else False)
        if check.sum() == len(check):
            reported_done.append([col, len(check)])
    results_text += "Completed Verification :"
    for col in reported_done:
        results_text += f" {col[0]} ({col[1]} sequences completed),"

    return results_text