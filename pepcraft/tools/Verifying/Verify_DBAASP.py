import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_match_utils import find_best_match

# Point this at your locally downloaded DBAASP CSV. Get the bulk export
# from DBAASP's own download page on dbaasp.org (I can't confirm the exact
# current download URL live -- check their site directly). Save/convert it
# to CSV with at least a "sequence" column and ideally a "name"/"id" column.
DBAASP_CSV_PATH = os.environ.get(
    "DBAASP_CSV_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "dbaasp.csv"),
)


def Verify_DBAASP(input_data: dict) -> str:
    folder_path = input_data.get("folder_path")
    if not folder_path or not os.path.exists(folder_path):
        raise ValueError("Valid folder_path is required")

    csv_path = os.path.join(folder_path, "generated_sequences.csv")
    df = pd.read_csv(csv_path)

    filter_cols = [c for c in df.columns if "filter" in c]
    full_report_log = []
    dbaasp_results_col = []

    for index, row in df.iterrows():
        if not all(row.get(col) == 1 for col in filter_cols):
            dbaasp_results_col.append("")
            continue

        if 'dbaasp_report' in df.columns and row.get('dbaasp_report') and row['dbaasp_report'] != "new":
            dbaasp_results_col.append(row['dbaasp_report'])
            continue

        sequence = row["sequence"].upper()
        print(f"[Verify_DBAASP] local search for sequence {index+1}: {sequence}...")

        match = find_best_match(sequence, DBAASP_CSV_PATH, sequence_col="sequence", id_col="name")

        if match is None:
            report_data = "No match in DBAASP -- not similar to any known antimicrobial peptide."
            log_entry = f"Seq: {sequence} | No match in local DBAASP."
        else:
            caveat = (" (LOW COVERAGE -- only a short sub-region aligned, weak evidence)"
                      if match.get("low_coverage") else "")
            report_data = (
                f"Closest known AMP in DBAASP: {match['name']} | "
                f"Identity: {match['identity_pct']}% | Query coverage: {match['coverage_pct']}%"
                f" | alignment score: {match['alignment_score']}{caveat}"
            )
            log_entry = (f"Seq: {sequence} | Closest DBAASP AMP: {match['name']} "
                         f"({match['identity_pct']}% id, {match['coverage_pct']}% cov)")

        full_report_log.append(log_entry)
        dbaasp_results_col.append(report_data)

    if len(full_report_log) == 0:
        return "No sequences passed all filters."

    df['dbaasp_report'] = dbaasp_results_col
    df.to_csv(csv_path, index=False)

    reported = [col for col in df.columns if "report" in col]
    reported_done = []
    for col in reported:
        if col in df.columns:
            check = df[col].apply(lambda x: True if (x is not None) and (x != "") and (x != "new") else False)
        if check.sum() == len(check):
            reported_done.append([col, len(check)])
    report = "Completed Verification :"
    for col in reported_done:
        report += f" {col[0]} ({col[1]} sequences completed),"

    full_report_log.append(report)
    return "\n".join(full_report_log)