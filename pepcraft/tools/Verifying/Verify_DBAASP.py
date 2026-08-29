import os
import time
import json
import logging
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# DBAASP requires an API key for programmatic access. Set this as an
# environment variable, never hardcode it in the file:
#   export DBAASP_API_KEY="your-key-here"
#
# NOTE: verify the exact header name DBAASP expects against their current
# API docs (dbaasp.org/docs or similar) before relying on this in
# production -- I could not browse their live docs to confirm it, so this
# is written to be a one-line change if the header name differs.
DBAASP_API_KEY = os.environ.get("DBAASP_API_KEY")
DBAASP_BASE = "https://dbaasp.org"


def safe_float(value):
    if value is None:
        return float('inf')
    try:
        clean_val = ''.join(c for c in str(value) if c.isdigit() or c == '.')
        return float(clean_val) if clean_val else float('inf')
    except ValueError:
        return float('inf')


def _headers():
    headers = {"Accept": "application/json"}
    if DBAASP_API_KEY:
        headers["Authorization"] = f"Bearer {DBAASP_API_KEY}"
    return headers


def fetch_dbaasp(url, max_retries=3, backoff=2.0):
    """GET against the DBAASP API with retries/backoff (handles rate limiting)."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=_headers(), timeout=20)
            if response.status_code == 429:
                # rate limited -- wait and retry
                wait = backoff * attempt
                logging.warning(f"Rate limited by DBAASP, retrying in {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            logging.error(f"API Error at {url} (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff * attempt)
    return None


def parse_llm_ready_summary(data):
    """Extracts high-value fields from a DBAASP peptide JSON record safely."""
    if not isinstance(data, dict):
        return {"error": "Invalid data format"}

    summary = {
        "name": data.get("name") or "Unknown",
        "sequence": data.get("sequence") or "N/A",
        "charge": "Unknown",
        "hydrophobic_moment": "Unknown",
        "efficacy": [],
        "toxicity": [],
    }

    for prop in data.get("physicoChemicalProperties") or []:
        if not isinstance(prop, dict):
            continue
        name = prop.get("name")
        if name == "Net Charge":
            summary["charge"] = prop.get("value")
        elif name == "Normalized Hydrophobic Moment":
            summary["hydrophobic_moment"] = prop.get("value")

    effs = []
    for act in data.get("targetActivities") or []:
        if not isinstance(act, dict):
            continue
        mic = act.get("concentration")
        species_dict = act.get("targetSpecies") or {}
        species = species_dict.get("name")
        measure_dict = act.get("activityMeasureGroup") or {}
        measure = str(measure_dict.get("name", "")).upper()
        unit_dict = act.get("unit") or {}
        unit = unit_dict.get("name", "")

        if species and mic and "MIC" in measure:
            effs.append({"pathogen": species, "mic": mic, "unit": unit})

    effs.sort(key=lambda x: safe_float(x["mic"]))
    summary["efficacy"] = effs[:2]

    for tox in data.get("hemoliticCytotoxicActivities") or []:
        if not isinstance(tox, dict):
            continue
        cell_dict = tox.get("targetCell") or {}
        cell = cell_dict.get("name")
        unit_dict = tox.get("unit") or {}
        if cell and tox.get("concentration"):
            summary["toxicity"].append({
                "cell_type": cell,
                "concentration": tox.get("concentration"),
                "unit": unit_dict.get("name", ""),
                "effect": tox.get("activityMeasureForLysisValue", "Unknown"),
            })
            break

    return summary


def lookup_sequence_in_dbaasp(sequence):
    """
    Direct online lookup, no local BLAST db / id-map required.

    Since we no longer BLAST against a local mirror of DBAASP, we can't
    produce a %identity score against the *closest* known peptide the way
    the original pipeline did. Instead this does an exact-sequence lookup:
    if AMPGAN generated something that is already a known, catalogued AMP,
    it shows up here; anything not found is reported as novel relative to
    DBAASP. That's a real trade-off vs the original homology search --
    call it out in your writeup/thesis as a limitation of the lightweight
    reproduction.
    """
    search = fetch_dbaasp(f"{DBAASP_BASE}/peptides?sequence.value={sequence}")

    if not isinstance(search, dict) or not search.get("data"):
        return {"status": "novel", "summary": None}

    first_item = search["data"][0]
    if not isinstance(first_item, dict):
        return {"status": "novel", "summary": None}

    peptide_id = first_item.get("dbaaspId")
    if peptide_id is None:
        return {"status": "novel", "summary": None}

    details = fetch_dbaasp(f"{DBAASP_BASE}/peptides/{peptide_id}")
    if not details:
        return {"status": "match_no_details", "summary": None}

    return {"status": "match", "summary": parse_llm_ready_summary(details)}


def Verify_DBAASP(input_data: dict) -> str:
    folder_path = input_data.get("folder_path")
    if not folder_path or not os.path.exists(folder_path):
        raise ValueError("Valid folder_path is required")
    if not DBAASP_API_KEY:
        logging.warning(
            "DBAASP_API_KEY is not set -- requests will be sent unauthenticated "
            "and may be rate-limited or rejected by the DBAASP API."
        )

    csv_path = os.path.join(folder_path, "generated_sequences.csv")
    df = pd.read_csv(csv_path)

    filter_cols = [c for c in df.columns if "filter" in c]
    full_report_log = []
    dbaasp_results_col = []

    for _, row in df.iterrows():
        if not all(row.get(col) == 1 for col in filter_cols):
            dbaasp_results_col.append("")
            continue

        # already verified in a previous run -- skip re-querying the API
        if 'dbaasp_report' in df.columns and row.get('dbaasp_report') and row['dbaasp_report'] != "new":
            dbaasp_results_col.append(row['dbaasp_report'])
            continue

        sequence = row["sequence"]
        result = lookup_sequence_in_dbaasp(sequence)

        if result["status"] == "match":
            report_data = json.dumps(result["summary"])
            log_entry = f"Seq: {sequence} | Found in DBAASP (exact match) | id: {result['summary'].get('name')}"
        elif result["status"] == "match_no_details":
            report_data = "Matched a DBAASP entry but details could not be retrieved."
            log_entry = f"Seq: {sequence} | DBAASP match found, details fetch failed."
        else:
            report_data = json.dumps({"status": "novel", "note": "No exact sequence match found in DBAASP."})
            log_entry = f"Seq: {sequence} | No match in DBAASP (novel)."

        full_report_log.append(log_entry)
        dbaasp_results_col.append(report_data)

        # be polite to the API -- avoid hammering it row by row
        time.sleep(0.3)

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