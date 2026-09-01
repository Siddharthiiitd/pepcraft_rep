import os
import time
import pandas as pd
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import Entrez, SeqIO

# NCBI requires an email for both Entrez and remote BLAST usage.
# Set this to YOUR OWN address -- the original repo hardcoded the paper
# authors' email, which you should not keep using.
#   export NCBI_EMAIL="you@example.com"
NCBI_EMAIL = os.environ.get("NCBI_EMAIL")
if not NCBI_EMAIL:
    raise RuntimeError("Set NCBI_EMAIL to your own email address before running -- NCBI requires it.")
Entrez.email = NCBI_EMAIL

# Simple in-memory cache so re-querying the same sequence within one run
# (e.g. across retries) doesn't hit NCBI's rate limits twice.
_SWISSPROT_CACHE = {}


def get_taxonomy_and_functional_text(accession_id):
    try:
        handle = Entrez.efetch(db="protein", id=accession_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()

        taxonomy = record.annotations.get("taxonomy", [])
        description = record.description

        products = []
        for feature in record.features:
            if "product" in feature.qualifiers:
                products.extend(feature.qualifiers["product"])
        products = list(set(products))

        taxonomy_str = "--- Taxonomy & Functional Info ---"
        taxonomy_str += f"\nTaxonomy: {' -> '.join(taxonomy)}"
        taxonomy_str += f"\nDescription: {description}"
        taxonomy_str += f"\nProducts: {', '.join(products)}"
        return taxonomy_str
    except Exception:
        return "No data available due to error."


def _remote_blast_against_swissprot(sequence, max_retries=3):
    """Runs a single sequence through NCBI's remote BLAST server against
    the swissprot database. No local db/index required."""
    if sequence in _SWISSPROT_CACHE:
        return _SWISSPROT_CACHE[sequence]

    for attempt in range(1, max_retries + 1):
        try:
            result_handle = NCBIWWW.qblast(
                program="blastp",
                database="swissprot",
                sequence=sequence,
                hitlist_size=5,
                expect=10.0,
            )
            blast_record = NCBIXML.read(result_handle)
            result_handle.close()

            if not blast_record.alignments:
                _SWISSPROT_CACHE[sequence] = None
                return None

            top_alignment = blast_record.alignments[0]
            top_hsp = top_alignment.hsps[0]
            identity_pct = (top_hsp.identities / top_hsp.align_length) * 100

            hit_info = {
                "title": top_alignment.title,
                "accession": top_alignment.accession,
                "length": top_alignment.length,
                "e_value": top_hsp.expect,
                "identity_pct": round(identity_pct, 2),
                "alignment_query": top_hsp.query,
                "alignment_match": top_hsp.match,
                "alignment_subject": top_hsp.sbjct,
            }
            _SWISSPROT_CACHE[sequence] = hit_info
            return hit_info

        except Exception as e:
            print(f"[Verify_SwissProt] remote BLAST attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(5 * attempt)

    return None


def Verify_SwissProt(input_data: dict) -> str:
    folder_path = input_data.get("folder_path", None)
    if folder_path:
        csv_path = os.path.join(folder_path, "generated_sequences.csv")
    else:
        raise ValueError("folder_path is required in input_data")

    df = pd.read_csv(csv_path)
    filter_columns = [col for col in df.columns if "filter" in col]

    BLAST_results = ""
    swissprot_report = []

    for index, row in df.iterrows():
        current_row_text = ""

        try:
            if 'swissprot_report' in df.columns and row.get('swissprot_report') and row['swissprot_report'] != "new":
                swissprot_report.append(row['swissprot_report'])
                continue

            if all(row[key] == 1 for key in filter_columns):
                sequence = row["sequence"].upper()
                print(f"[Verify_SwissProt] querying NCBI for sequence {index+1}: {sequence} (this can take 30s-2min)...")
                current_row_text += f"{sequence}\n"

                hit_info = _remote_blast_against_swissprot(sequence)

                if hit_info is None:
                    current_row_text += "[1] No matches found.\n"
                else:
                    current_row_text += (
                        f"[1] Top Hit: {hit_info['accession']} | "
                        f"Identity: {hit_info['identity_pct']}% | e_value: {hit_info['e_value']}\n"
                    )
                    taxonomy_info = get_taxonomy_and_functional_text(hit_info['accession'])
                    current_row_text += f"\n{taxonomy_info}\n\n"

                swissprot_report.append(current_row_text)
                BLAST_results += current_row_text

                # be polite to NCBI's shared servers between queries
                time.sleep(2)
            else:
                swissprot_report.append("")  # didn't pass filters

        except Exception as e:
            error_msg = f"No hits or error during BLAST search: {e}\n"
            swissprot_report.append(error_msg)
            BLAST_results += error_msg

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
    BLAST_results += "Completed Verification :"
    for col in reported_done:
        BLAST_results += f" {col[0]} ({col[1]} sequences completed),"

    return BLAST_results