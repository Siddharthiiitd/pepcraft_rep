"""
Standalone test for Verify_SwissProt -- run this directly, no LLM calls,
no quota spent. Just exercises the remote NCBI BLAST path against your
existing filtered candidates.

Usage (from the pepcraft/ folder):
    python test_swissprot.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools", "Verifying"))

# Load .env the same way Initialization.py does (no python-dotenv needed)
def _load_dotenv(env_path=".env"):
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

if "NCBI_EMAIL" not in os.environ:
    raise RuntimeError("Set NCBI_EMAIL in your .env before running this test.")

from Verify_SwissProt import Verify_SwissProt

# Point this at whatever folder your generated_sequences.csv actually lives in
FOLDER_PATH = os.path.join(os.path.dirname(__file__), "output")

print(f"Testing Verify_SwissProt against: {FOLDER_PATH}")
print("This will make real remote calls to NCBI's BLAST server -- expect")
print("roughly 30s-2min PER sequence, this is normal, not a hang.\n")

start = time.time()
result = Verify_SwissProt({"folder_path": FOLDER_PATH})
elapsed = time.time() - start

print(f"\n--- Done in {elapsed:.1f}s ---\n")
print(result)