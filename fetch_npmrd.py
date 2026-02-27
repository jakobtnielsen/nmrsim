"""fetch_npmrd.py — Download NP-MRD compounds with experimental HSQC assignments.

Strategy:
  1. Download assignment_tables.zip (experimental NMR shift assignments, ~454 KB)
  2. Find all CHSQC tables (H-C correlation data) → experimental HSQC peak lists
  3. Download SMILES CSVs for the relevant NP ID ranges
  4. Filter to molecules with 8–20 carbons and standard organic atoms
  5. Select 20 compounds with diverse carbon-count spread
  6. Save to data/npmrd_testset.json with embedded experimental hsqc_peaks

Each testset entry contains both the SMILES and experimental HSQC peaks from NP-MRD,
enabling direct comparison with Zakodium-predicted HSQC (see benchmark_npmrd.py).

Usage:
    python fetch_npmrd.py [--output data/npmrd_testset.json] [--count 20]
                          [--force-refresh]
"""

import argparse
import csv
import gzip
import json
import os
import random
import re
import time
import zipfile
import io
from collections import defaultdict

import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NPMRD_BASE = "https://np-mrd.org"

ASSIGNMENT_TABLES_URL = (
    f"{NPMRD_BASE}/system/downloads/current/assignment_tables.zip"
)
# SMILES CSV ranges (compressed) — we download all needed ranges
_SMILES_RANGES = [
    "NP0000001_NP0050000",
    "NP0050001_NP0100000",
    "NP0100001_NP0150000",
    "NP0150001_NP0200000",
    "NP0200001_NP0250000",
    "NP0250001_NP0300000",
    "NP0300001_NP0350000",
]

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_DIR, "data")
TESTSET_PATH = os.path.join(DATA_DIR, "npmrd_testset.json")

TARGET_COUNT = 20
MIN_C, MAX_C = 8, 20

# Atomic numbers allowed (H, C, N, O, F, S, Cl, Br, I)
ALLOWED_ATOMS = {1, 6, 7, 8, 9, 16, 17, 35, 53}

_RANDOM_SEED = 42

# Browser-like header (NP-MRD returns 403 for plain requests)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
        "Gecko/20100101 Firefox/120.0"
    )
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NpmrdError(Exception):
    """Base exception for fetch_npmrd errors."""


class DownloadError(NpmrdError):
    """HTTP download failed."""


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_bytes(url: str, description: str, retries: int = 3) -> bytes:
    """Download URL content with retries, returning raw bytes."""
    last_exc: Exception = RuntimeError("no attempt")
    for attempt in range(retries):
        try:
            print(f"  Downloading {description} ...", end="", flush=True)
            r = requests.get(url, headers=_HEADERS, timeout=180)
            r.raise_for_status()
            size_kb = len(r.content) / 1024
            print(f" {size_kb:.0f} KB")
            time.sleep(1)
            return r.content
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            print(f" FAILED (attempt {attempt + 1}/{retries}): {exc}")
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise DownloadError(f"Failed to download {description}: {last_exc}")


def _load_smiles_csv_gz(content: bytes) -> dict:
    """
    Parse a gzip-compressed NP-MRD SMILES CSV.

    Columns: Natural_Products_Name, NP_MRD_ID, SMILES
    Returns {NP_MRD_ID: SMILES}.
    """
    with gzip.open(io.BytesIO(content), "rt") as fh:
        reader = csv.DictReader(fh)
        return {row["NP_MRD_ID"]: row["SMILES"] for row in reader}


# ---------------------------------------------------------------------------
# CHSQC table parsing
# ---------------------------------------------------------------------------

def _parse_chsqc_table(content: str) -> list:
    """
    Parse a NP-MRD CHSQC assignment table and return unique HSQC peaks.

    Format (one row per H-C pair, may have duplicate peaks for equivalent Hs):
        H,{h_atom_num},C,{c_atom_num},{h_ppm},{c_ppm}

    Returns list of {'h_ppm': float, 'c_ppm': float}, deduplicated.
    """
    seen: set = set()
    peaks: list = []
    for line in content.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        if parts[0] != "H" or parts[2] != "C":
            continue
        try:
            h_ppm = round(float(parts[4]), 4)
            c_ppm = round(float(parts[5]), 4)
        except ValueError:
            continue
        key = (h_ppm, c_ppm)
        if key not in seen:
            seen.add(key)
            peaks.append({"h_ppm": h_ppm, "c_ppm": c_ppm})
    return peaks


def _extract_chsqc_candidates(zf: zipfile.ZipFile, smiles_map: dict) -> list:
    """
    For each CHSQC table in the assignment ZIP, extract experimental HSQC peaks
    and pair with SMILES. Returns list of candidate dicts for compounds with:
      - SMILES available
      - 8–20 carbons, allowed atoms only
      - ≥ 2 experimental HSQC peaks

    When a compound has multiple CHSQC tables, keeps the one with most peaks.
    """
    all_names = zf.namelist()
    chsqc_names = [n for n in all_names if "chsqc" in n.lower()]

    # Group by NP ID, keep best (most peaks) table per compound
    best: dict = {}  # np_id -> {peaks, n_c, smiles, table}

    for name in chsqc_names:
        m = re.search(r"(NP\d+)", name)
        if not m:
            continue
        np_id = m.group(1)
        smi = smiles_map.get(np_id)
        if not smi:
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        atom_set = {a.GetAtomicNum() for a in mol.GetAtoms()}
        if not atom_set.issubset(ALLOWED_ATOMS):
            continue
        n_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        if not (MIN_C <= n_c <= MAX_C):
            continue

        content = zf.read(name).decode("utf-8", errors="replace")
        peaks = _parse_chsqc_table(content)
        if len(peaks) < 2:
            continue

        if np_id not in best or len(peaks) > len(best[np_id]["peaks"]):
            canon_smi = Chem.MolToSmiles(mol)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            mw = round(rdMolDescriptors.CalcExactMolWt(mol), 3)
            best[np_id] = {
                "np_mrd_id": np_id,
                "smiles": canon_smi,
                "molecular_formula": formula,
                "n_carbons": n_c,
                "mw": mw,
                "n_exp_peaks": len(peaks),
                "hsqc_peaks": peaks,
                "chsqc_table": name,
            }

    candidates = list(best.values())
    print(
        f"  Found {len(candidates)} compounds with experimental CHSQC data "
        f"in {MIN_C}–{MAX_C}C range."
    )
    return candidates


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_testset(
    candidates: list,
    target: int = TARGET_COUNT,
    seed: int = _RANDOM_SEED,
) -> list:
    """
    Select `target` compounds with diversity in carbon count.

    Groups into four C-count buckets (8–10, 11–13, 14–16, 17–20),
    picks ~target//4 from each. Within each bucket, shuffles with fixed seed
    for reproducibility. Tops up from the full pool if any bucket is short.
    """
    rng = random.Random(seed)
    buckets: dict[int, list] = {0: [], 1: [], 2: [], 3: []}

    def _bucket(n_c: int) -> int:
        if n_c <= 10: return 0
        if n_c <= 13: return 1
        if n_c <= 16: return 2
        return 3

    for c in candidates:
        buckets[_bucket(c["n_carbons"])].append(c)

    for b in buckets:
        rng.shuffle(buckets[b])

    selected = []
    per_bucket = max(1, target // 4)
    for b in range(4):
        selected.extend(buckets[b][:per_bucket])

    seen = {c["np_mrd_id"] for c in selected}
    all_shuffled = list(candidates)
    rng.shuffle(all_shuffled)
    for c in all_shuffled:
        if len(selected) >= target:
            break
        if c["np_mrd_id"] not in seen:
            selected.append(c)
            seen.add(c["np_mrd_id"])

    return selected[:target]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def fetch_testset(
    output_path: str = TESTSET_PATH,
    target_count: int = TARGET_COUNT,
    force_refresh: bool = False,
) -> list:
    """
    Download NP-MRD assignment tables + SMILES, build testset with experimental
    HSQC peaks, and save to output_path.

    Returns list of testset compound dicts. If output_path exists and
    force_refresh is False, loads and returns the existing file.

    Each entry has:
      np_mrd_id, smiles, molecular_formula, n_carbons, mw,
      n_exp_peaks, hsqc_peaks (experimental H-C correlations from NP-MRD)
    """
    if not force_refresh and os.path.exists(output_path):
        print(f"Loading existing testset from {output_path}")
        with open(output_path) as fh:
            return json.load(fh)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Assignment tables
    print("\n[Step 1] Downloading NP-MRD assignment tables ...")
    asgn_bytes = _download_bytes(ASSIGNMENT_TABLES_URL, "assignment_tables.zip")
    zf = zipfile.ZipFile(io.BytesIO(asgn_bytes))

    # Determine which NP ID ranges are needed
    all_np_ids = set()
    for name in zf.namelist():
        m = re.search(r"(NP\d+)", name)
        if m:
            all_np_ids.add(m.group(1))

    # Step 2: SMILES CSVs — download all ranges (they are small, ~1-3 MB each)
    print(f"\n[Step 2] Downloading SMILES CSVs for {len(_SMILES_RANGES)} ranges ...")
    smiles_map: dict = {}
    for rng in _SMILES_RANGES:
        url = f"{NPMRD_BASE}/system/downloads/current/smiles_{rng}.csv.gz"
        content = _download_bytes(url, f"SMILES {rng}")
        smiles_map.update(_load_smiles_csv_gz(content))

    print(f"  Total SMILES loaded: {len(smiles_map)}")

    # Step 3: Extract CHSQC candidates
    print("\n[Step 3] Extracting experimental CHSQC data ...")
    candidates = _extract_chsqc_candidates(zf, smiles_map)

    if not candidates:
        raise NpmrdError(
            "No candidates found. Ensure NP-MRD has CHSQC assignment tables "
            "for compounds in the 8–20 carbon range."
        )

    # Step 4: Select diverse testset
    print(f"\n[Step 4] Selecting {target_count} compounds with C-count diversity ...")
    testset = select_testset(candidates, target=target_count)

    # Remove internal-only fields before saving
    for entry in testset:
        entry.pop("chsqc_table", None)

    # Step 5: Save
    print(f"\n[Step 5] Saving testset to {output_path} ...")
    with open(output_path, "w") as fh:
        json.dump(testset, fh, indent=2)

    print(f"  Saved {len(testset)} compounds.")
    return testset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download NP-MRD experimental HSQC assignments and build benchmark testset."
        )
    )
    parser.add_argument(
        "--output", default=TESTSET_PATH,
        help=f"Output JSON path (default: {TESTSET_PATH})"
    )
    parser.add_argument(
        "--count", type=int, default=TARGET_COUNT,
        help=f"Number of compounds to select (default: {TARGET_COUNT})"
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Re-download even if output file already exists"
    )
    args = parser.parse_args()

    testset = fetch_testset(
        output_path=args.output,
        target_count=args.count,
        force_refresh=args.force_refresh,
    )

    print(f"\n{'NP-MRD ID':<14} {'C':>3} {'Exp':>4}  {'Formula':<12}  SMILES")
    print("-" * 80)
    for c in testset:
        smi = c["smiles"][:40] + ("..." if len(c["smiles"]) > 40 else "")
        print(
            f"{c['np_mrd_id']:<14} {c['n_carbons']:>3} "
            f"{c['n_exp_peaks']:>4}  {c['molecular_formula']:<12}  {smi}"
        )

    print(f"\nSelected {len(testset)} compounds → {args.output}")
    print("Each entry includes experimental HSQC peaks (hsqc_peaks).")
    print("Run benchmark_npmrd.py to compare vs Zakodium predictions.")


if __name__ == "__main__":
    _cli()
