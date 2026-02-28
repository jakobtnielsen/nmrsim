"""
benchmark_hybrid.py — LSD structure elucidation on hybrid NMR data.

For each compound in the NP-MRD testset (20 natural products, 8–20C):
  - HSQC: experimental (derived from NP-MRD CHSQC assignment tables, with n_h)
  - HMBC: semi-simulated — experimental 1H/13C shifts + RDKit molecular topology
  - COSY: semi-simulated — experimental 1H shifts  + RDKit molecular topology

Atom mapping: NP-MRD atom numbers are 1-based and follow RDKit canonical atom
ordering. Mapping: rdkit_idx = npmrd_atom_num - 1.

Usage:
    python benchmark_hybrid.py [--testset data/npmrd_testset.json]
                               [--zip    data/assignment_tables.zip]
                               [--output-dir outputs/]
                               [--max-candidates 50]
"""

import argparse
import csv
import json
import os
import zipfile
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from build_2d import (
    _get_graph_distances,
    _get_equivalent_h_groups,
    _representative,
)
from collapse_peaks import collapse_2d_peaks
from predict_shifts import predict_shifts, BackendError
from solve_structure import (
    extract_atom_nodes,
    extract_constraints,
    detect_fragments,
    run_lsd,
    run_lsd_iterative,
    rank_candidates,
)
from score_answer import tanimoto_similarity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DIR              = os.path.dirname(os.path.abspath(__file__))
TESTSET_PATH      = os.path.join(_DIR, "data", "npmrd_testset.json")
V2_TESTSET_PATH   = os.path.join(_DIR, "data", "npmrd_v2_testset.json")
ZIP_PATH          = os.path.join(_DIR, "data", "assignment_tables.zip")
OUTPUT_DIR        = os.path.join(_DIR, "outputs")
SHIFT_CACHE_PATH  = os.path.join(_DIR, "data", "predict_shifts_cache.json")
GEN_CACHE_PATH    = os.path.join(_DIR, "data", "generate_problem_cache.json")


# ---------------------------------------------------------------------------
# Predict-shifts cache (persisted across runs)
# ---------------------------------------------------------------------------

def _load_shift_cache() -> dict:
    """Load cached predict_shifts results from disk, or return empty dict."""
    if os.path.exists(SHIFT_CACHE_PATH):
        try:
            with open(SHIFT_CACHE_PATH) as f:
                raw = json.load(f)
            # JSON serialises dict keys as strings; convert back to int for shift dicts.
            result = {}
            for smi, entry in raw.items():
                result[smi] = {
                    "c_shifts": {int(k): v for k, v in (entry.get("c_shifts") or {}).items()},
                    "h_shifts": {int(k): v for k, v in (entry.get("h_shifts") or {}).items()},
                }
            return result
        except Exception:
            return {}
    return {}


def _save_shift_cache(cache: dict) -> None:
    """Persist predict_shifts cache to disk (atomic write via temp file)."""
    tmp = SHIFT_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, SHIFT_CACHE_PATH)


def _load_gen_cache() -> dict:
    """Load cached generate_problem results from disk, or return empty dict."""
    if os.path.exists(GEN_CACHE_PATH):
        try:
            with open(GEN_CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_gen_cache(cache: dict) -> None:
    """Persist generate_problem cache to disk (atomic write via temp file)."""
    tmp = GEN_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, GEN_CACHE_PATH)


# ---------------------------------------------------------------------------
# Parsing assignment tables
# ---------------------------------------------------------------------------

def parse_assign_table(content: str) -> tuple:
    """
    Parse a NP-MRD regular assignment table.

    Format per line: atom_type,atom_num,shift,multiplicity[,J_values]
    atom_num is 1-based = RDKit canonical atom index + 1.

    Returns (h_shifts, c_shifts) as {rdkit_idx: ppm}.
    Atoms with missing/NA shifts are excluded.
    """
    h_shifts: dict = {}
    c_shifts: dict = {}

    for line in content.strip().splitlines():
        parts = line.split(",", 4)
        if len(parts) < 3:
            continue
        atom_type = parts[0].strip().upper()
        if atom_type not in ("H", "C"):
            continue
        try:
            atom_num = int(parts[1].strip())
        except ValueError:
            continue
        shift_str = parts[2].strip()
        if not shift_str or shift_str.upper() in ("NA", "N/A"):
            continue
        try:
            shift = float(shift_str)
        except ValueError:
            continue

        rdkit_idx = atom_num - 1  # 1-based → 0-based
        if atom_type == "H":
            h_shifts[rdkit_idx] = shift
        else:
            c_shifts[rdkit_idx] = shift

    return h_shifts, c_shifts


def parse_chsqc_table(content: str) -> list:
    """
    Parse a NP-MRD CHSQC assignment table into HSQC peaks with n_h.

    Format: H,h_atom_num,C,c_atom_num,h_ppm,c_ppm
    Groups H entries sharing the same C atom number into one peak,
    deduplicates identical (h_ppm, c_ppm) pairs, and sums n_h.

    Returns [{h_ppm, c_ppm, n_h, c_npmrd_idx}, ...] sorted by h_ppm.
    """
    rows_by_c: dict = defaultdict(list)

    for line in content.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            c_num = int(parts[3].strip())
            h_ppm = float(parts[4].strip())
            c_ppm = float(parts[5].strip())
        except (ValueError, IndexError):
            continue
        rows_by_c[c_num].append((round(h_ppm, 4), round(c_ppm, 4)))

    peaks = []
    seen: set = set()
    for c_num, entries in sorted(rows_by_c.items()):
        # Deduplicate: count distinct (h_ppm, c_ppm) occurrences
        from collections import Counter
        cnt = Counter(entries)
        for (h_ppm, c_ppm), n_h in cnt.items():
            key = (h_ppm, c_ppm)
            if key in seen:
                continue
            seen.add(key)
            peaks.append({
                "h_ppm":      h_ppm,
                "c_ppm":      c_ppm,
                "n_h":        n_h,
                "c_npmrd_num": c_num,          # kept for diagnostics
            })

    return sorted(peaks, key=lambda p: p["h_ppm"])


# ---------------------------------------------------------------------------
# CHSQC-derived shift fallbacks (for assign tables missing H or C entries)
# ---------------------------------------------------------------------------

def _c_shifts_from_chsqc(chsqc_content: str) -> dict:
    """
    Build {rdkit_c_idx: c_ppm} from CHSQC table.
    c_rdkit = c_npmrd_num - 1 (1-based → 0-based).
    Used when the regular assign table has no C entries.
    """
    c_shifts: dict = {}
    for line in chsqc_content.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            c_num = int(parts[3].strip())
            c_ppm = float(parts[5].strip())
        except (ValueError, IndexError):
            continue
        c_rdkit = c_num - 1
        c_shifts[c_rdkit] = round(c_ppm, 4)
    return c_shifts


def _h_shifts_from_chsqc(mol_h, chsqc_content: str) -> dict:
    """
    Build {rdkit_h_idx: h_ppm} from CHSQC table, mapping via C neighbours.

    For each CHSQC entry (c_num, h_ppm): find H atoms attached to c_rdkit
    in mol_h and assign h_ppm to each.  For CH2/CH3 groups all H get the
    same shift (mean of CHSQC entries for that C).
    """
    c_to_h_ppms: dict = defaultdict(list)
    for line in chsqc_content.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            c_num = int(parts[3].strip())
            h_ppm = float(parts[4].strip())
        except (ValueError, IndexError):
            continue
        c_to_h_ppms[c_num].append(round(h_ppm, 4))

    h_shifts: dict = {}
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        c_rdkit = atom.GetIdx()
        c_num   = c_rdkit + 1
        ppms    = c_to_h_ppms.get(c_num)
        if not ppms:
            continue
        mean_ppm = round(sum(ppms) / len(ppms), 4)
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                h_shifts[nbr.GetIdx()] = mean_ppm
    return h_shifts


# ---------------------------------------------------------------------------
# Building HMBC and COSY from experimental shifts + RDKit topology
# ---------------------------------------------------------------------------

def build_hmbc_exp(mol_h, h_shifts: dict, c_shifts: dict) -> list:
    """
    Build HMBC peak list from experimental shifts + RDKit bond-graph topology.

    Uses standard H equivalence groups (no diastereotopic splitting —
    experimental shifts are used exactly as assigned).
    Emits one peak per (h_ppm, c_ppm) pair at graph distance 2 or 3.
    """
    dist    = _get_graph_distances(mol_h)
    h_groups = _get_equivalent_h_groups(mol_h)

    peaks = []
    seen: set = set()

    for h_idxs in h_groups.values():
        rep_h = _representative(h_idxs)
        h_ppm = h_shifts.get(rep_h)
        if h_ppm is None:
            continue

        for c_atom in mol_h.GetAtoms():
            if c_atom.GetAtomicNum() != 6:
                continue
            c_idx = c_atom.GetIdx()
            d = dist[rep_h][c_idx]
            if d not in (2, 3):
                continue
            c_ppm = c_shifts.get(c_idx)
            if c_ppm is None:
                continue

            key = (round(h_ppm, 4), round(c_ppm, 4))
            if key in seen:
                continue
            seen.add(key)
            peaks.append({"h_ppm": round(h_ppm, 4), "c_ppm": round(c_ppm, 4)})

    return sorted(peaks, key=lambda p: (p["h_ppm"], p["c_ppm"]))


def build_cosy_exp(mol_h, h_shifts: dict) -> list:
    """
    Build COSY peak list from experimental shifts + RDKit bond-graph topology.

    Emits both (h1,h2) and (h2,h1) for each H–H pair at distance 2 or 3.
    """
    dist     = _get_graph_distances(mol_h)
    h_groups = _get_equivalent_h_groups(mol_h)
    group_list = list(h_groups.values())

    peaks = []
    seen: set = set()

    for i in range(len(group_list)):
        for j in range(len(group_list)):
            if i == j:
                continue
            rep_a = _representative(group_list[i])
            rep_b = _representative(group_list[j])
            d = dist[rep_a][rep_b]
            if d not in (2, 3):
                continue
            h1 = h_shifts.get(rep_a)
            h2 = h_shifts.get(rep_b)
            if h1 is None or h2 is None:
                continue
            key = (round(h1, 4), round(h2, 4))
            if key in seen:
                continue
            seen.add(key)
            peaks.append({"h1_ppm": round(h1, 4), "h2_ppm": round(h2, 4)})

    return sorted(peaks, key=lambda p: (p["h1_ppm"], p["h2_ppm"]))


# ---------------------------------------------------------------------------
# 1D 13C spectrum from experimental assignments
# ---------------------------------------------------------------------------

def build_c13_spectrum(chsqc_content: str, assign_content: str) -> list:
    """
    Build a 1D 13C spectrum list from CHSQC + regular assignment tables.

    Returns [{"c_ppm": float, "n_h": int, "h_ppm": float|None}, ...]
    One entry per carbon atom (keyed by atom number from assign table).
    n_h  = number of H entries in CHSQC table pointing to this C.
    h_ppm = mean of CHSQC H shifts for this C (None if n_h == 0).
    """
    # Build c_num → [h_ppm, ...] from CHSQC table (format: H,h#,C,c#,h_ppm,c_ppm)
    c_to_h_ppms: dict = defaultdict(list)
    for line in chsqc_content.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            c_num = int(parts[3].strip())
            h_ppm = float(parts[4].strip())
        except (ValueError, IndexError):
            continue
        c_to_h_ppms[c_num].append(round(h_ppm, 4))

    c13_list = []
    for line in assign_content.strip().splitlines():
        parts = line.split(",", 4)
        if len(parts) < 3:
            continue
        atom_type = parts[0].strip().upper()
        if atom_type != 'C':
            continue
        shift_str = parts[2].strip()
        if not shift_str or shift_str.upper() in ('NA', 'N/A'):
            continue
        try:
            c_num = int(parts[1].strip())
            c_ppm = float(shift_str)
        except ValueError:
            continue
        hs = c_to_h_ppms.get(c_num, [])
        n_h = len(hs)
        h_ppm = round(sum(hs) / len(hs), 4) if hs else None
        c13_list.append({
            "c_ppm": round(c_ppm, 4),
            "n_h":   n_h,
            "h_ppm": h_ppm,
        })
    return c13_list


def _build_c13_from_shifts(c_shifts: dict, chsqc_content: str) -> list:
    """
    Build a 1D 13C spectrum list from a complete {rdkit_c_idx: c_ppm} dict.

    Uses CHSQC table to determine n_h and h_ppm for each CH carbon.
    Quaternary carbons (not in CHSQC) get n_h=0 and h_ppm=None.

    Returns [{"c_ppm": float, "n_h": int, "h_ppm": float|None}, ...].
    """
    c_to_h_ppms: dict = defaultdict(list)
    for line in chsqc_content.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            c_num = int(parts[3].strip())
            h_ppm = float(parts[4].strip())
        except (ValueError, IndexError):
            continue
        c_to_h_ppms[c_num].append(round(h_ppm, 4))

    c13 = []
    for c_rdkit, c_ppm in sorted(c_shifts.items()):
        c_num = c_rdkit + 1
        hs    = c_to_h_ppms.get(c_num, [])
        n_h   = len(hs)
        h_ppm = round(sum(hs) / len(hs), 4) if hs else None
        c13.append({"c_ppm": round(c_ppm, 4), "n_h": n_h, "h_ppm": h_ppm})
    return c13


# ---------------------------------------------------------------------------
# Per-compound pipeline
# ---------------------------------------------------------------------------

def _molecular_formula(mol) -> str:
    return rdMolDescriptors.CalcMolFormula(mol)


def run_compound(entry: dict, zf: zipfile.ZipFile,
                 chsqc_map: dict, assign_map: dict,
                 max_candidates: int = 50,
                 lsd_timeout: int = 30,
                 shift_cache: dict = None) -> dict:
    """
    Run the hybrid elucidation pipeline for one compound.

    Returns a result dict with:
      np_mrd_id, smiles, formula,
      n_hsqc, n_hmbc, n_cosy,
      h_assigned, c_assigned,
      lsd_count, gt_rank, gt_tanimoto,
      error (if any)
    """
    npid   = entry["np_mrd_id"]
    smiles = entry["smiles"]
    result_base = {"np_mrd_id": npid, "smiles": smiles, "error": None}

    try:
        mol   = Chem.MolFromSmiles(smiles)
        mol_h = Chem.AddHs(mol)
        formula = _molecular_formula(mol)

        # --- Parse experimental data ---
        with zf.open(assign_map[npid]) as f:
            assign_content = f.read().decode()
        with zf.open(chsqc_map[npid]) as f:
            chsqc_content = f.read().decode()

        h_shifts_exp, c_shifts_exp = parse_assign_table(assign_content)
        hsqc_peaks = parse_chsqc_table(chsqc_content)

        # --- Supplement missing shifts from CHSQC when assign table is partial ---
        # Some assign tables have only H entries (no C) or only C entries (no H).
        # CHSQC provides reliable fallback values for both, keyed by atom index.
        c_shifts_chsqc = _c_shifts_from_chsqc(chsqc_content)
        # Filter CHSQC entries to actual C atom indices.  NP-MRD canonical
        # atom numbers may differ from the current RDKit canonicalization for
        # some molecules, producing out-of-range or non-C atom indices that
        # would inflate the C count.
        c_atom_indices = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
        c_shifts_chsqc = {k: v for k, v in c_shifts_chsqc.items() if k in c_atom_indices}
        h_shifts_chsqc = _h_shifts_from_chsqc(mol_h, chsqc_content)
        # Merge: experimental assign table overrides CHSQC for overlapping atoms
        c_shifts_merged = {**c_shifts_chsqc, **c_shifts_exp}
        h_shifts_merged = {**h_shifts_chsqc, **h_shifts_exp}

        # --- Supplement missing quaternary C shifts (H-only assign tables) ---
        # When the assign table has no C entries, c_shifts_merged contains only
        # CHSQC-derived CH carbons.  Quaternary C atoms (no attached H) have
        # no c_ppm → HMBC cannot reach them → LSD gets wrong atom count → 0 solutions.
        # Fix: use predicted 13C shifts from NMRShiftDB for the missing carbons.
        formula_c = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        if len(c_shifts_merged) < formula_c:
            try:
                canon_smi = Chem.MolToSmiles(mol)
                if shift_cache is not None and canon_smi in shift_cache:
                    pred = shift_cache[canon_smi]
                else:
                    pred = predict_shifts(canon_smi, backend='nmrshiftdb')
                    if shift_cache is not None:
                        # Only store JSON-serialisable fields; skip Mol objects etc.
                        shift_cache[canon_smi] = {
                            "c_shifts": {str(k): v for k, v in (pred.get("c_shifts") or {}).items()},
                            "h_shifts": {str(k): v for k, v in (pred.get("h_shifts") or {}).items()},
                        }
                for idx, c_ppm in (pred.get('c_shifts') or {}).items():
                    if int(idx) not in c_shifts_merged:
                        c_shifts_merged[int(idx)] = c_ppm
            except (BackendError, Exception):
                pass  # fall through; quaternary C will be detected via HMBC only

        # --- Build 1D 13C spectrum (includes quaternary C) ---
        # Always build from c_shifts_merged so that supplemented quat C is included.
        # For compounds with full C entries the assign-table c13 equals this result.
        c13 = build_c13_spectrum(chsqc_content, assign_content)
        if len(c13) < formula_c and len(c_shifts_merged) >= formula_c:
            # Assign table was incomplete; rebuild from the augmented shifts dict
            c13 = _build_c13_from_shifts(c_shifts_merged, chsqc_content)

        # --- Build semi-simulated HMBC and COSY ---
        # Use the final (possibly augmented) c_shifts_merged so that HMBC peaks
        # to quaternary C atoms are included when their shifts are now known.
        hmbc_raw = build_hmbc_exp(mol_h, h_shifts_merged, c_shifts_merged)
        cosy_raw = build_cosy_exp(mol_h, h_shifts_merged)

        # Collapse (standard tolerances)
        hmbc = collapse_2d_peaks(hmbc_raw, "hmbc")
        cosy = collapse_2d_peaks(cosy_raw, "cosy")

        # LSD solver needs HSQC without c_npmrd_num
        hsqc_for_lsd = [
            {"h_ppm": p["h_ppm"], "c_ppm": p["c_ppm"], "n_h": p["n_h"]}
            for p in hsqc_peaks
        ]

        problem = {
            "molecular_formula": formula,
            "spectra": {
                "hsqc": hsqc_for_lsd,
                "hmbc": hmbc,
                "cosy": cosy,
                "c13":  c13,   # 1D 13C enables direct atom node building
            },
        }

        # --- Run LSD (iterative ELIM relaxation) ---
        atom_nodes  = extract_atom_nodes(problem)
        constraints = extract_constraints(problem, atom_nodes)
        fragments   = detect_fragments(atom_nodes, formula)
        smiles_list, lsd_text, lsd_pass = run_lsd_iterative(
            atom_nodes, constraints, fragments, formula,
            lsd_timeout=lsd_timeout,
        )
        lsd_count = len(smiles_list)

        base_fields = {
            "formula":    formula,
            "n_hsqc":     len(hsqc_peaks),
            "n_c13":      len(c13),
            "n_hmbc":     len(hmbc),
            "n_cosy":     len(cosy),
            "h_assigned": len(h_shifts_exp),
            "c_assigned": len(c_shifts_exp),
            "lsd_pass":   lsd_pass,
        }

        if lsd_count == 0:
            return {**result_base, **base_fields,
                    "lsd_count": 0, "gt_rank": None, "gt_tanimoto": None,
                    "candidates": []}

        if lsd_count <= max_candidates:
            candidates = rank_candidates(smiles_list, atom_nodes, problem)
        else:
            candidates = [
                {"smiles": s, "score": None, "rank": i + 1}
                for i, s in enumerate(smiles_list[:max_candidates])
            ]

        # Score: find ground truth rank
        gt_rank = None
        gt_tanimoto = None
        for cand in candidates:
            t = tanimoto_similarity(cand["smiles"], smiles)
            if gt_tanimoto is None or t > gt_tanimoto:
                gt_tanimoto = t
                if t >= 0.99:
                    gt_rank = cand["rank"]

        return {
            **result_base,
            **base_fields,
            "lsd_count":   lsd_count,
            "gt_rank":     gt_rank,
            "gt_tanimoto": round(gt_tanimoto, 4) if gt_tanimoto else None,
            "candidates":  candidates[:10],  # top 10 for JSON output
            "lsd_text":    lsd_text,
        }

    except Exception as exc:
        import traceback
        return {**result_base, "error": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(testset_path: str = TESTSET_PATH,
                  zip_path: str = ZIP_PATH,
                  output_dir: str = OUTPUT_DIR,
                  max_candidates: int = 50,
                  output_prefix: str = "hybrid_benchmark",
                  lsd_timeout: int = 30) -> None:

    os.makedirs(output_dir, exist_ok=True)
    csv_path  = os.path.join(output_dir, f"{output_prefix}.csv")
    json_path = os.path.join(output_dir, f"{output_prefix}.json")

    with open(testset_path) as f:
        testset = json.load(f)

    with zipfile.ZipFile(zip_path) as zf:
        all_names = zf.namelist()

    chsqc_map  = {n.split("_")[1]: n for n in all_names if "chsqc" in n}
    assign_map = {n.split("_")[1]: n for n in all_names if "chsqc" not in n}

    # Filter testset to compounds present in both maps
    runnable = [e for e in testset
                if e["np_mrd_id"] in chsqc_map and e["np_mrd_id"] in assign_map]
    print(f"Running hybrid benchmark on {len(runnable)}/{len(testset)} compounds …\n")

    shift_cache = _load_shift_cache()
    print(f"Shift cache: {len(shift_cache)} entries loaded from {SHIFT_CACHE_PATH}\n")

    results = []
    with zipfile.ZipFile(zip_path) as zf:
        for i, entry in enumerate(runnable, 1):
            npid = entry["np_mrd_id"]
            n_c  = entry.get("n_carbons", "?")
            print(f"[{i:2d}/{len(runnable)}] {npid} ({n_c}C) … ", end="", flush=True)

            r = run_compound(entry, zf, chsqc_map, assign_map, max_candidates,
                             lsd_timeout, shift_cache)
            results.append(r)
            _save_shift_cache(shift_cache)  # persist after each compound

            if r.get("error"):
                print(f"ERROR: {r['error'].splitlines()[-1]}")
            else:
                rank_str = (f"rank={r['gt_rank']}" if r["gt_rank"]
                            else f"tanimoto={r['gt_tanimoto']}")
                c13_str  = f", C13={r.get('n_c13', '-')}" if r.get('n_c13') else ""
                print(f"LSD={r['lsd_count']} solutions ({r.get('lsd_pass','?')}), GT {rank_str}  "
                      f"[HSQC={r['n_hsqc']}{c13_str}, HMBC={r['n_hmbc']}, COSY={r['n_cosy']}]")

    # --- Summary ---
    n_ok       = sum(1 for r in results if not r.get("error"))
    n_found    = sum(1 for r in results if r.get("gt_rank") is not None)
    n_rank1    = sum(1 for r in results if r.get("gt_rank") == 1)
    lsd_counts = [r["lsd_count"] for r in results if not r.get("error") and "lsd_count" in r]
    gt_tanimotos = [r["gt_tanimoto"] for r in results
                    if r.get("gt_tanimoto") is not None]

    print(f"\n{'='*60}")
    print(f"Results: {n_ok}/{len(runnable)} ran successfully")
    print(f"  GT found (Tanimoto≥0.99): {n_found}/{n_ok}")
    print(f"  GT rank=1:                {n_rank1}/{n_ok}")
    if lsd_counts:
        import math
        avg_log = sum(math.log10(max(c, 1)) for c in lsd_counts) / len(lsd_counts)
        print(f"  Median LSD solutions:     {sorted(lsd_counts)[len(lsd_counts)//2]}")
        print(f"  Mean log10(solutions):    {avg_log:.2f}")
    if gt_tanimotos:
        print(f"  Mean GT Tanimoto:         {sum(gt_tanimotos)/len(gt_tanimotos):.3f}")

    # Per-compound GT rank breakdown
    rank_hist: dict = defaultdict(int)
    for r in results:
        if r.get("gt_rank"):
            rank_hist[r["gt_rank"]] += 1
    if rank_hist:
        print(f"  Rank distribution: " +
              ", ".join(f"#{k}:{v}" for k, v in sorted(rank_hist.items())))
    print()

    # --- Save CSV ---
    csv_fields = [
        "np_mrd_id", "smiles", "formula",
        "n_hsqc", "n_c13", "n_hmbc", "n_cosy",
        "h_assigned", "c_assigned",
        "lsd_count", "lsd_pass", "gt_rank", "gt_tanimoto", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Saved → {csv_path}")

    # --- Save JSON (strip lsd_text to keep file manageable) ---
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k != "lsd_text"}
        json_results.append(jr)
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Saved → {json_path}")


# ---------------------------------------------------------------------------
# Measurement-noise injection for simulated spectra
# ---------------------------------------------------------------------------

def _add_measurement_noise(problem: dict,
                           sigma_h: float = 0.02,
                           sigma_c: float = 0.05,
                           random_seed: int = 42) -> dict:
    """
    Add independent Gaussian measurement noise to every peak coordinate.

    Each coordinate gets its own draw from N(0, σ), modelling the
    digitisation/processing error of each individually measured FID point.
    The same resonance therefore appears at slightly different positions in
    HSQC vs HMBC (and between different HMBC peaks for the same nucleus),
    as in real 2D NMR where indirect-dimension points are independently
    quantised in each experiment.

    Practical upper limits imposed by the solver's matching tolerances:
      sigma_h < 0.025 ppm  (_H_PPM_TOL = 0.05; larger → H-constraints missed)
      sigma_c < 0.08 ppm   (_C_QUAT_TOL = 0.25; larger → spurious quat-C nodes)
    Studying noise beyond these limits requires also widening those tolerances.

    Args:
        problem:     Problem dict as returned by generate_problem().
        sigma_h:     Gaussian σ for ¹H coordinates in ppm (default 0.02).
        sigma_c:     Gaussian σ for ¹³C coordinates in ppm (default 0.05).
        random_seed: Seed for reproducibility.

    Returns:
        New problem dict with noisy peak coordinates; original is unchanged.
    """
    import random as _rnd
    rng = _rnd.Random(random_seed + 999)   # distinct from diastereotopic seed

    def n_h(): return rng.gauss(0, sigma_h)
    def n_c(): return rng.gauss(0, sigma_c)

    spectra = problem.get("spectra", {})
    noisy_spectra = dict(spectra)

    noisy_spectra["hsqc"] = [
        {"h_ppm": round(p["h_ppm"] + n_h(), 4),
         "c_ppm": round(p["c_ppm"] + n_c(), 4),
         "n_h":   p["n_h"]}
        for p in spectra.get("hsqc", [])
    ]
    noisy_spectra["hmbc"] = [
        {"h_ppm": round(p["h_ppm"] + n_h(), 4),
         "c_ppm": round(p["c_ppm"] + n_c(), 4)}
        for p in spectra.get("hmbc", [])
    ]
    noisy_spectra["cosy"] = [
        {"h1_ppm": round(p["h1_ppm"] + n_h(), 4),
         "h2_ppm": round(p["h2_ppm"] + n_h(), 4)}
        for p in spectra.get("cosy", [])
    ]
    if "c13" in spectra:
        noisy_spectra["c13"] = [
            {**entry, "c_ppm": round(entry["c_ppm"] + n_c(), 4)}
            for entry in spectra["c13"]
        ]

    noisy = dict(problem)
    noisy["spectra"] = noisy_spectra
    return noisy


# ---------------------------------------------------------------------------
# Fully-simulated benchmark (generate_problem → LSD, no ZIP needed)
# ---------------------------------------------------------------------------

def run_simulated_compound(entry: dict, backend: str = "nmrdb",
                           lsd_timeout: int = 30, max_candidates: int = 50,
                           gen_cache: dict = None,
                           sigma_h: float = 0.0, sigma_c: float = 0.0) -> dict:
    """
    Run LSD on a fully-simulated problem (no experimental data).

    Calls generate_problem(smiles, backend) to obtain HSQC/HMBC/COSY, then
    runs the same LSD pipeline as run_compound.  Results are cached by
    (canonical_smiles, backend) in gen_cache so re-runs skip the API call.

    If sigma_h > 0 or sigma_c > 0, Gaussian measurement noise is injected
    into the peak coordinates after loading from cache (the cache always
    stores the clean, noiseless problem).
    """
    from generate_problem import generate_problem

    npid   = entry["np_mrd_id"]
    smiles = entry["smiles"]
    result_base = {"np_mrd_id": npid, "smiles": smiles, "error": None}

    try:
        mol       = Chem.MolFromSmiles(smiles)
        formula   = _molecular_formula(mol)
        canon_smi = Chem.MolToSmiles(mol)

        cache_key = f"{canon_smi}|{backend}"
        if gen_cache is not None and cache_key in gen_cache:
            problem = gen_cache[cache_key]
        else:
            problem = generate_problem(canon_smi, npid, backend=backend, random_seed=42)
            if gen_cache is not None:
                gen_cache[cache_key] = problem

        # Apply measurement noise after cache lookup (cache stores clean data).
        if sigma_h > 0.0 or sigma_c > 0.0:
            problem = _add_measurement_noise(problem, sigma_h, sigma_c, random_seed=42)

        spectra = problem.get("spectra", {})
        hsqc = spectra.get("hsqc", [])
        hmbc = spectra.get("hmbc", [])
        cosy = spectra.get("cosy", [])
        c13  = spectra.get("c13",  [])

        atom_nodes  = extract_atom_nodes(problem)
        constraints = extract_constraints(problem, atom_nodes)
        fragments   = detect_fragments(atom_nodes, formula)
        smiles_list, lsd_text, lsd_pass = run_lsd_iterative(
            atom_nodes, constraints, fragments, formula, lsd_timeout=lsd_timeout,
        )
        lsd_count = len(smiles_list)

        base_fields = {
            "formula":  formula,
            "n_hsqc":   len(hsqc),
            "n_c13":    len(c13),
            "n_hmbc":   len(hmbc),
            "n_cosy":   len(cosy),
            "lsd_pass": lsd_pass,
        }

        if lsd_count == 0:
            return {**result_base, **base_fields,
                    "lsd_count": 0, "gt_rank": None, "gt_tanimoto": None,
                    "candidates": []}

        if lsd_count <= max_candidates:
            candidates = rank_candidates(smiles_list, atom_nodes, problem)
        else:
            candidates = [
                {"smiles": s, "score": None, "rank": i + 1}
                for i, s in enumerate(smiles_list[:max_candidates])
            ]

        gt_rank     = None
        gt_tanimoto = None
        for cand in candidates:
            t = tanimoto_similarity(cand["smiles"], smiles)
            if gt_tanimoto is None or t > gt_tanimoto:
                gt_tanimoto = t
                if t >= 0.99:
                    gt_rank = cand["rank"]

        return {
            **result_base,
            **base_fields,
            "lsd_count":   lsd_count,
            "gt_rank":     gt_rank,
            "gt_tanimoto": round(gt_tanimoto, 4) if gt_tanimoto else None,
            "candidates":  candidates[:10],
            "lsd_text":    lsd_text,
        }

    except Exception as exc:
        import traceback
        return {**result_base, "error": traceback.format_exc(), "lsd_count": 0}


def run_simulated_benchmark(testset_path: str,
                            backend: str = "nmrdb",
                            output_dir: str = OUTPUT_DIR,
                            max_candidates: int = 50,
                            output_prefix: str = "simulated_benchmark",
                            lsd_timeout: int = 30,
                            sigma_h: float = 0.0,
                            sigma_c: float = 0.0) -> None:

    os.makedirs(output_dir, exist_ok=True)
    csv_path  = os.path.join(output_dir, f"{output_prefix}.csv")
    json_path = os.path.join(output_dir, f"{output_prefix}.json")

    with open(testset_path) as f:
        testset = json.load(f)

    gen_cache = _load_gen_cache()
    cached = sum(1 for e in testset
                 if f"{Chem.MolToSmiles(Chem.MolFromSmiles(e['smiles']))}|{backend}" in gen_cache)
    noise_str = (f"σ_H={sigma_h} ppm, σ_C={sigma_c} ppm"
                 if (sigma_h or sigma_c) else "none")
    print(f"Generate-problem cache: {len(gen_cache)} total entries, "
          f"{cached}/{len(testset)} testset compounds already cached")
    print(f"Measurement noise: {noise_str}\n")

    results = []
    for i, entry in enumerate(testset, 1):
        npid = entry["np_mrd_id"]
        n_c  = entry.get("n_carbons", "?")
        print(f"[{i:2d}/{len(testset)}] {npid} ({n_c}C) … ", end="", flush=True)

        r = run_simulated_compound(entry, backend, lsd_timeout, max_candidates,
                                    gen_cache, sigma_h, sigma_c)
        results.append(r)
        _save_gen_cache(gen_cache)

        if r.get("error"):
            print(f"ERROR: {r['error'].splitlines()[-1]}")
        else:
            rank_str = (f"rank={r['gt_rank']}" if r["gt_rank"]
                        else f"tanimoto={r['gt_tanimoto']}")
            c13_str  = f", C13={r.get('n_c13', '-')}" if r.get('n_c13') else ""
            print(f"LSD={r['lsd_count']} solutions ({r.get('lsd_pass','?')}), GT {rank_str}  "
                  f"[HSQC={r['n_hsqc']}{c13_str}, HMBC={r['n_hmbc']}, COSY={r['n_cosy']}]")

    # --- Summary ---
    n_ok      = sum(1 for r in results if not r.get("error"))
    n_found   = sum(1 for r in results if r.get("gt_rank") is not None)
    n_rank1   = sum(1 for r in results if r.get("gt_rank") == 1)
    lsd_counts   = [r["lsd_count"] for r in results if not r.get("error") and "lsd_count" in r]
    gt_tanimotos = [r["gt_tanimoto"] for r in results if r.get("gt_tanimoto") is not None]

    print(f"\n{'='*60}")
    print(f"Results: {n_ok}/{len(testset)} ran successfully")
    print(f"  GT found (Tanimoto≥0.99): {n_found}/{n_ok}")
    print(f"  GT rank=1:                {n_rank1}/{n_ok}")
    if lsd_counts:
        import math
        avg_log = sum(math.log10(max(c, 1)) for c in lsd_counts) / len(lsd_counts)
        print(f"  Median LSD solutions:     {sorted(lsd_counts)[len(lsd_counts)//2]}")
        print(f"  Mean log10(solutions):    {avg_log:.2f}")
    if gt_tanimotos:
        print(f"  Mean GT Tanimoto:         {sum(gt_tanimotos)/len(gt_tanimotos):.3f}")

    rank_hist: dict = defaultdict(int)
    for r in results:
        if r.get("gt_rank"):
            rank_hist[r["gt_rank"]] += 1
    if rank_hist:
        print(f"  Rank distribution: " +
              ", ".join(f"#{k}:{v}" for k, v in sorted(rank_hist.items())))
    print()

    # --- Save CSV ---
    csv_fields = [
        "np_mrd_id", "smiles", "formula",
        "n_hsqc", "n_c13", "n_hmbc", "n_cosy",
        "lsd_count", "lsd_pass", "gt_rank", "gt_tanimoto", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Saved → {csv_path}")

    # --- Save JSON ---
    json_results = [{k: v for k, v in r.items() if k != "lsd_text"} for r in results]
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Saved → {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testset",        default=None,
                   help="Path to testset JSON (default: npmrd_testset.json)")
    p.add_argument("--v2",             action="store_true",
                   help="Use v2 testset (npmrd_v2_testset.json) with 1D 13C support")
    p.add_argument("--zip",            default=ZIP_PATH)
    p.add_argument("--output-dir",     default=OUTPUT_DIR)
    p.add_argument("--output-prefix",  default=None,
                   help="Output file prefix (default: derived from testset flag)")
    p.add_argument("--max-candidates", type=int, default=50)
    p.add_argument("--lsd-timeout",    type=int, default=30,
                   help="LSD timeout in seconds per pass (default 30)")
    p.add_argument("--simulated",      action="store_true",
                   help="Use fully-simulated data (generate_problem) instead of experimental ZIP")
    p.add_argument("--backend",        default="nmrdb",
                   help="Shift-prediction backend for --simulated mode (default: nmrdb)")
    p.add_argument("--sigma-h",        type=float, default=0.0,
                   help="Gaussian noise σ for ¹H shifts in ppm (default 0, no noise)")
    p.add_argument("--sigma-c",        type=float, default=0.0,
                   help="Gaussian noise σ for ¹³C shifts in ppm (default 0, no noise)")
    args = p.parse_args()

    if args.simulated:
        testset = args.testset or TESTSET_PATH
        prefix  = args.output_prefix or f"simulated_{args.backend}_benchmark"
        run_simulated_benchmark(testset, args.backend, args.output_dir,
                                args.max_candidates, prefix, args.lsd_timeout,
                                args.sigma_h, args.sigma_c)
    elif args.v2:
        testset = args.testset or V2_TESTSET_PATH
        prefix  = args.output_prefix or "hybrid_v2_benchmark"
        run_benchmark(testset, args.zip, args.output_dir, args.max_candidates, prefix,
                      args.lsd_timeout)
    else:
        testset = args.testset or TESTSET_PATH
        prefix  = args.output_prefix or "hybrid_benchmark"
        run_benchmark(testset, args.zip, args.output_dir, args.max_candidates, prefix,
                      args.lsd_timeout)


if __name__ == "__main__":
    _cli()
