"""benchmark_npmrd.py — Compare Zakodium-predicted HSQC vs NP-MRD experimental HSQC.

For each compound in the NP-MRD testset (which embeds experimental HSQC peaks
from the NP-MRD assignment_tables.zip), runs the Zakodium prediction pipeline
and compares the predicted HSQC peaks to the experimental reference.

Computes:
  - Hungarian-matched peak count and mean absolute error (H and 13C ppm)
  - Optional linear re-referencing to remove systematic offsets

Outputs:
  outputs/npmrd_comparison.csv   — per-compound metric table
  outputs/npmrd_comparison.json  — full peak-by-peak details

Usage:
    python benchmark_npmrd.py [--testset data/npmrd_testset.json]
                              [--output-dir outputs/]
                              [--workers 2]
                              [--fetch]
"""

import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import scipy.optimize

from generate_problem import generate_problem, ProblemGenerationError
from predict_shifts import BackendError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(os.path.abspath(__file__))
TESTSET_PATH = os.path.join(_DIR, "data", "npmrd_testset.json")
OUTPUT_DIR   = os.path.join(_DIR, "outputs")
CSV_PATH     = os.path.join(OUTPUT_DIR, "npmrd_comparison.csv")
JSON_PATH    = os.path.join(OUTPUT_DIR, "npmrd_comparison.json")

H_TOL     = 0.50   # ppm: H tolerance for a matched pair
C_TOL     = 5.00   # ppm: C tolerance for a matched pair
C_TOL_OCH = 7.00   # ppm: wider C tolerance for O-bearing CH (50–90 ppm)
MIN_PAIRS_REREF = 3

MAX_WORKERS  = 2   # conservative: each compound makes 2 API calls (1H + 13C)
_RANDOM_SEED = 42

# Serialize generate_problem() calls to respect API rate limits
_API_LOCK = threading.Semaphore(1)


# ---------------------------------------------------------------------------
# Matching and re-referencing
# ---------------------------------------------------------------------------

def _match_hsqc_peaks(obs: list, sim: list,
                      h_tol: float = H_TOL, c_tol: float = C_TOL,
                      c_tol_och: float = C_TOL_OCH) -> dict:
    """
    Hungarian-algorithm matching of two HSQC peak lists.

    Cost: sqrt( (dH/h_tol)^2 + (dC/c_tol_i)^2 )
    c_tol_i is C_TOL_OCH for O-bearing CH carbons (obs_c 50–90 ppm),
    C_TOL otherwise.
    A pair is "valid" if |dH| < h_tol AND |dC| < c_tol_i.

    Returns dict with n_matched, mae_h_raw, mae_c_raw, mae_h_matched,
    mae_c_matched, pairs (list of dicts), obs_idx, sim_idx, valid_mask.
    """
    empty = {
        "obs_idx": np.array([], dtype=int), "sim_idx": np.array([], dtype=int),
        "valid_mask": np.array([], dtype=bool), "n_matched": 0,
        "mae_h_raw": float("nan"), "mae_c_raw": float("nan"),
        "mae_h_matched": None, "mae_c_matched": None, "pairs": [],
    }
    if not obs or not sim:
        return empty

    obs_h = np.array([p["h_ppm"] for p in obs])
    obs_c = np.array([p["c_ppm"] for p in obs])
    sim_h = np.array([p["h_ppm"] for p in sim])
    sim_c = np.array([p["c_ppm"] for p in sim])

    # Per-row C tolerance: wider for O-bearing CH carbons (50–90 ppm)
    c_tols = np.where((obs_c >= 50) & (obs_c <= 90), c_tol_och, c_tol)

    dh = np.abs(obs_h[:, None] - sim_h[None, :])
    dc = np.abs(obs_c[:, None] - sim_c[None, :])
    cost = np.sqrt((dh / h_tol) ** 2 + (dc / c_tols[:, None]) ** 2)

    row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost)
    dh_a = dh[row_idx, col_idx]
    dc_a = dc[row_idx, col_idx]
    c_tols_a = c_tols[row_idx]
    valid = (dh_a < h_tol) & (dc_a < c_tols_a)

    pairs = [
        {"obs_h": float(obs_h[r]), "obs_c": float(obs_c[r]),
         "sim_h": float(sim_h[s]), "sim_c": float(sim_c[s]),
         "dh": float(dh_a[i]),     "dc": float(dc_a[i]),
         "valid": bool(valid[i])}
        for i, (r, s) in enumerate(zip(row_idx, col_idx))
    ]

    n_matched = int(valid.sum())
    return {
        "obs_idx": row_idx, "sim_idx": col_idx, "valid_mask": valid,
        "n_matched": n_matched,
        "mae_h_raw": float(dh_a.mean()) if len(dh_a) else float("nan"),
        "mae_c_raw": float(dc_a.mean()) if len(dc_a) else float("nan"),
        "mae_h_matched": float(dh_a[valid].mean()) if n_matched else None,
        "mae_c_matched": float(dc_a[valid].mean()) if n_matched else None,
        "pairs": pairs,
    }


def _fit_linear_reref(obs: np.ndarray, sim: np.ndarray) -> tuple:
    """Fit obs = a*sim + b; return (a, b)."""
    if len(sim) < 2 or len(np.unique(sim)) < 2:
        raise ValueError("Need ≥ 2 unique sim values")
    a, b = np.polyfit(sim, obs, 1)
    return float(a), float(b)


def _apply_reref(peaks: list, a_h: float, b_h: float,
                 a_c: float, b_c: float) -> list:
    """Return new peak list with re-referenced shifts."""
    return [{"h_ppm": round(a_h * p["h_ppm"] + b_h, 4),
             "c_ppm": round(a_c * p["c_ppm"] + b_c, 4),
             "n_h":   p.get("n_h", 1)}
            for p in peaks]


# ---------------------------------------------------------------------------
# Per-compound metrics
# ---------------------------------------------------------------------------

def _compute_metrics(np_mrd_id: str, exp_hsqc: list,
                     sim_hsqc: list, entry: dict) -> dict:
    """Compute all comparison metrics for one compound."""
    raw = _match_hsqc_peaks(exp_hsqc, sim_hsqc)

    m: dict = {
        "np_mrd_id":        np_mrd_id,
        "smiles":           entry["smiles"],
        "molecular_formula": entry.get("molecular_formula", ""),
        "n_carbons":        entry.get("n_carbons", 0),
        "n_peaks_exp":      len(exp_hsqc),
        "n_peaks_sim":      len(sim_hsqc),
        "peak_count_diff":  abs(len(exp_hsqc) - len(sim_hsqc)),
        "n_matched":        raw["n_matched"],
        "mae_h_raw":        raw["mae_h_raw"],
        "mae_c_raw":        raw["mae_c_raw"],
        "reref_a_h": None, "reref_b_h": None,
        "reref_a_c": None, "reref_b_c": None,
        "mae_h_corrected":  None,
        "mae_c_corrected":  None,
        "reref_skipped":    False,
        "error":            None,
        # private: for JSON output
        "_exp_hsqc":              exp_hsqc,
        "_sim_hsqc":              sim_hsqc,
        "_sim_hsqc_corrected":    None,
        "_pairs_raw":             raw["pairs"],
        "_pairs_corrected":       None,
    }

    if raw["n_matched"] < MIN_PAIRS_REREF:
        m["reref_skipped"] = True
        return m

    valid = raw["valid_mask"]
    obs_h = np.array([p["obs_h"] for p in raw["pairs"]])[valid]
    sim_h = np.array([p["sim_h"] for p in raw["pairs"]])[valid]
    obs_c = np.array([p["obs_c"] for p in raw["pairs"]])[valid]
    sim_c = np.array([p["sim_c"] for p in raw["pairs"]])[valid]

    try:
        a_h, b_h = _fit_linear_reref(obs_h, sim_h)
        a_c, b_c = _fit_linear_reref(obs_c, sim_c)
    except (ValueError, np.linalg.LinAlgError):
        m["reref_skipped"] = True
        return m

    sim_corr = _apply_reref(sim_hsqc, a_h, b_h, a_c, b_c)
    corr = _match_hsqc_peaks(exp_hsqc, sim_corr)

    m.update({
        "reref_a_h": a_h, "reref_b_h": b_h,
        "reref_a_c": a_c, "reref_b_c": b_c,
        "mae_h_corrected": corr["mae_h_matched"],
        "mae_c_corrected": corr["mae_c_matched"],
        "_sim_hsqc_corrected": sim_corr,
        "_pairs_corrected":    corr["pairs"],
    })
    return m


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_compound(compound: dict) -> tuple:
    """
    Run Zakodium pipeline for one compound (thread-safe via _API_LOCK).

    Returns (np_mrd_id, sim_hsqc, error_str_or_None).
    """
    nid    = compound["np_mrd_id"]
    smiles = compound["smiles"]
    _API_LOCK.acquire()
    try:
        problem = generate_problem(
            smiles=smiles,
            problem_id=nid,
            backend="nmrdb",
            collapse=True,
            include_1h=False,
            random_seed=_RANDOM_SEED,
        )
        time.sleep(1)
        return nid, problem["spectra"]["hsqc"], None
    except (BackendError, ProblemGenerationError) as exc:
        return nid, None, str(exc)
    except Exception as exc:
        return nid, None, f"Unexpected: {exc}"
    finally:
        _API_LOCK.release()


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

_CSV_COLS = [
    "np_mrd_id", "smiles", "molecular_formula", "n_carbons",
    "n_peaks_exp", "n_peaks_sim", "peak_count_diff", "n_matched",
    "mae_h_raw", "mae_c_raw",
    "mae_h_corrected", "mae_c_corrected",
    "reref_a_h", "reref_b_h", "reref_a_c", "reref_b_c",
    "reref_skipped", "error",
]


def _fmt(v) -> str:
    if v is None: return ""
    if isinstance(v, float):
        return "" if v != v else f"{v:.4f}"  # nan → ""
    return str(v)


def _save_csv(results: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_COLS)
        for r in results:
            w.writerow([_fmt(r.get(c)) for c in _CSV_COLS])


def _save_json(results: list, summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    per = []
    for r in results:
        entry = {c: r.get(c) for c in _CSV_COLS}
        for k, v in entry.items():
            if isinstance(v, float) and v != v:
                entry[k] = None
        entry["exp_hsqc"]              = r.get("_exp_hsqc", [])
        entry["sim_hsqc"]              = r.get("_sim_hsqc", [])
        entry["sim_hsqc_corrected"]    = r.get("_sim_hsqc_corrected")
        entry["matched_pairs_raw"]     = r.get("_pairs_raw", [])
        entry["matched_pairs_corrected"] = r.get("_pairs_corrected")
        per.append(entry)
    out = {
        "summary": summary,
        "description": (
            "Exp HSQC = NP-MRD experimental CHSQC assignment table. "
            "Sim HSQC = Zakodium (nmrdb) prediction pipeline."
        ),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "per_compound": per,
    }
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    testset_path: str = TESTSET_PATH,
    output_csv:   str = CSV_PATH,
    output_json:  str = JSON_PATH,
    max_workers:  int = MAX_WORKERS,
) -> dict:
    """Run the full benchmark and return the summary dict."""
    if not os.path.exists(testset_path):
        raise FileNotFoundError(
            f"Testset not found: {testset_path}\n"
            "Run: python fetch_npmrd.py  (or use --fetch)"
        )
    with open(testset_path) as fh:
        testset = json.load(fh)

    # Validate that testset has experimental HSQC
    if testset and "hsqc_peaks" not in testset[0]:
        raise ValueError(
            "Testset does not contain 'hsqc_peaks'. "
            "Re-run fetch_npmrd.py --force-refresh to rebuild with experimental data."
        )

    n = len(testset)
    print(f"\nNP-MRD benchmark: {n} compounds "
          f"(Zakodium predicted vs NP-MRD experimental HSQC) ...\n")

    id_map = {c["np_mrd_id"]: c for c in testset}
    results = []
    n_ok = n_fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_compound, c): c["np_mrd_id"] for c in testset}
        done = 0
        for fut in as_completed(futures):
            done += 1
            nid, sim_hsqc, err = fut.result()
            compound = id_map[nid]
            exp_hsqc = compound["hsqc_peaks"]
            prefix = f"[{done:2d}/{n}] {nid} ({compound['n_carbons']}C)"

            if err:
                n_fail += 1
                m = {c: None for c in _CSV_COLS}
                m.update({
                    "np_mrd_id": nid, "smiles": compound["smiles"],
                    "n_carbons": compound["n_carbons"],
                    "molecular_formula": compound["molecular_formula"],
                    "error": err, "reref_skipped": True,
                    "_exp_hsqc": exp_hsqc, "_sim_hsqc": [],
                })
                print(f"{prefix}  FAIL: {err[:60]}")
            else:
                n_ok += 1
                try:
                    m = _compute_metrics(nid, exp_hsqc, sim_hsqc, compound)
                except Exception as exc:
                    m = {c: None for c in _CSV_COLS}
                    m.update({"np_mrd_id": nid, "error": str(exc),
                               "_exp_hsqc": exp_hsqc, "_sim_hsqc": sim_hsqc})
                    print(f"{prefix}  METRICS-FAIL: {exc}")
                else:
                    nm   = m["n_matched"]
                    mhr  = m["mae_h_raw"];    mcr = m["mae_c_raw"]
                    mhc  = m.get("mae_h_corrected")
                    mcc  = m.get("mae_c_corrected")
                    reref = (f" → corr H={mhc:.3f} C={mcc:.2f}"
                             if (mhc is not None and mcc is not None) else "")
                    mhr_s = f"{mhr:.3f}" if mhr == mhr else "nan"
                    mcr_s = f"{mcr:.2f}" if mcr == mcr else "nan"
                    print(f"{prefix}  exp={m['n_peaks_exp']} sim={m['n_peaks_sim']} "
                          f"matched={nm}  MAE H={mhr_s} C={mcr_s}{reref}")
            results.append(m)

    def _mean(vals):
        clean = [v for v in vals if v is not None and v == v]
        return float(np.mean(clean)) if clean else None

    summary = {
        "n_compounds":          n,
        "n_success":            n_ok,
        "n_failed":             n_fail,
        "ref_source":           "NP-MRD experimental CHSQC",
        "sim_backend":          "nmrdb (Zakodium)",
        "mean_mae_h_raw":       _mean([r.get("mae_h_raw")       for r in results]),
        "mean_mae_c_raw":       _mean([r.get("mae_c_raw")       for r in results]),
        "mean_mae_h_corrected": _mean([r.get("mae_h_corrected") for r in results]),
        "mean_mae_c_corrected": _mean([r.get("mae_c_corrected") for r in results]),
        "mean_n_matched":       _mean([r.get("n_matched")       for r in results]),
        "mean_n_peaks_exp":     _mean([r.get("n_peaks_exp")     for r in results]),
        "mean_n_peaks_sim":     _mean([r.get("n_peaks_sim")     for r in results]),
    }

    _save_csv(results, output_csv)
    _save_json(results, summary, output_json)

    print("\n" + "=" * 60)
    print(f"Results: {n_ok}/{n} success, {n_fail} failed")
    for label, key in [
        ("MAE H  raw      ", "mean_mae_h_raw"),
        ("MAE C  raw      ", "mean_mae_c_raw"),
        ("MAE H  corrected", "mean_mae_h_corrected"),
        ("MAE C  corrected", "mean_mae_c_corrected"),
    ]:
        v = summary[key]
        if v is not None:
            print(f"  {label}: {v:.3f} ppm")
    if summary["mean_n_matched"] is not None:
        print(f"  Mean matched     : {summary['mean_n_matched']:.1f} / "
              f"{summary['mean_n_peaks_exp']:.1f} experimental peaks")
    print(f"\nSaved → {output_csv}")
    print(f"Saved → {output_json}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark: Zakodium-predicted HSQC vs NP-MRD experimental HSQC."
        )
    )
    parser.add_argument("--testset",    default=TESTSET_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--workers",    type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--fetch", action="store_true",
        help="Run fetch_npmrd.py first if testset is missing"
    )
    args = parser.parse_args()

    if args.fetch and not os.path.exists(args.testset):
        from fetch_npmrd import fetch_testset
        print("Testset not found — running fetch_npmrd.py first ...")
        fetch_testset(output_path=args.testset)

    run_benchmark(
        testset_path=args.testset,
        output_csv=os.path.join(args.output_dir, "npmrd_comparison.csv"),
        output_json=os.path.join(args.output_dir, "npmrd_comparison.json"),
        max_workers=args.workers,
    )


if __name__ == "__main__":
    _cli()
