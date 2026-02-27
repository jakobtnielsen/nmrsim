"""
generate_problem.py — Top-level pipeline: SMILES → benchmark JSON.

Usage (CLI):
    python generate_problem.py --smiles "CC(=O)c1ccccc1" --id "test_001" --backend nmrdb
"""

import argparse
import json
import os
from datetime import datetime, timezone

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from predict_shifts import predict_shifts, BackendError
from build_2d import build_hsqc, build_hmbc, build_cosy
from collapse_peaks import collapse_2d_peaks


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ProblemGenerationError(Exception):
    """Raised when a generated problem fails sanity checks."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _molecular_formula(mol) -> str:
    return rdMolDescriptors.CalcMolFormula(mol)


def _degree_of_unsaturation(mol) -> int:
    """
    Degree of unsaturation = (2C + 2 + N - H - X) / 2
    where X = halogens. Returns integer (always a whole number for valid mols).
    """
    formula = rdMolDescriptors.CalcMolFormula(mol)
    # Use the standard formula: DoU = (2*C + 2 + N - H - F - Cl - Br - I) / 2
    from rdkit.Chem import rdMolDescriptors as rmd
    # Count atoms
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    C = counts.get("C", 0)
    H = counts.get("H", 0)
    N = counts.get("N", 0)
    halogens = (counts.get("F", 0) + counts.get("Cl", 0) +
                counts.get("Br", 0) + counts.get("I", 0))
    # Add implicit H
    mol_h = Chem.AddHs(mol)
    H = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
    dou = (2 * C + 2 + N - H - halogens) / 2
    return int(round(dou))


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

_H_SHIFT_MIN  = -1.0
_H_SHIFT_MAX  = 14.0
_C_SHIFT_MIN  = -10.0
_C_SHIFT_MAX  = 230.0


def _sanity_check(problem: dict, smiles: str) -> None:
    """
    Raise ProblemGenerationError if any sanity check fails.
    Checks are per NMR_SKILL.md Section 6 and the task specification.
    """
    spectra = problem["spectra"]
    hsqc    = spectra["hsqc"]
    hmbc    = spectra["hmbc"]
    cosy    = spectra["cosy"]

    # 1. Shift range checks
    for peak in hsqc:
        if not (_H_SHIFT_MIN <= peak["h_ppm"] <= _H_SHIFT_MAX):
            raise ProblemGenerationError(
                f"HSQC h_ppm {peak['h_ppm']} out of range "
                f"[{_H_SHIFT_MIN}, {_H_SHIFT_MAX}] for SMILES '{smiles}'"
            )
        if not (_C_SHIFT_MIN <= peak["c_ppm"] <= _C_SHIFT_MAX):
            raise ProblemGenerationError(
                f"HSQC c_ppm {peak['c_ppm']} out of range "
                f"[{_C_SHIFT_MIN}, {_C_SHIFT_MAX}] for SMILES '{smiles}'"
            )
    for peak in hmbc:
        if not (_H_SHIFT_MIN <= peak["h_ppm"] <= _H_SHIFT_MAX):
            raise ProblemGenerationError(
                f"HMBC h_ppm {peak['h_ppm']} out of range for SMILES '{smiles}'"
            )
        if not (_C_SHIFT_MIN <= peak["c_ppm"] <= _C_SHIFT_MAX):
            raise ProblemGenerationError(
                f"HMBC c_ppm {peak['c_ppm']} out of range for SMILES '{smiles}'"
            )

    # 2. Non-empty HSQC
    if len(hsqc) == 0:
        raise ProblemGenerationError(
            f"HSQC peak list is empty for SMILES '{smiles}'"
        )

    # 3. Non-empty HMBC (unless trivial — no H present at all besides methyl)
    #    We warn but don't fail for molecules with very few H.

    # 4. COSY symmetry: if (a,b) present, (b,a) must be present
    cosy_pairs = {(round(p["h1_ppm"], 3), round(p["h2_ppm"], 3)) for p in cosy}
    for h1, h2 in cosy_pairs:
        if (h2, h1) not in cosy_pairs:
            raise ProblemGenerationError(
                f"COSY is not symmetric: ({h1}, {h2}) present but "
                f"({h2}, {h1}) missing for SMILES '{smiles}'"
            )

    # 5. No exact duplicate peaks (after rounding)
    def _check_no_duplicates(peaks: list, name: str) -> None:
        seen = set()
        for p in peaks:
            key = tuple(round(v, 4) for v in p.values())
            if key in seen:
                raise ProblemGenerationError(
                    f"Duplicate {name} peak {key} for SMILES '{smiles}'"
                )
            seen.add(key)

    _check_no_duplicates(hsqc, "HSQC")
    _check_no_duplicates(hmbc, "HMBC")
    _check_no_duplicates(cosy, "COSY")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_problem(
    smiles: str,
    problem_id: str,
    backend: str = "nmrdb",
    collapse: bool = True,
    include_1h: bool = False,
    random_seed: int = 42,
) -> dict:
    """
    Full pipeline: SMILES → benchmark problem JSON.

    Steps:
      1. Canonicalize SMILES
      2. Compute molecular formula and degree of unsaturation
      3. Predict shifts (using specified backend)
      4. Build HSQC, HMBC, COSY peak lists
      5. Collapse peaks (if collapse=True)
      6. Optionally build 1H 1D (if include_1h=True — Phase 2)
      7. Run sanity checks
      8. Assemble and return problem JSON

    Returns the full problem dict including ground_truth.
    Use strip_ground_truth() before giving to the evaluated AI.
    """
    # 1. Canonicalize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ProblemGenerationError(f"Invalid SMILES: '{smiles}'")
    canonical_smiles = Chem.MolToSmiles(mol)

    # 2. Molecular properties
    formula = _molecular_formula(mol)
    dou     = _degree_of_unsaturation(mol)

    # 3. Predict shifts
    shift_result = predict_shifts(canonical_smiles, backend=backend)

    # 4. Build 2D peaks
    hsqc_raw = build_hsqc(shift_result, random_seed=random_seed)
    hmbc_raw = build_hmbc(shift_result, random_seed=random_seed)
    cosy_raw = build_cosy(shift_result, random_seed=random_seed)

    # 5. Collapse peaks
    if collapse:
        hsqc_out = collapse_2d_peaks(hsqc_raw, "hsqc")
        hmbc_out = collapse_2d_peaks(hmbc_raw, "hmbc")
        cosy_out = collapse_2d_peaks(cosy_raw, "cosy")
    else:
        hsqc_out = hsqc_raw
        hmbc_out = hmbc_raw
        cosy_out = cosy_raw

    # 6. 1H 1D (Phase 2 stub)
    h1_1d = None
    if include_1h:
        from build_1h import build_1h_multiplets
        h1_1d = build_1h_multiplets(shift_result, random_seed=random_seed)

    # 7. InChI for ground truth
    from rdkit.Chem.inchi import MolToInchi
    inchi = MolToInchi(mol) or ""

    # 8. Assemble problem dict
    problem = {
        "problem_id":             problem_id,
        "molecular_formula":      formula,
        "degree_of_unsaturation": dou,
        "shift_backend":          backend,
        "generation_timestamp":   datetime.now(timezone.utc).isoformat(),
        "spectra": {
            "hsqc": hsqc_out,
            "hmbc": hmbc_out,
            "cosy": cosy_out,
            "h1_1d": h1_1d,
        },
        "ground_truth": {
            "smiles":           canonical_smiles,
            "canonical_smiles": canonical_smiles,
            "inchi":            inchi,
        },
    }

    # Run sanity checks
    _sanity_check(problem, canonical_smiles)

    return problem


def strip_ground_truth(problem: dict) -> dict:
    """Remove the ground_truth block. Returns the dict the evaluated AI sees."""
    result = problem.copy()
    result.pop("ground_truth", None)
    return result


def save_problem(problem: dict, output_dir: str = "outputs/") -> str:
    """Save problem JSON to outputs/. Returns the filepath."""
    os.makedirs(output_dir, exist_ok=True)
    pid  = problem["problem_id"]
    path = os.path.join(output_dir, f"nmr_{pid}.json")
    with open(path, "w") as fh:
        json.dump(problem, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an NMR benchmark problem from a SMILES string."
    )
    parser.add_argument("--smiles",   required=True, help="Input SMILES")
    parser.add_argument("--id",       required=True, dest="problem_id",
                        help="Problem ID (e.g. test_001)")
    parser.add_argument("--backend",  default="nmrdb",
                        choices=["nmrdb", "nmrshiftdb", "sgnn"],
                        help="Shift prediction backend (default: nmrdb)")
    parser.add_argument("--no-collapse", action="store_true",
                        help="Skip peak collapsing")
    parser.add_argument("--output-dir", default="outputs/",
                        help="Output directory (default: outputs/)")
    args = parser.parse_args()

    try:
        problem = generate_problem(
            smiles=args.smiles,
            problem_id=args.problem_id,
            backend=args.backend,
            collapse=not args.no_collapse,
        )
    except (BackendError, ProblemGenerationError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    path = save_problem(problem, output_dir=args.output_dir)
    print(f"Problem saved → {path}")

    # Print summary
    spectra = problem["spectra"]
    print(f"\nMolecular formula: {problem['molecular_formula']}")
    print(f"DoU:               {problem['degree_of_unsaturation']}")
    print(f"HSQC peaks:        {len(spectra['hsqc'])}")
    print(f"HMBC peaks:        {len(spectra['hmbc'])}")
    print(f"COSY peaks:        {len(spectra['cosy'])}")
    print(f"Backend:           {problem['shift_backend']}")


if __name__ == "__main__":
    _cli()
