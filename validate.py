"""
validate.py — Sanity-check the full pipeline against known molecules.

Run:
    python validate.py --backend nmrdb
"""

import argparse
import json
import os
import time

from generate_problem import generate_problem, ProblemGenerationError
from predict_shifts import BackendError

# ---------------------------------------------------------------------------
# Test molecule set (NMR_SKILL.md Section 8)
# ---------------------------------------------------------------------------

TEST_MOLECULES = [
    {
        "id": "001",
        "name": "acetophenone",
        "smiles": "CC(=O)c1ccccc1",
        "expected_hsqc_count": 4,   # CH3, ortho, meta, para
        "expected_hmbc_key": True,  # must have CH3→C=O HMBC peak
        "expected_cosy_symmetric": True,
    },
    {
        "id": "002",
        "name": "4-methoxyacetophenone",
        "smiles": "COc1ccc(C(C)=O)cc1",
        "expected_hsqc_count": 4,   # OCH3, CH3, 2 sets ring H (AA'BB')
        "expected_hmbc_key": True,
        "expected_cosy_symmetric": True,
    },
    {
        "id": "003",
        "name": "caffeine",
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "expected_hsqc_count": 4,   # 3 N-CH3 + 1 vinyl CH
        "expected_hmbc_key": False,
        "expected_cosy_symmetric": True,
    },
    {
        "id": "004",
        "name": "ibuprofen",
        "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "expected_hsqc_count": 7,
        "expected_hmbc_key": True,
        "expected_cosy_symmetric": True,
    },
    {
        "id": "005",
        "name": "vanillin",
        "smiles": "COc1cc(C=O)ccc1O",
        "expected_hsqc_count": 5,   # OCH3, CHO, 3 unique ring H
        "expected_hmbc_key": True,
        "expected_cosy_symmetric": True,
    },
    {
        "id": "006",
        "name": "2-acetylthiophene",
        "smiles": "CC(=O)c1cccs1",
        "expected_hsqc_count": 4,   # CH3 + 3 thiophene H
        "expected_hmbc_key": True,
        "expected_cosy_symmetric": True,
    },
    {
        "id": "007",
        "name": "4-aminobenzoic_acid",
        "smiles": "Nc1ccc(C(=O)O)cc1",
        "expected_hsqc_count": 2,   # 2 sets ring H (AA'BB')
        "expected_hmbc_key": True,
        "expected_cosy_symmetric": True,
    },
    {
        "id": "008",
        "name": "coumarin",
        "smiles": "O=c1ccc2ccccc2o1",
        "expected_hsqc_count": 6,   # H3, H4 (vinyl) + 4 aromatic H (H5-H8)
        "expected_hmbc_key": True,
        "expected_cosy_symmetric": True,
    },
    {
        "id": "009",
        "name": "ethyl_benzoate",
        "smiles": "CCOC(=O)c1ccccc1",
        "expected_hsqc_count": 5,   # CH3, O-CH2, ortho/meta/para ring H
        "expected_hmbc_key": True,
        "expected_cosy_symmetric": True,
    },
    {
        "id": "010",
        "name": "menthol",
        "smiles": "OC1CC(C(C)C)CCC1C",
        "expected_hsqc_count": 8,   # complex aliphatic cyclohexane
        "expected_hmbc_key": False,  # no carbonyl in menthol
        "expected_cosy_symmetric": True,
    },
]

# Challenge molecules
CHALLENGE_MOLECULES = [
    {
        "id": "011",
        "name": "naproxen",
        "smiles": "CC(C(=O)O)c1ccc2cc(OC)ccc2c1",
        "expected_hsqc_count": None,
        "expected_hmbc_key": True,
        "expected_cosy_symmetric": True,
    },
    {
        "id": "012",
        "name": "p-nitroaniline",
        "smiles": "Nc1ccc([N+](=O)[O-])cc1",
        "expected_hsqc_count": 2,
        "expected_hmbc_key": False,  # no C=O; nitro group is N=O
        "expected_cosy_symmetric": True,
    },
    {
        "id": "013",
        "name": "quercetin",
        "smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
        "expected_hsqc_count": None,
        "expected_hmbc_key": True,
        "expected_cosy_symmetric": True,
    },
    {
        "id": "014",
        "name": "indole",
        "smiles": "c1ccc2[nH]ccc2c1",
        "expected_hsqc_count": 6,   # H2, H3 (pyrrole) + H4,H5,H6,H7 (benzene)
        "expected_hmbc_key": False,  # no C=O in indole
        "expected_cosy_symmetric": True,
    },
    {
        "id": "015",
        "name": "isoquinoline",
        "smiles": "c1ccc2cnccc2c1",
        "expected_hsqc_count": 7,   # 7 unique aromatic C-H
        "expected_hmbc_key": False,  # no C=O in isoquinoline
        "expected_cosy_symmetric": True,
    },
]


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def _check_cosy_symmetric(cosy_peaks: list) -> bool:
    pairs = {(round(p["h1_ppm"], 3), round(p["h2_ppm"], 3)) for p in cosy_peaks}
    for h1, h2 in pairs:
        if (h2, h1) not in pairs:
            return False
    return True


def _check_hmbc_carbonyl(hsqc_peaks, hmbc_peaks) -> bool:
    """
    Return True if there is at least one HMBC peak to a carbonyl-like carbon.
    Threshold of 155 ppm covers: ketone/aldehyde (~190-220), ester/acid (~160-180),
    amide/lactone (~155-175), aromatic C=O.
    """
    return any(p["c_ppm"] > 155.0 for p in hmbc_peaks)


def validate_pipeline(
    test_molecules: list,
    backend: str = "nmrdb",
    output_dir: str = "outputs/",
    save_problems: bool = True,
) -> dict:
    """
    Run the full pipeline on test molecules and return a validation report.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    pass_count = 0
    total = len(test_molecules)

    for mol_info in test_molecules:
        mid    = mol_info["id"]
        name   = mol_info["name"]
        smiles = mol_info["smiles"]
        exp_hsqc = mol_info.get("expected_hsqc_count")

        entry = {
            "id":     mid,
            "name":   name,
            "smiles": smiles,
            "passed": False,
            "checks": {},
            "error":  None,
            "problem_path": None,
        }

        # Retry up to 3 times on transient network errors
        last_exc = None
        problem = None
        for attempt in range(3):
            try:
                problem = generate_problem(
                    smiles=smiles,
                    problem_id=f"{mid}_{name}",
                    backend=backend,
                )
                break
            except ProblemGenerationError:
                raise   # don't retry sanity/logic errors
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(3)
        if problem is None:
            raise last_exc

        try:

            spectra  = problem["spectra"]
            hsqc     = spectra["hsqc"]
            hmbc     = spectra["hmbc"]
            cosy     = spectra["cosy"]

            checks = {}

            # HSQC count
            if exp_hsqc is not None:
                checks["hsqc_count"] = {
                    "expected": exp_hsqc,
                    "got":      len(hsqc),
                    "pass":     len(hsqc) == exp_hsqc,
                }
            else:
                checks["hsqc_count"] = {
                    "expected": "any",
                    "got":      len(hsqc),
                    "pass":     len(hsqc) > 0,
                }

            # HMBC non-empty
            checks["hmbc_nonempty"] = {
                "got":  len(hmbc),
                "pass": len(hmbc) > 0,
            }

            # COSY symmetry
            sym = _check_cosy_symmetric(cosy)
            checks["cosy_symmetric"] = {"pass": sym}

            # HMBC carbonyl correlation (for molecules with carbonyl)
            if mol_info.get("expected_hmbc_key"):
                from rdkit import Chem as _Chem
                mol_check = _Chem.MolFromSmiles(smiles)
                # Strict C=O detection: carbon must be one end of the double bond
                has_carbonyl = any(
                    bond.GetBondTypeAsDouble() == 2.0 and
                    (
                        (bond.GetBeginAtom().GetAtomicNum() == 6 and
                         bond.GetEndAtom().GetAtomicNum() == 8) or
                        (bond.GetBeginAtom().GetAtomicNum() == 8 and
                         bond.GetEndAtom().GetAtomicNum() == 6)
                    )
                    for bond in mol_check.GetBonds()
                )
                if has_carbonyl:
                    carbonyl_hmbc = _check_hmbc_carbonyl(hsqc, hmbc)
                    checks["hmbc_carbonyl"] = {"pass": carbonyl_hmbc}

            all_pass = all(c["pass"] for c in checks.values())
            entry["checks"] = checks
            entry["passed"] = all_pass

            if save_problems:
                from generate_problem import save_problem as _save
                path = _save(problem, output_dir=output_dir)
                entry["problem_path"] = path

            if all_pass:
                pass_count += 1

        except (BackendError, ProblemGenerationError) as exc:
            entry["error"] = str(exc)
        except Exception as exc:
            entry["error"] = f"Unexpected error: {exc}"

        results.append(entry)
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"  [{status}] {mid} {name}: "
              f"HSQC={len(spectra['hsqc']) if entry['error'] is None else 'ERR'}")
        if entry["error"]:
            print(f"         ERROR: {entry['error']}")

        time.sleep(0.5)  # brief pause between molecules

    report = {
        "backend":    backend,
        "total":      total,
        "passed":     pass_count,
        "pass_rate":  round(pass_count / total, 3) if total else 0.0,
        "results":    results,
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the NMR pipeline against known molecules."
    )
    parser.add_argument("--backend", default="nmrdb",
                        choices=["nmrdb", "nmrshiftdb", "sgnn"])
    parser.add_argument("--output-dir", default="outputs/")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save individual problem JSONs")
    args = parser.parse_args()

    all_mols = TEST_MOLECULES + CHALLENGE_MOLECULES

    print(f"\nRunning validation against {len(all_mols)} molecules "
          f"(backend={args.backend})...\n")

    report = validate_pipeline(
        all_mols,
        backend=args.backend,
        output_dir=args.output_dir,
        save_problems=not args.no_save,
    )

    # Save report
    report_path = os.path.join(args.output_dir, "validation_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nValidation report → {report_path}")

    # Print summary table
    print("\n" + "=" * 72)
    print(f"{'ID':<6} {'Molecule':<28} {'HSQC':>5} {'HMBC':>5} "
          f"{'COSY':>5} {'Status'}")
    print("-" * 72)
    for r in report["results"]:
        if r["error"] is None:
            from generate_problem import generate_problem as _gp
            # Read saved file for counts
            p_path = r.get("problem_path", "")
            if p_path and os.path.exists(p_path):
                with open(p_path) as fh:
                    p = json.load(fh)
                nh = len(p["spectra"]["hsqc"])
                nmb = len(p["spectra"]["hmbc"])
                nc  = len(p["spectra"]["cosy"])
            else:
                nh = nmb = nc = "?"
            status = "PASS" if r["passed"] else "FAIL"
            print(f"{r['id']:<6} {r['name']:<28} {nh!s:>5} {nmb!s:>5} "
                  f"{nc!s:>5} {status}")
        else:
            print(f"{r['id']:<6} {r['name']:<28} {'ERR':>5} {'ERR':>5} "
                  f"{'ERR':>5} ERROR")

    print("-" * 72)
    print(f"\nResult: {report['passed']}/{report['total']} passed "
          f"({report['pass_rate']*100:.0f}%)")

    # Save CSV summary
    import csv
    csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["problem_id", "molecule", "smiles",
                         "hsqc_peaks", "hmbc_peaks", "cosy_peaks",
                         "backend", "status"])
        for r in report["results"]:
            p_path = r.get("problem_path", "")
            if p_path and os.path.exists(p_path):
                with open(p_path) as pfh:
                    p = json.load(pfh)
                writer.writerow([
                    p["problem_id"],
                    r["name"],
                    r["smiles"],
                    len(p["spectra"]["hsqc"]),
                    len(p["spectra"]["hmbc"]),
                    len(p["spectra"]["cosy"]),
                    p["shift_backend"],
                    "PASS" if r["passed"] else "FAIL",
                ])
    print(f"CSV summary → {csv_path}")


if __name__ == "__main__":
    _cli()
