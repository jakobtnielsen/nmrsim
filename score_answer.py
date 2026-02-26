"""
score_answer.py — Grade a predicted SMILES against the ground truth.

Usage (CLI):
    python score_answer.py --predicted "CC(=O)c1ccccc1" --problem outputs/nmr_test_001.json
"""

import argparse
import json

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors


def tanimoto_similarity(smiles_pred: str, smiles_true: str) -> float:
    """Morgan fingerprint Tanimoto similarity (radius=2, 2048 bits)."""
    mol_pred = Chem.MolFromSmiles(smiles_pred)
    mol_true = Chem.MolFromSmiles(smiles_true)
    if mol_pred is None or mol_true is None:
        return 0.0
    fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, radius=2, nBits=2048)
    fp_true = AllChem.GetMorganFingerprintAsBitVect(mol_true, radius=2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp_pred, fp_true))


def score_smiles(predicted_smiles: str, true_smiles: str) -> dict:
    """
    Score a predicted SMILES against the ground truth.

    Returns:
    {
      "exact_match":   bool,    # canonical SMILES identical
      "tanimoto":      float,   # Morgan fp Tanimoto (radius=2, 2048 bits)
      "formula_match": bool,    # molecular formula identical
      "mw_delta":      float,   # |MW_pred - MW_true| in Da
      "valid_smiles":  bool,    # predicted SMILES parseable
    }
    """
    mol_pred = Chem.MolFromSmiles(predicted_smiles)
    mol_true = Chem.MolFromSmiles(true_smiles)

    valid = mol_pred is not None

    if not valid or mol_true is None:
        return {
            "exact_match":   False,
            "tanimoto":      0.0,
            "formula_match": False,
            "mw_delta":      float("inf"),
            "valid_smiles":  valid,
        }

    can_pred  = Chem.MolToSmiles(mol_pred)
    can_true  = Chem.MolToSmiles(mol_true)
    exact     = (can_pred == can_true)
    tanimoto  = tanimoto_similarity(predicted_smiles, true_smiles)

    formula_pred  = rdMolDescriptors.CalcMolFormula(mol_pred)
    formula_true  = rdMolDescriptors.CalcMolFormula(mol_true)
    formula_match = (formula_pred == formula_true)

    mw_pred = Descriptors.MolWt(mol_pred)
    mw_true = Descriptors.MolWt(mol_true)

    return {
        "exact_match":   exact,
        "tanimoto":      round(tanimoto, 4),
        "formula_match": formula_match,
        "mw_delta":      round(abs(mw_pred - mw_true), 3),
        "valid_smiles":  True,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Score a predicted SMILES against a benchmark problem."
    )
    parser.add_argument("--predicted", required=True,
                        help="Predicted SMILES string")
    parser.add_argument("--problem",   required=True,
                        help="Path to benchmark problem JSON")
    args = parser.parse_args()

    with open(args.problem) as fh:
        problem = json.load(fh)

    gt = problem.get("ground_truth")
    if gt is None:
        print("ERROR: problem JSON has no ground_truth block.")
        raise SystemExit(1)

    true_smiles = gt["canonical_smiles"]
    result = score_smiles(args.predicted, true_smiles)

    print(f"Predicted:    {args.predicted}")
    print(f"Ground truth: {true_smiles}")
    print()
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
