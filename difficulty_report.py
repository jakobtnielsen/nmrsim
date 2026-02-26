"""difficulty_report.py — Batch difficulty reporter for all benchmark problems.

Runs solve_structure.solve() over every outputs/nmr_*.json and emits a CSV
summary with lsd_count, difficulty score, and whether the ground truth is
within the top-ranked candidates.

Usage
-----
    uv run python difficulty_report.py [--max-candidates N] [--output PATH]

Output columns
--------------
problem_id, molecular_formula, lsd_count, difficulty,
gt_rank (rank of ground truth in candidates; 0 = not found),
gt_score (13C MAE of ground truth candidate; None if not found),
top1_smiles, top1_score
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from rdkit import Chem

from solve_structure import solve


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def _find_gt_rank(candidates: list, gt_smiles: str) -> tuple:
    """Return (rank, score) of the ground truth SMILES among candidates."""
    gt_canon = _canonical(gt_smiles)
    if not gt_canon:
        return 0, None
    for c in candidates:
        if _canonical(c['smiles']) == gt_canon:
            return c['rank'], c['score']
    return 0, None


def run_report(output_dir: str = "outputs",
               max_candidates: int = 50,
               output_path: str = "difficulty_report.csv",
               verbose: bool = True) -> list:
    """
    Run the solver on every nmr_*.json in output_dir.

    Returns list of result dicts (also written to output_path CSV).
    """
    problem_files = sorted(Path(output_dir).glob("nmr_*.json"))
    if not problem_files:
        print(f"No nmr_*.json files found in {output_dir}/", file=sys.stderr)
        return []

    rows = []
    for pf in problem_files:
        with open(pf) as f:
            problem = json.load(f)

        pid = problem.get('problem_id', pf.stem)
        formula = problem.get('molecular_formula', '')
        gt_smiles = (problem.get('ground_truth') or {}).get('smiles', '')

        if verbose:
            print(f"Solving {pid} ({formula})...", end=' ', flush=True)

        t0 = time.time()
        try:
            result = solve(problem, max_candidates=max_candidates)
        except Exception as exc:
            print(f"ERROR: {exc}")
            rows.append({
                'problem_id': pid,
                'molecular_formula': formula,
                'lsd_count': -1,
                'difficulty': -1,
                'gt_rank': 0,
                'gt_score': None,
                'top1_smiles': '',
                'top1_score': None,
                'elapsed_s': round(time.time() - t0, 1),
                'error': str(exc),
            })
            continue

        elapsed = round(time.time() - t0, 1)
        candidates = result['candidates']
        gt_rank, gt_score = _find_gt_rank(candidates, gt_smiles)
        top1 = candidates[0] if candidates else {}

        row = {
            'problem_id': pid,
            'molecular_formula': formula,
            'lsd_count': result['lsd_count'],
            'difficulty': round(result['difficulty'], 3),
            'gt_rank': gt_rank,
            'gt_score': round(gt_score, 3) if gt_score is not None else None,
            'top1_smiles': top1.get('smiles', ''),
            'top1_score': round(top1['score'], 3) if top1.get('score') is not None else None,
            'elapsed_s': elapsed,
            'error': '',
        }
        rows.append(row)

        if verbose:
            if result['lsd_count'] == 0:
                print("0 solutions (unsolvable with current constraints)")
            else:
                gt_info = f"GT rank={gt_rank}" if gt_rank else "GT not found"
                print(f"{result['lsd_count']} solutions, "
                      f"difficulty={row['difficulty']:.2f}, {gt_info} "
                      f"({elapsed}s)")

    # Write CSV
    if rows:
        fieldnames = [
            'problem_id', 'molecular_formula', 'lsd_count', 'difficulty',
            'gt_rank', 'gt_score', 'top1_smiles', 'top1_score',
            'elapsed_s', 'error',
        ]
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if verbose:
            print(f"\nReport written to {output_path}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Batch difficulty report for NMR benchmark problems"
    )
    parser.add_argument('--max-candidates', type=int, default=50,
                        help='Max candidates to rank per problem (default 50)')
    parser.add_argument('--output', default='difficulty_report.csv',
                        help='CSV output path (default difficulty_report.csv)')
    parser.add_argument('--dir', default='outputs',
                        help='Directory with nmr_*.json files (default outputs/)')
    args = parser.parse_args()

    rows = run_report(
        output_dir=args.dir,
        max_candidates=args.max_candidates,
        output_path=args.output,
        verbose=True,
    )

    if rows:
        solvable = [r for r in rows if r['lsd_count'] > 0]
        gt_found = [r for r in rows if r['gt_rank'] == 1]
        print(f"\nSummary: {len(rows)} problems, "
              f"{len(solvable)} solvable, "
              f"{len(gt_found)} with GT ranked #1")


if __name__ == '__main__':
    main()
