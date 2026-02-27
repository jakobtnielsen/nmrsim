# NMRSim Project Memory

## Status
Phase 1 (2D NMR generation) + Phase 2 (1H 1D) + Tier C inverse solver — all complete.
120/120 tests passing (1 transient network test may fail when API unavailable).

## NP-MRD Benchmark (`fetch_npmrd.py` + `benchmark_npmrd.py`)
- NP-MRD bulk JSON: structural metadata only (no NMR shifts); NMR data is web-interface-only
- NP-MRD has a bulk `assignment_tables.zip` (~454 KB) with 194 experimental CHSQC tables
- CHSQC table format: `H,{h_atom_num},C,{c_atom_num},{h_ppm},{c_ppm}` (one row per H atom, deduplicate for HSQC peaks)
- SMILES CSV gzipped files at `/system/downloads/current/smiles_NP{range}.csv.gz` (need browser User-Agent header)
- Strategy: use CHSQC tables as experimental HSQC reference; Zakodium as prediction
- Tolerances: H_TOL=0.50, C_TOL=5.00, C_TOL_OCH=7.00 (O-bearing CH, 50–90 ppm)
- Results v2 (20 compounds, diastereotopic splits + wider tol): raw MAE H=0.279/C=2.77; after reref H=0.132/C=1.146 ppm
- Mean matched peaks: 5.2/7.9 (66%); outliers: NP0001161 (tryptophan, C=9 ppm off), NP0001364 (ferulic acid, C=6 ppm off)
- Testset: `data/npmrd_testset.json` (includes `hsqc_peaks` experimental field); results: `outputs/npmrd_comparison.{csv,json}`

## Diastereotopic CH₂ splits (build_2d.py)
- `_get_diastereotopic_ch2_pairs`: detects CH₂XY (prochiral) only — NOT CH₂X₂ (symmetric)
  - CH₂X₂: same rank with and without chirality → no split (correct)
  - CH₂XY: same rank without chirality, different with chirality → ±|N(0,0.25)| split
- `apply_diastereotopic_splits(shift_result, random_seed=42)`: same δ across HSQC/HMBC/COSY via same seed

## COSY symmetry fix (collapse_peaks.py)
- `_collapse_hh` now normalizes pairs (h1 ≤ h2) before clustering, then re-emits (h1,h2)+(h2,h1)
- This fixes COSY asymmetry for long-chain aliphatics (decanoic acid, oleic acid)
- Tests updated: `test_cosy_merged_within_tolerance` → expects 2 peaks (pair+mirror), `test_cosy_not_merged_different_h2` → expects 4 peaks; new `test_cosy_symmetric_input_preserved`

## Key Files
- `predict_shifts.py` — Stage 1: 1H/13C shifts (nmrdb, nmrshiftdb, sgnn backends)
- `build_2d.py` — Stage 2: HSQC/HMBC/COSY from shifts+RDKit graph
- `build_1h.py` — Stage 3: 1H 1D multiplets (stub with NotImplementedError)
- `collapse_peaks.py` — Stage 4: merge overlapping 2D peaks
- `generate_problem.py` — top-level: SMILES → benchmark JSON
- `solve_structure.py` — **Tier C inverse solver**: peak lists → candidate SMILES
- `difficulty_report.py` — batch difficulty scorer over outputs/nmr_*.json
- `score_answer.py` — grading (Tanimoto similarity)
- `validate.py` — sanity checks
- `lsd_bin/lsd` + `lsd_bin/outlsd` — LSD solver binaries (standalone, no SWI-Prolog)

## LSD Solver (solve_structure.py)
See `memory/lsd_notes.md` for detailed LSD binary notes.

Key points:
- `_C_QUAT_TOL = 0.25` ppm for quat-C detection (tight, to distinguish 130.1 quat from 129.8 CH)
- `_C_PPM_TOL = 1.0` ppm for HMBC c_ppm → atom node matching
- HMBC file format: `HMBC {c_id} {h_id}` (carbon first, H-bearing atom second)
- LSD exit code 1 = success; check .sol file existence instead
- Carbonyl O threshold: c_ppm > 175 (ketone/ester AND aldehyde n_h=1)
- Fix A: formula-based quat-C completion in extract_atom_nodes(), capped at deficit==1 only
  (larger deficits like quercetin's 4 cause LSD timeout — skip them)
- N assigned before remaining O in detect_fragments() so NH2 H budget is reserved for N
- N=O fallback pairing in step-2 sp2 parity when no eligible sp2 C > 150 ppm
- rank_candidates() short-circuits after 3 consecutive NMRShiftDB failures (network down)
- Candidates with score=None sorted last (stable); sort key: (score is None, score or 0.0)

## Current benchmark state (16 molecules) — after HMBC all-atoms fix
- 12/16 GT rank=1: acetophenone(1sol), 4-methoxyacetophenone(4sol), ibuprofen(1sol),
  2-acetylthiophene(10sol), 4-aminobenzoic_acid(5sol), coumarin(5sol), ethyl_benzoate(1sol),
  menthol(1sol), naproxen(13sol), indole(3sol), isoquinoline(2sol), test_001(1sol)
- GT found but rank>1: vanillin (45 solutions, rank 2)
- GT not found / excluded: caffeine (633 solutions, no aromatic H), p-nitroaniline
  (9 solutions, LSD can't represent nitro group formal charges), quercetin (4639 solutions, complex)

## Key fix: HMBC all-atoms constraint (extract_constraints)
Previously only the representative (lowest lsd_id) in an equivalent group got HMBC.
Now all atoms in the group get the HMBC constraint. This was the root cause of:
- ibuprofen 334 solutions → 1 (the two equivalent aromatic CH pairs were unconstrained)
- menthol 52 solutions → 1 (equivalent CH3 groups were unconstrained)
- 4-methoxyacetophenone 43 → 4, 4-aminobenzoic_acid 38 → 5, etc.

## Running
```
uv run python generate_problem.py SMILES
uv run python difficulty_report.py
uv run python -m pytest tests/ -v
```

## Test structure
- `tests/test_shifts.py` — predict_shifts
- `tests/test_2d.py` — build_2d
- `tests/test_collapse.py` — collapse_peaks
- `tests/test_known_molecules.py` — regression on real molecules
- `tests/test_1h.py` — build_1h (stubs)
- `tests/test_solver.py` — solve_structure (25 tests)
