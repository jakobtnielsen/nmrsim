# NMRSim Project Memory

## Status
Phase 1 (2D NMR generation) + Phase 2 (1H 1D) + Tier C inverse solver — all complete.
119/119 tests passing.

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
- Carbonyl O threshold: c_ppm > 175 (ketone/ester); aryl-ether C at 145-165 gets sp3 O

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
