# LSD Binary Notes

## Location
- `/home/jakob/projects/nmrsim/lsd_bin/lsd` — solver (ELF 64-bit, standalone)
- `/home/jakob/projects/nmrsim/lsd_bin/outlsd` — output converter
- Source: https://github.com/nuzillard/PyLSD (linux tarball, version a8)

## Input Format (base binary — no FORM/PIEC/range notation)
```
MULT {id} {symbol} {hyb} {n_h}   ; hyb: 1=sp, 2=sp2, 3=sp3
HMQC {c_id} {c_id}               ; declares 1-bond C-H (same id twice)
HMBC {c_id} {h_id}               ; CARBON first, H-bearing atom second!
COSY {h_id1} {h_id2}
BOND {id1} {id2}                  ; forced bond
ELIM {n} {p}                      ; allow n HMBC violations with path > p
EXIT
```

## Critical: HMBC argument order
**HMBC {c_target} {h_source}** — carbon goes first, H-bearing atom second.
H atom MUST be declared by HMQC. Quaternary C (no HMQC) goes in position 1.

## Running
```bash
lsd input.lsd                # writes input.sol
outlsd 5 < input.sol         # outputs SMILES (one per line)
outlsd 10 < input.sol        # outputs SDF 0D
```
LSD exits with code 1 even on success. Check for .sol file, not exit code.

## Common Errors
- "Odd number of sp2 atoms" → count sp2 atoms (all MULT with hyb=2); must be even
- "Cannot set HMBC between X and H-Y because H-Y is not defined by HMQC" → HMBC args reversed (H atom must be HMQC-declared)
- "error 116 Cannot read integer" → range notation (2 3) not supported; use single int
- "Unknown command name: FORM" → FORM/PIEC are PyLSD preprocessor commands only

## Quat C Detection
Use `_C_QUAT_TOL = 0.25` ppm (tight) to avoid merging nearby quat C into CH peaks.
Example: in 4-methoxyacetophenone, ipso-quat at 130.1 ppm is only 0.3 ppm from
meta-CH at 129.8 ppm. With 1.0 ppm tolerance (too loose), quat is missed.

## Tested Molecules
- Ethane (C2H6): 1 solution → "CC"
- Acetophenone (C8H8O): 2 solutions, GT rank 1
- 4-methoxyacetophenone (C9H10O2): 43 solutions, GT rank 1
