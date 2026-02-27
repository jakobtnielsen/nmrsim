"""solve_structure.py — Tier C inverse NMR solver.

Takes a benchmark problem JSON (spectra fields only; ground_truth not required)
and returns a ranked list of candidate SMILES strings.

Pipeline
--------
Stage 1  extract_atom_nodes()   HSQC + HMBC  → labeled C-atom inventory
Stage 2  extract_constraints()  COSY + HMBC  → LSD connectivity constraints
Stage 3  detect_fragments()     shift heuristics → heteroatom placement
Stage 4  run_lsd()              write .lsd, call LSD binary, parse SMILES
Stage 5  rank_candidates()      score by 13C shift MAE (Hungarian algorithm)

Public API
----------
solve(problem, max_candidates=50) → {
    "candidates":  [{"smiles": str, "score": float|None, "rank": int}, ...],
    "lsd_count":   int,
    "difficulty":  float,
    "lsd_file":    str,
}

LSD binary notes
----------------
* Standalone C binary — no SWI-Prolog required.
* Located at  lsd_bin/lsd  (relative to this file).
* Input format: MULT / HMQC / HMBC / COSY / BOND / ELIM / EXIT commands.
* Output: <stem>.sol file consumed by lsd_bin/outlsd to produce SMILES.
* LSD exits with code 1 even on success; check for .sol file existence instead.
"""

import math
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from itertools import cycle, islice
from typing import Optional

import numpy as np
import scipy.optimize
from rdkit import Chem

from predict_shifts import predict_shifts, BackendError

# ---------------------------------------------------------------------------
# Paths to LSD binaries
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_LSD_BIN    = os.path.join(_DIR, "lsd_bin", "lsd")
_OUTLSD_BIN = os.path.join(_DIR, "lsd_bin", "outlsd")

# Tolerances for peak matching
_C_PPM_TOL  = 1.0   # ppm: HMBC c_ppm → atom node matching (×1.5 in _match_c)
_C_QUAT_TOL = 0.25  # ppm: tight tolerance for quat-C detection from HMBC
                    # Must be < typical shift difference between nearby C atoms
                    # (e.g. 0.3 ppm between ipso quat and adjacent CH in 4-MeAP)
_H_PPM_TOL  = 0.05  # ppm: HMBC/COSY h_ppm → H-group matching


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_formula(formula: str) -> dict:
    """'C8H8O' → {'C': 8, 'H': 8, 'O': 1}"""
    result: dict[str, int] = {}
    for m in re.finditer(r'([A-Z][a-z]?)(\d*)', formula):
        elem = m.group(1)
        if not elem:
            continue
        count = int(m.group(2)) if m.group(2) else 1
        result[elem] = result.get(elem, 0) + count
    return result


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _build_h_index(atom_nodes: list) -> dict:
    """Return {group_h_ppm: [lsd_id, ...]} for atoms with H."""
    idx: dict[float, list] = defaultdict(list)
    for n in atom_nodes:
        if n.get('n_h', 0) > 0 and n.get('group_h_ppm') is not None:
            idx[n['group_h_ppm']].append(n['lsd_id'])
    return dict(idx)


def _match_h(h_obs: float, h_index: dict) -> list:
    """Return LSD IDs of the H group nearest to h_obs (within _H_PPM_TOL)."""
    best_diff = float('inf')
    best_ids: list = []
    for h_ref, ids in h_index.items():
        diff = abs(h_ref - h_obs)
        if diff < best_diff:
            best_diff = diff
            best_ids = ids
    return best_ids if best_diff <= _H_PPM_TOL else []


def _match_c(c_obs: float, c_nodes: list,
             exclude_ids: Optional[set] = None) -> Optional[int]:
    """
    Return LSD ID of C atom nearest c_obs within 1.5×_C_PPM_TOL.
    Prefers atoms NOT in exclude_ids (avoids self-HMBC).
    When equidistant, non-excluded atoms win.
    """
    exclude_ids = exclude_ids or set()
    candidates: list[tuple] = []
    for n in c_nodes:
        diff = abs(n['c_ppm'] - c_obs)
        if diff <= _C_PPM_TOL * 1.5:
            candidates.append((diff, n['lsd_id']))
    if not candidates:
        return None
    candidates.sort()
    best_diff = candidates[0][0]
    # Prefer non-excluded among atoms tied within 0.1 ppm of the best
    non_excl = [(d, cid) for d, cid in candidates
                if abs(d - best_diff) < 0.1 and cid not in exclude_ids]
    if non_excl:
        return non_excl[0][1]
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Stage 1: HSQC → labeled atom inventory
# ---------------------------------------------------------------------------

def _build_nodes_from_c13(c13_list: list) -> list:
    """
    Build atom-node dicts directly from a 1D 13C spectrum list.

    c13_list: [{"c_ppm": float, "n_h": int, "h_ppm": float|None}, ...]
    One entry per distinct carbon atom — NP-MRD assignment tables already
    separate each C, so no expansion logic is needed.

    Returns nodes sorted by c_ppm descending with 1-based lsd_id assigned.
    """
    nodes: list[dict] = []
    for entry in c13_list:
        c_ppm = entry['c_ppm']
        n_h   = entry.get('n_h', 0)
        h_ppm = entry.get('h_ppm')
        # Use 5.5 ppm threshold (not 6.0) to catch flavonoid A-ring CH at
        # 94–99 ppm / 5.7–6.1 ppm (e.g. catechin C-6, C-8).
        is_aromatic = (h_ppm is not None and h_ppm > 5.5 and c_ppm > 80.0)
        hyb = 2 if (c_ppm > 100.0 or is_aromatic) else 3
        nodes.append({
            'symbol':      'C',
            'hyb':         hyb,
            'n_h':         n_h,
            'c_ppm':       c_ppm,
            'h_ppm':       h_ppm if n_h > 0 else None,
            'is_quat':     (n_h == 0),
            'is_aromatic': is_aromatic,
            'group_c_ppm': c_ppm,
            'group_h_ppm': h_ppm if n_h > 0 else None,
        })
    nodes.sort(key=lambda n: n['c_ppm'], reverse=True)
    for i, node in enumerate(nodes):
        node['lsd_id'] = i + 1
    return nodes


def extract_atom_nodes(problem: dict) -> list:
    """
    Build a list of LSD atom-node dicts from HSQC peaks plus quaternary
    carbon detection from HMBC.

    Each dict contains:
        lsd_id       int    1-based unique identifier used in LSD commands
        symbol       str    element symbol ('C')
        hyb          int    LSD hybridisation code: 1=sp, 2=sp2, 3=sp3
        n_h          int    number of directly attached H
        c_ppm        float  13C chemical shift
        h_ppm        float  representative 1H shift (None for quaternary C)
        is_quat      bool   no directly attached H (not in HSQC)
        is_aromatic  bool   shift suggests aromatic environment
        group_c_ppm  float  original HSQC/HMBC c_ppm for group exclusion
        group_h_ppm  float  original HSQC h_ppm for H-group lookup

    Aromatic CH groups with n_h > 1 are expanded into n_h individual atoms
    (each with n_h=1) because LSD treats each MULT entry as a distinct atom.
    sp2 CH2 (=CH2, n_h=2) is kept as a single atom.
    """
    # Fast path: if a 1D 13C spectrum is provided, use it directly.
    # This bypasses HMBC-based quaternary C detection and formula completion,
    # giving exact atom counts and shifts directly from experimental data.
    c13_list = (problem.get('spectra') or {}).get('c13') or []
    if c13_list:
        return _build_nodes_from_c13(c13_list)

    hsqc = problem['spectra']['hsqc'] or []
    hmbc = problem['spectra'].get('hmbc') or []
    hsqc_c_ppms = [p['c_ppm'] for p in hsqc]

    nodes: list[dict] = []

    # --- Build C nodes from HSQC ---
    for peak in hsqc:
        c_ppm = peak['c_ppm']
        h_ppm = peak['h_ppm']
        n_h   = peak['n_h']

        # Aromatic: h_ppm > 5.5 catches low-ppm aromatics (e.g. quercetin/catechin
        # C-6/C-8 at 94–99 ppm with H at 5.7–6.2 ppm) that would be missed by the
        # c_ppm > 100 rule alone.
        is_aromatic = (h_ppm is not None and h_ppm > 5.5 and c_ppm > 80.0)
        hyb = 2 if (c_ppm > 100.0 or is_aromatic) else 3

        # Max H atoms allowed on a single carbon of this type:
        #   aromatic CH → 1   (ring C with exactly one H)
        #   olefinic sp2 → 2  (=CH2 terminal alkene)
        #   sp3 → 3           (CH3 methyl group)
        # If n_h exceeds this limit, the HSQC peak represents multiple
        # equivalent carbons collapsed to one peak; split them.
        max_nh_single = 1 if is_aromatic else (2 if hyb == 2 else 3)

        if n_h > max_nh_single:
            # Find largest divisor of n_h that is ≤ max_nh_single
            n_h_per = max_nh_single
            while n_h_per >= 1:
                if n_h % n_h_per == 0:
                    break
                n_h_per -= 1
            if n_h_per == 0:
                n_h_per = 1
            n_equiv = n_h // n_h_per
            for _ in range(n_equiv):
                nodes.append({
                    'symbol': 'C', 'hyb': hyb, 'n_h': n_h_per,
                    'c_ppm': c_ppm, 'h_ppm': h_ppm,
                    'is_quat': False, 'is_aromatic': is_aromatic,
                    'group_c_ppm': c_ppm, 'group_h_ppm': h_ppm,
                })
        else:
            nodes.append({
                'symbol': 'C', 'hyb': hyb, 'n_h': n_h,
                'c_ppm': c_ppm, 'h_ppm': h_ppm,
                'is_quat': False, 'is_aromatic': is_aromatic,
                'group_c_ppm': c_ppm, 'group_h_ppm': h_ppm,
            })

    # --- Detect quaternary C from HMBC ---
    # Any HMBC c_ppm not within _C_QUAT_TOL of an HSQC c_ppm → quaternary C.
    # Use tight tolerance (_C_QUAT_TOL=0.25 ppm) to avoid absorbing nearby
    # quat carbons into adjacent CH peaks (e.g. ipso at 130.1 vs CH at 129.8).
    # Cluster nearby quat-C candidates (within _C_QUAT_TOL) to one node.
    clusters: dict[float, set] = {}
    for peak in hmbc:
        c = peak['c_ppm']
        if any(abs(c - hc) <= _C_QUAT_TOL for hc in hsqc_c_ppms):
            continue  # close enough to an HSQC peak → not quaternary
        matched_rep = None
        for rep in clusters:
            if abs(c - rep) <= _C_QUAT_TOL:
                matched_rep = rep
                break
        if matched_rep is None:
            clusters[c] = {c}
        else:
            clusters[matched_rep].add(c)

    # Number of HSQC-derived C atoms (before adding quat nodes).
    n_c_hsqc = len(nodes)

    for cluster in clusters.values():
        c_ppm = sum(cluster) / len(cluster)
        hyb = 2 if c_ppm > 100.0 else 3
        nodes.append({
            'symbol': 'C', 'hyb': hyb, 'n_h': 0,
            'c_ppm': c_ppm, 'h_ppm': None,
            'is_quat': True,
            'is_aromatic': (100.0 < c_ppm < 165.0),
            'group_c_ppm': c_ppm, 'group_h_ppm': None,
        })

    # Formula-based quat-C completion: if HMBC-detected quat C < expected
    # (e.g. two nearly-identical ring C merged by the 0.25-ppm tolerance,
    # like vanillin C3/C4 both at ~148.5 ppm), add placeholder atoms at the
    # shift of the most-populated cluster so LSD has the right atom count.
    mol_formula = problem.get('molecular_formula', '') if isinstance(problem, dict) else ''
    formula_c = _parse_formula(mol_formula).get('C', 0)
    n_quat_expected = max(0, formula_c - n_c_hsqc)
    n_quat_detected = len(clusters)
    if formula_c > 0 and n_quat_detected < n_quat_expected:
        deficit = n_quat_expected - n_quat_detected
        # Only fix the common case of exactly 1 merged cluster (e.g. vanillin
        # C3/C4 both at ~148.5 ppm merged by 0.25-ppm tolerance).  Larger
        # deficits (quercetin, caffeine) arise from incomplete HMBC coverage
        # across many quat-C atoms; adding free-floating atoms in those cases
        # explodes the LSD search space and causes timeouts.
        if deficit == 1:
            sorted_by_count = sorted(
                [(len(vs), rep) for rep, vs in clusters.items()],
                reverse=True,
            )
            if sorted_by_count:
                _, rep = sorted_by_count[0]
                c_ppm = sum(clusters[rep]) / len(clusters[rep])
            else:
                c_ppm = 130.0  # fallback aromatic C shift
            hyb = 2 if c_ppm > 100.0 else 3
            nodes.append({
                'symbol': 'C', 'hyb': hyb, 'n_h': 0,
                'c_ppm': c_ppm, 'h_ppm': None,
                'is_quat': True,
                'is_aromatic': (100.0 < c_ppm < 165.0),
                'group_c_ppm': c_ppm, 'group_h_ppm': None,
            })

    # Sort by c_ppm descending, assign 1-based LSD IDs
    nodes.sort(key=lambda n: n['c_ppm'], reverse=True)
    for i, n in enumerate(nodes):
        n['lsd_id'] = i + 1
    return nodes


# ---------------------------------------------------------------------------
# Stage 2: COSY + HMBC → connectivity constraints
# ---------------------------------------------------------------------------

def extract_constraints(problem: dict, atom_nodes: list) -> dict:
    """
    Translate COSY and HMBC peaks into LSD constraint pairs.

    HMBC strategy: one constraint per NMR peak (representative first atom
    from the H group), excluding atoms in the same HSQC group as the H
    to prevent spurious self-correlations.

    COSY strategy: pair atoms from the two H groups using zip-with-cycling
    so that a group of size m pairs with a group of size n by repeating
    the shorter list (ensures all atoms in the larger group get a partner).

    Returns:
        {
            'hmbc': [(h_lsd_id, c_lsd_id), ...],
            'cosy': [(id1, id2), ...],          # canonical (min, max) pairs
        }
    """
    cosy_peaks = problem['spectra'].get('cosy') or []
    hmbc_peaks = problem['spectra'].get('hmbc') or []

    c_nodes = [n for n in atom_nodes if n['symbol'] == 'C']
    h_index = _build_h_index(atom_nodes)

    # Map group_c_ppm → set of lsd_ids (for HMBC group exclusion)
    group_c_to_ids: dict[float, set] = defaultdict(set)
    for n in atom_nodes:
        if n.get('group_c_ppm') is not None:
            group_c_to_ids[n['group_c_ppm']].add(n['lsd_id'])

    id_to_node: dict[int, dict] = {n['lsd_id']: n for n in atom_nodes}

    # --- HMBC ---
    # Apply each HMBC peak to ALL atoms in the matching H-group (not just the
    # representative).  For equivalent aromatic CH groups (e.g. the two ortho
    # CH atoms in a para-disubstituted benzene), every member must be
    # constrained identically; using only the representative leaves its
    # partner unconstrained and explodes the LSD solution count.
    hmbc_set: set[tuple] = set()
    for peak in hmbc_peaks:
        h_ids = _match_h(peak['h_ppm'], h_index)
        if not h_ids:
            continue
        for h_id in sorted(h_ids):
            h_node = id_to_node.get(h_id)
            if h_node is None:
                continue
            gc = h_node.get('group_c_ppm')
            same_group = group_c_to_ids.get(gc, {h_id}) if gc is not None else {h_id}
            c_id = _match_c(peak['c_ppm'], c_nodes, exclude_ids=same_group)
            if c_id is not None and c_id not in same_group:
                hmbc_set.add((h_id, c_id))

    # --- COSY ---
    cosy_set: set[tuple] = set()
    for peak in cosy_peaks:
        a_ids = _match_h(peak['h1_ppm'], h_index)
        b_ids = _match_h(peak['h2_ppm'], h_index)
        if not a_ids or not b_ids:
            continue
        sa = sorted(a_ids)
        sb = sorted(b_ids)

        # Skip COSY if either side is a multi-atom non-aromatic group.
        # Such groups arise from equivalent sp3 CH2/CH3 carbons collapsed
        # into one HSQC peak (e.g. two isopropyl CH3 at the same shift →
        # n_h=6 → split into 2 atoms).  Assigning the same COSY partner to
        # ALL atoms in the group would over-constrain each CH2/CH3 atom and
        # trigger LSD error 283 ("max number of neighbors exceeded").
        # Aromatic multi-atom groups (ortho pairs etc.) are fine: each atom
        # genuinely bonds to the same neighbouring aromatic CH.
        def _is_multi_nonaromatic(ids: list) -> bool:
            return len(ids) > 1 and not all(id_to_node[i]['is_aromatic'] for i in ids)

        if _is_multi_nonaromatic(sa) or _is_multi_nonaromatic(sb):
            continue

        longer, shorter = (sa, sb) if len(sa) >= len(sb) else (sb, sa)
        shorter_cyc = list(islice(cycle(shorter), len(longer)))
        for a, b in zip(longer, shorter_cyc):
            if a != b:
                cosy_set.add((min(a, b), max(a, b)))

    return {'hmbc': list(hmbc_set), 'cosy': list(cosy_set)}


# ---------------------------------------------------------------------------
# Stage 3: Shift-range heuristics → heteroatom placement
# ---------------------------------------------------------------------------

def detect_fragments(atom_nodes: list, molecular_formula: str) -> dict:
    """
    Heuristic analysis of chemical shifts + molecular formula to place
    heteroatom (O, N, S) nodes and add high-confidence BOND constraints.

    Rules
    -----
    Oxygen
        • For each quaternary C with c_ppm > 160 (carbonyl/heteroaromatic):
          add one sp2 O (n_h=0) and BOND it to that C.
        • Remaining O: sp3, n_h from unaccounted formula H.
    Nitrogen
        • sp2 (n_h=0) if aromatic C atoms are present (assume heteroaromatic
          or aromatic amide; covers pyridine, imidazole, caffeine, etc.).
        • sp3 otherwise, n_h from unaccounted formula H (secondary amine default).
    Sulfur
        • sp3, n_h from unaccounted formula H.

    Returns:
        {
            'extra_atoms':       [atom_node_dict, ...],
            'bond_constraints':  [(lsd_id1, lsd_id2), ...],
            'elim':              (n_elim, p_elim),
        }
    """
    formula = _parse_formula(molecular_formula)
    n_o = formula.get('O', 0)
    n_n = formula.get('N', 0)
    n_s = formula.get('S', 0)

    extra_atoms: list[dict] = []
    bond_constraints: list[tuple] = []
    next_id = (max(n['lsd_id'] for n in atom_nodes) + 1) if atom_nodes else 1

    # H balance: heteroatom n_h accounts for H not on carbon
    total_h  = formula.get('H', 0)
    h_on_c   = sum(n['n_h'] for n in atom_nodes)
    remaining_h = max(0, total_h - h_on_c)

    # Does the molecule have aromatic atoms?
    has_aromatic = any(n.get('is_aromatic') for n in atom_nodes)

    # Quaternary C candidates for sp2 =O (carbonyl) placement:
    # Use c_ppm > 163 to catch ketones (~190-215), aldehydes (~195-205),
    # esters/lactones (~163-185), while excluding aryl-ether C (~145-162)
    # which has a sigma C-O bond (sp3 O), NOT a carbonyl double bond.
    # Also include sp2 CH with c_ppm > 163 to catch aldehyde C (CHO group:
    # n_h=1, c_ppm ~190-200 ppm) which is NOT quaternary (it has one H) but
    # still requires a sp2 =O partner.
    carbonyl_nodes = sorted(
        [n for n in atom_nodes
         if n['hyb'] == 2 and n['c_ppm'] > 163.0
         and (n['is_quat'] or n.get('n_h', 0) == 1)],
        key=lambda n: n['c_ppm'], reverse=True,
    )

    # --- Oxygen ---
    assigned_o = 0
    for cn in carbonyl_nodes:
        if assigned_o >= n_o:
            break
        extra_atoms.append({
            'lsd_id': next_id, 'symbol': 'O', 'hyb': 2, 'n_h': 0,
            'c_ppm': None, 'h_ppm': None,
            'is_quat': False, 'is_aromatic': False,
            'group_c_ppm': None, 'group_h_ppm': None,
        })
        bond_constraints.append((cn['lsd_id'], next_id))
        next_id += 1
        assigned_o += 1

    # --- Nitrogen (assigned before remaining O) ---
    # N is assigned BEFORE remaining O so that amine H are reserved for N.
    # For p-nitroaniline (C6H6N2O2): remaining_h=2 after aromatic C, so
    # N1 (amino N) correctly gets sp3 n_h=2 and remaining_h→0, leaving
    # the nitro N (N2) to be sp2 n_h=0 and both O atoms with no H.
    for _ in range(n_n):
        if remaining_h >= 2:
            # Primary amine: NH2 (aniline, p-nitroaniline, PABA, etc.)
            hyb_n, n_h_n = 3, 2
            remaining_h -= 2
        elif remaining_h == 1:
            # Secondary amine: NH (indole, pyrrole, etc.)
            hyb_n, n_h_n = 3, 1
            remaining_h -= 1
        elif has_aromatic:
            # No H left: sp2 N (pyridine, nitro, imide, isoquinoline, etc.)
            hyb_n, n_h_n = 2, 0
        else:
            # Non-aromatic, no H (tertiary amine, quaternary N, etc.)
            hyb_n, n_h_n = 3, 0
        extra_atoms.append({
            'lsd_id': next_id, 'symbol': 'N', 'hyb': hyb_n, 'n_h': n_h_n,
            'c_ppm': None, 'h_ppm': None,
            'is_quat': False, 'is_aromatic': False,
            'group_c_ppm': None, 'group_h_ppm': None,
        })
        next_id += 1

    # --- Remaining Oxygen (after N has taken its H) ---
    for _ in range(n_o - assigned_o):
        o_h = 1 if remaining_h > 0 else 0
        if o_h:
            remaining_h -= 1
        extra_atoms.append({
            'lsd_id': next_id, 'symbol': 'O', 'hyb': 3, 'n_h': o_h,
            'c_ppm': None, 'h_ppm': None,
            'is_quat': False, 'is_aromatic': False,
            'group_c_ppm': None, 'group_h_ppm': None,
        })
        next_id += 1

    # --- Sulfur ---
    for _ in range(n_s):
        s_h = 1 if remaining_h > 0 else 0
        if s_h:
            remaining_h -= 1
        extra_atoms.append({
            'lsd_id': next_id, 'symbol': 'S', 'hyb': 3, 'n_h': s_h,
            'c_ppm': None, 'h_ppm': None,
            'is_quat': False, 'is_aromatic': False,
            'group_c_ppm': None, 'group_h_ppm': None,
        })
        next_id += 1

    # -------------------------------------------------------------------
    # Post-processing: ensure even sp2 count and correct H balance.
    #
    # LSD requires:
    #   (a) total sp2 atom count is even (each π-bond pairs two sp2 atoms)
    #   (b) total valence = 2 × bonds = even (sum of element valences + H counts)
    #
    # Two common failures:
    #   • Ester/lactone/COOH C=O at 155–175 ppm: our carbonyl threshold (>175)
    #     misses these, leaving sp2 C without a sp2 O partner → odd sp2 count.
    #   • Pyrrole/amine N assigned sp2(n_h=0) instead of sp3 with its H atoms
    #     → H deficit in the molecule → odd total valence.
    # -------------------------------------------------------------------
    all_atoms = atom_nodes + extra_atoms
    n_h_formula = formula.get('H', 0)
    h_on_atoms  = sum(a['n_h'] for a in all_atoms)
    sp2_count   = sum(1 for a in all_atoms if a['hyb'] == 2)

    # Step 1 — H balance: if atoms have fewer H than the formula requires,
    # convert sp2 N (n_h=0) atoms to sp3 and give them the missing H.
    # This fixes pyrrole-type N and primary/secondary aromatic amines.
    h_needed = n_h_formula - h_on_atoms
    if h_needed > 0:
        for ea in extra_atoms:
            if h_needed <= 0:
                break
            if ea['symbol'] == 'N' and ea['hyb'] == 2 and ea['n_h'] == 0:
                give = min(h_needed, 2)   # at most 2 H per N (primary amine)
                ea['n_h'] = give
                ea['hyb'] = 3             # sp3: N has H, no formal C=N double bond
                h_on_atoms += give
                h_needed   -= give
                sp2_count  -= 1

    # Step 2 — sp2 parity: if sp2 count is odd, find the highest-ppm
    # unassigned quat sp2 C (likely an ester/lactone/COOH carbonyl at
    # 150–175 ppm) and upgrade one sp3 O to sp2 O with a BOND constraint.
    if sp2_count % 2 == 1:
        bonded_ids = (
            {id1 for id1, id2 in bond_constraints}
            | {id2 for id1, id2 in bond_constraints}
        )
        candidate_cs = sorted(
            [n for n in atom_nodes
             if n['symbol'] == 'C' and n['is_quat'] and n['hyb'] == 2
             and n['c_ppm'] > 150.0 and n['lsd_id'] not in bonded_ids],
            key=lambda n: n['c_ppm'], reverse=True,
        )
        # Sort sp3 O atoms: prefer n_h=0 (ether O) over n_h=1 (OH)
        # to avoid unnecessarily removing a hydroxyl proton.
        sp3_os = sorted(
            [ea for ea in extra_atoms
             if ea['symbol'] == 'O' and ea['hyb'] == 3],
            key=lambda a: a['n_h'],
        )
        if candidate_cs and sp3_os:
            # C=O pairing: BOND highest-ppm eligible C to a sp3 O.
            cn = candidate_cs[0]
            oa = sp3_os[0]
            freed_h = oa['n_h']
            oa['hyb'] = 2
            oa['n_h'] = 0
            bond_constraints.append((cn['lsd_id'], oa['lsd_id']))
            sp2_count += 1
            # Redistribute any freed H to sp3 N that could use more
            # (e.g. amino N that only got 1 H in Step 1 but needs 2).
            if freed_h > 0:
                for ea in extra_atoms:
                    if freed_h == 0:
                        break
                    if ea['symbol'] == 'N' and ea['hyb'] == 3:
                        ea['n_h'] += freed_h
                        freed_h = 0
        elif sp3_os:
            # Fallback N=O pairing: no eligible high-ppm C (e.g. nitro group
            # where ring C is below 150 ppm).  Bond a sp2 N (n_h=0) to a
            # sp3 O to form the first N=O of the nitro/nitroso group.
            sp2_ns = [ea for ea in extra_atoms
                      if ea['symbol'] == 'N' and ea['hyb'] == 2 and ea['n_h'] == 0]
            if sp2_ns:
                na = sp2_ns[0]
                oa = sp3_os[0]
                freed_h = oa['n_h']
                oa['hyb'] = 2
                oa['n_h'] = 0
                bond_constraints.append((na['lsd_id'], oa['lsd_id']))
                sp2_count += 1
                if freed_h > 0:
                    for ea in extra_atoms:
                        if freed_h == 0:
                            break
                        if ea['symbol'] == 'N' and ea['hyb'] == 3:
                            ea['n_h'] += freed_h
                            freed_h = 0

    return {
        'extra_atoms': extra_atoms,
        'bond_constraints': bond_constraints,
        'elim': (2, 4),
    }


# ---------------------------------------------------------------------------
# Stage 4: LSD binary interface
# ---------------------------------------------------------------------------

def _build_lsd_text(atom_nodes: list, extra_atoms: list,
                    constraints: dict, bond_constraints: list,
                    elim: tuple) -> str:
    """Render LSD input file as a string."""
    all_atoms = sorted(atom_nodes + extra_atoms, key=lambda a: a['lsd_id'])
    lines = ["; Generated by nmrsim solve_structure.py"]

    for a in all_atoms:
        lines.append(f"MULT {a['lsd_id']} {a['symbol']} {a['hyb']} {a['n_h']}")

    for a in all_atoms:
        if a['symbol'] == 'C' and a['n_h'] > 0:
            lines.append(f"HMQC {a['lsd_id']} {a['lsd_id']}")

    # LSD HMBC format: HMBC {c_id} {h_id}  (carbon first, H-bearing atom second)
    # The H atom must be HMQC-declared; quaternary carbons go in the first slot.
    for h_id, c_id in sorted(constraints['hmbc']):
        lines.append(f"HMBC {c_id} {h_id}")

    for id1, id2 in sorted(constraints['cosy']):
        lines.append(f"COSY {id1} {id2}")

    for id1, id2 in sorted(bond_constraints):
        lines.append(f"BOND {id1} {id2}")

    n_elim, p_elim = elim
    if n_elim > 0:
        lines.append(f"ELIM {n_elim} {p_elim}")

    lines.append("EXIT")
    return "\n".join(lines) + "\n"


def run_lsd(atom_nodes: list, constraints: dict, fragments: dict,
            molecular_formula: str = "", lsd_timeout: int = 60) -> tuple:
    """
    Stage 4: write LSD input file, run LSD binary, parse SMILES.

    Returns
    -------
    (smiles_list, lsd_text) where smiles_list is a deduplicated list of
    canonical SMILES strings found by LSD, and lsd_text is the generated
    LSD input (for debugging).

    Notes
    -----
    * LSD exits with code 1 even on success; check for .sol file existence.
    * outlsd mode 5 outputs one SMILES per line (one per solution).
    """
    extra_atoms     = fragments.get('extra_atoms', [])
    bond_constr     = fragments.get('bond_constraints', [])
    elim            = fragments.get('elim', (2, 4))

    lsd_text = _build_lsd_text(atom_nodes, extra_atoms, constraints,
                                bond_constr, elim)

    with tempfile.TemporaryDirectory() as tmpdir:
        lsd_path = os.path.join(tmpdir, "input.lsd")
        sol_path = os.path.join(tmpdir, "input.sol")

        with open(lsd_path, 'w') as fh:
            fh.write(lsd_text)

        try:
            subprocess.run(
                [_LSD_BIN, lsd_path],
                capture_output=True, text=True,
                timeout=lsd_timeout, cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return [], lsd_text

        if not os.path.exists(sol_path):
            return [], lsd_text

        try:
            with open(sol_path, 'rb') as fh:
                outlsd_result = subprocess.run(
                    [_OUTLSD_BIN, "5"],
                    stdin=fh,
                    capture_output=True, text=True,
                    timeout=30,
                )
        except subprocess.TimeoutExpired:
            return [], lsd_text

        smiles_raw = [
            line.strip()
            for line in outlsd_result.stdout.splitlines()
            if line.strip() and not line.startswith('#')
        ]

    seen: set[str] = set()
    smiles_list: list[str] = []
    for s in smiles_raw:
        canon = _canonicalize_smiles(s)
        if canon and canon not in seen:
            seen.add(canon)
            smiles_list.append(canon)

    return smiles_list, lsd_text


def run_lsd_iterative(atom_nodes: list, constraints: dict, fragments: dict,
                      molecular_formula: str = "",
                      lsd_timeout: int = 60) -> tuple:
    """
    Run LSD with progressively relaxed ELIM parameters until solutions appear.

    Pass schedule
    -------------
    1. strict            — ELIM  2 4  (default; catches well-constrained molecules)
    2. elim5             — ELIM  5 4  (5 unsatisfied correlations allowed)
    3. elim10            — ELIM 10 4  (further relaxation)
    4. elim10_nocosy     — ELIM 10 4, COSY constraints dropped
    5. elim20_nocosy_p3  — ELIM 20 3, COSY constraints dropped  (max relaxation)

    Dropping COSY in passes 4-5 handles over-constrained sp3 chains where
    COSY constraints conflict with contradictory HMBC evidence.

    Returns
    -------
    (smiles_list, lsd_text, pass_label)
        smiles_list : deduplicated canonical SMILES (empty list if all passes fail)
        lsd_text    : LSD input file from the *last attempted* pass (for debugging)
        pass_label  : str identifying the successful pass, or 'failed'
    """
    passes = [
        (2,  4, False, "strict"),
        (5,  4, False, "elim5"),
        (10, 4, False, "elim10"),
        (10, 4, True,  "elim10_nocosy"),
        (20, 3, True,  "elim20_nocosy_p3"),
    ]
    last_text = ""
    for elim_n, elim_p, drop_cosy, label in passes:
        frags_mod = {**fragments, 'elim': (elim_n, elim_p)}
        cons_mod  = {**constraints, 'cosy': [] if drop_cosy else constraints['cosy']}
        smiles, lsd_text = run_lsd(
            atom_nodes, cons_mod, frags_mod, molecular_formula, lsd_timeout,
        )
        last_text = lsd_text
        if smiles:
            return smiles, lsd_text, label
    return [], last_text, "failed"


# ---------------------------------------------------------------------------
# Stage 5: Rank by 13C shift MAE
# ---------------------------------------------------------------------------

def rank_candidates(smiles_list: list, atom_nodes: list,
                    problem: dict) -> list:
    """
    Score each candidate SMILES by 13C shift MAE using the Hungarian
    (linear_sum_assignment) algorithm for optimal bijective matching.

    Observed shifts come from the atom_nodes (HSQC unique c_ppm values).
    Predicted shifts come from NMRShiftDB2 via predict_shifts().

    Returns list sorted by ascending score (best first):
        [{'smiles': str, 'score': float, 'rank': int}, ...]
    """
    obs = sorted({n['c_ppm'] for n in atom_nodes if n['symbol'] == 'C'})

    results: list[dict] = []
    _consecutive_failures = 0
    _network_ok = True
    for smiles in smiles_list:
        if not _network_ok:
            # Network is down — skip remaining HTTP calls rather than timing out.
            results.append({'smiles': smiles, 'score': None, 'rank': 0})
            continue
        try:
            pred_data = predict_shifts(smiles, backend='nmrshiftdb')
            pred = sorted(set(pred_data['c_shifts'].values()))
            _consecutive_failures = 0
        except BackendError:
            _consecutive_failures += 1
            if _consecutive_failures >= 3:
                _network_ok = False
            results.append({'smiles': smiles, 'score': None, 'rank': 0})
            continue

        if not obs or not pred:
            results.append({'smiles': smiles, 'score': float('inf'), 'rank': 0})
            continue

        cost = np.abs(np.array(obs)[:, None] - np.array(pred)[None, :])
        ri, ci = scipy.optimize.linear_sum_assignment(cost)
        mae = float(cost[ri, ci].mean())
        results.append({'smiles': smiles, 'score': mae, 'rank': 0})

    results.sort(key=lambda x: (x['score'] is None, x['score'] or 0.0))
    for i, r in enumerate(results):
        r['rank'] = i + 1
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve(problem: dict, max_candidates: int = 50) -> dict:
    """
    Full inverse solver: peak lists → ranked candidate SMILES.

    Parameters
    ----------
    problem : dict
        Benchmark problem dict (needs 'spectra' + 'molecular_formula').
        The 'ground_truth' field is NOT required.
    max_candidates : int
        Maximum candidates to rank by 13C MAE. If LSD returns more
        solutions, they are listed unranked (score=None).

    Returns
    -------
    {
        "candidates":  [{"smiles": str, "score": float|None, "rank": int}],
        "lsd_count":   int,    # total unique solutions from LSD
        "difficulty":  float,  # log10(lsd_count); 0 = unique solution
        "lsd_file":    str,    # LSD input file content (for debugging)
    }
    """
    formula     = problem.get('molecular_formula', '')
    atom_nodes  = extract_atom_nodes(problem)
    constraints = extract_constraints(problem, atom_nodes)
    fragments   = detect_fragments(atom_nodes, formula)

    smiles_list, lsd_text = run_lsd(atom_nodes, constraints, fragments, formula)
    lsd_count  = len(smiles_list)
    difficulty = math.log10(max(lsd_count, 1))

    if lsd_count == 0:
        return {
            "candidates": [],
            "lsd_count":  0,
            "difficulty": difficulty,
            "lsd_file":   lsd_text,
        }

    if lsd_count <= max_candidates:
        candidates = rank_candidates(smiles_list, atom_nodes, problem)
    else:
        candidates = [
            {"smiles": s, "score": None, "rank": i + 1}
            for i, s in enumerate(smiles_list[:max_candidates])
        ]

    return {
        "candidates": candidates,
        "lsd_count":  lsd_count,
        "difficulty": difficulty,
        "lsd_file":   lsd_text,
    }
