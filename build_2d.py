"""
build_2d.py — Stage 2: HSQC / HMBC / COSY peak lists from shift predictions.

All logic follows NMR_SKILL.md Sections 1–3.
"""

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_graph_distances(mol_with_hs) -> np.ndarray:
    """Return integer distance matrix (number of bonds) for all atom pairs."""
    dm = rdmolops.GetDistanceMatrix(mol_with_hs)
    return dm.astype(int)


def _get_equivalent_h_groups(mol_with_hs) -> dict:
    """
    Return dict: {canonical_rank: [h_atom_idx, ...]}

    Atoms sharing the same canonical rank (with breakTies=False) are
    chemically equivalent and produce a single NMR signal.
    """
    ranks = list(Chem.CanonicalRankAtoms(mol_with_hs, breakTies=False))
    groups: dict[int, list] = defaultdict(list)
    for atom in mol_with_hs.GetAtoms():
        if atom.GetAtomicNum() == 1:
            groups[ranks[atom.GetIdx()]].append(atom.GetIdx())
    return dict(groups)


def _representative(group_idxs: list) -> int:
    """Canonical representative of an equivalence group = lowest index."""
    return min(group_idxs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_hsqc(shift_result: dict) -> list:
    """
    Build HSQC peak list from shift predictions.

    HSQC: one-bond H–C correlations.
    One peak per unique H equivalence group that is bonded to a carbon.
    n_h = number of equivalent H in the group.

    Returns:
        [{"h_ppm": float, "c_ppm": float, "n_h": int}, ...]
        Sorted by h_ppm ascending.
    """
    mol   = shift_result["mol_with_hs"]
    h_shifts = shift_result["h_shifts"]
    c_shifts = shift_result["c_shifts"]

    dist = _get_graph_distances(mol)
    h_groups = _get_equivalent_h_groups(mol)

    peaks = []
    seen_peaks = set()

    for rank, h_idxs in h_groups.items():
        rep_h = _representative(h_idxs)

        # Find the C atom 1 bond away from the representative H
        bonded_c = None
        for atom in mol.GetAtomWithIdx(rep_h).GetNeighbors():
            if atom.GetAtomicNum() == 6 and dist[rep_h][atom.GetIdx()] == 1:
                bonded_c = atom.GetIdx()
                break

        if bonded_c is None:
            continue  # H not attached to carbon (e.g. OH, NH)

        h_ppm = h_shifts.get(rep_h)
        c_ppm = c_shifts.get(bonded_c)

        if h_ppm is None or c_ppm is None:
            continue  # No shift prediction for this atom

        key = (round(h_ppm, 4), round(c_ppm, 4))
        if key in seen_peaks:
            continue
        seen_peaks.add(key)

        peaks.append({
            "h_ppm": round(h_ppm, 4),
            "c_ppm": round(c_ppm, 4),
            "n_h":   len(h_idxs),
        })

    return sorted(peaks, key=lambda p: p["h_ppm"])


def build_hmbc(shift_result: dict) -> list:
    """
    Build HMBC peak list (2–3 bond H–C correlations).

    For each unique H equivalence group, find all C atoms at graph distance
    2 or 3. Exclude the 1-bond partner (HSQC peak).

    Returns:
        [{"h_ppm": float, "c_ppm": float}, ...]
        Sorted by h_ppm ascending, then c_ppm ascending.
    """
    mol      = shift_result["mol_with_hs"]
    h_shifts = shift_result["h_shifts"]
    c_shifts = shift_result["c_shifts"]

    dist    = _get_graph_distances(mol)
    h_groups = _get_equivalent_h_groups(mol)

    peaks = []
    seen_peaks = set()

    for _rank, h_idxs in h_groups.items():
        rep_h = _representative(h_idxs)
        h_ppm = h_shifts.get(rep_h)
        if h_ppm is None:
            continue

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            c_idx = atom.GetIdx()
            d = dist[rep_h][c_idx]
            if d not in (2, 3):
                continue  # Not 2 or 3 bonds away

            c_ppm = c_shifts.get(c_idx)
            if c_ppm is None:
                continue

            key = (round(h_ppm, 4), round(c_ppm, 4))
            if key in seen_peaks:
                continue
            seen_peaks.add(key)

            peaks.append({
                "h_ppm": round(h_ppm, 4),
                "c_ppm": round(c_ppm, 4),
            })

    return sorted(peaks, key=lambda p: (p["h_ppm"], p["c_ppm"]))


def build_cosy(shift_result: dict) -> list:
    """
    Build COSY peak list (2–3 bond H–H correlations).

    For each pair of H equivalence groups at distance 2 or 3, emit both
    (h1_ppm, h2_ppm) and (h2_ppm, h1_ppm). Diagonal peaks excluded.
    Pairs within the same equivalence group excluded.

    Returns:
        [{"h1_ppm": float, "h2_ppm": float}, ...]
        Sorted by h1_ppm ascending, then h2_ppm ascending.
    """
    mol      = shift_result["mol_with_hs"]
    h_shifts = shift_result["h_shifts"]

    dist     = _get_graph_distances(mol)
    h_groups = _get_equivalent_h_groups(mol)

    group_list = list(h_groups.values())
    peaks = []
    seen_pairs = set()

    for i in range(len(group_list)):
        for j in range(len(group_list)):
            if i == j:
                continue  # Same equivalence group → no COSY cross-peak

            rep_a = _representative(group_list[i])
            rep_b = _representative(group_list[j])

            d = dist[rep_a][rep_b]
            if d not in (2, 3):
                continue

            h1_ppm = h_shifts.get(rep_a)
            h2_ppm = h_shifts.get(rep_b)
            if h1_ppm is None or h2_ppm is None:
                continue

            key = (round(h1_ppm, 4), round(h2_ppm, 4))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            peaks.append({
                "h1_ppm": round(h1_ppm, 4),
                "h2_ppm": round(h2_ppm, 4),
            })

    return sorted(peaks, key=lambda p: (p["h1_ppm"], p["h2_ppm"]))
