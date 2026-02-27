"""
build_2d.py — Stage 2: HSQC / HMBC / COSY peak lists from shift predictions.

All logic follows NMR_SKILL.md Sections 1–3.
"""

import random as _random
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem

# Standard deviation (ppm) for the per-proton diastereotopic split offset.
# Each proton in a diastereotopic pair gets ±|N(0, σ)|, so the total H–H
# separation is ~2× this value on average.
_DIAST_SIGMA = 0.25


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


def _get_diastereotopic_ch2_pairs(mol_with_hs) -> set:
    """
    Return set of frozenset({h1_idx, h2_idx}) for diastereotopic CH₂ groups.

    A pair is diastereotopic when:
    - Both H atoms sit on the same carbon (CH₂ — exactly 2 H neighbors)
    - They share the same canonical rank without chirality (currently treated
      as equivalent homotopic protons)
    - But get different ranks when chirality is included (i.e., they are
      distinguishable due to a nearby stereocenter or prochiral centre)
    """
    ranks_nochir = list(Chem.CanonicalRankAtoms(mol_with_hs, breakTies=False))
    ranks_chiral = list(Chem.CanonicalRankAtoms(
        mol_with_hs, breakTies=False, includeChirality=True))
    pairs: set = set()
    for c_atom in mol_with_hs.GetAtoms():
        if c_atom.GetAtomicNum() != 6:
            continue
        h_nbrs = [n.GetIdx() for n in c_atom.GetNeighbors()
                  if n.GetAtomicNum() == 1]
        if len(h_nbrs) != 2:
            continue
        h1, h2 = h_nbrs
        # Already in separate groups without chirality → not our concern
        if ranks_nochir[h1] != ranks_nochir[h2]:
            continue
        # Different ranks with chirality → diastereotopic
        if ranks_chiral[h1] != ranks_chiral[h2]:
            pairs.add(frozenset({h1, h2}))
    return pairs


def apply_diastereotopic_splits(shift_result: dict,
                                random_seed: int = 42) -> dict:
    """
    Return a modified shift_result with diastereotopic CH₂ protons split.

    For each diastereotopic pair (h1, h2) the two protons receive offsets
    +δ and −δ (where δ ~ |N(0, _DIAST_SIGMA)|), so every downstream spectrum
    (HSQC, HMBC, COSY) sees the same individual H shifts.

    The modified dict contains:
    - 'h_shifts': updated per-atom H shifts
    - '_split_atoms': set of atom indices that received individual offsets
      (used by build functions to place them in singleton H groups)

    Pairs are processed in deterministic order (sorted by min(h1, h2)) so
    the same seed always produces the same split for a given molecule.
    """
    mol = shift_result["mol_with_hs"]
    h_shifts = shift_result["h_shifts"]

    diast_pairs = _get_diastereotopic_ch2_pairs(mol)
    rng = _random.Random(random_seed)

    h_shifts_eff = dict(h_shifts)
    split_atoms: set = set()

    for pair in sorted(diast_pairs, key=lambda p: min(p)):
        h1, h2 = sorted(pair)          # deterministic: lower idx first
        base = h_shifts.get(h1, h_shifts.get(h2))
        if base is None:
            continue
        delta = abs(rng.gauss(0, _DIAST_SIGMA))
        h_shifts_eff[h1] = base + delta
        h_shifts_eff[h2] = base - delta
        split_atoms.add(h1)
        split_atoms.add(h2)

    result = dict(shift_result)
    result["h_shifts"]     = h_shifts_eff
    result["_split_atoms"] = split_atoms
    return result


def _get_h_groups_with_splits(mol_with_hs, split_atoms: set) -> dict:
    """
    Like _get_equivalent_h_groups but places each split atom into its own
    singleton group so it acts as an independent NMR signal.
    """
    ranks = list(Chem.CanonicalRankAtoms(mol_with_hs, breakTies=False))
    max_rank = max(ranks) + 1
    groups: dict[int, list] = defaultdict(list)
    counter = max_rank
    for atom in mol_with_hs.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        idx = atom.GetIdx()
        if idx in split_atoms:
            groups[counter].append(idx)  # unique group per split proton
            counter += 1
        else:
            groups[ranks[idx]].append(idx)
    return dict(groups)


def _representative(group_idxs: list) -> int:
    """Canonical representative of an equivalence group = lowest index."""
    return min(group_idxs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_hsqc(shift_result: dict, random_seed: int = 42) -> list:
    """
    Build HSQC peak list from shift predictions.

    HSQC: one-bond H–C correlations.
    One peak per unique H equivalence group that is bonded to a carbon.
    n_h = number of equivalent H in the group.

    Diastereotopic CH₂ protons receive a small random ±δ split so they
    appear as two peaks at the same ¹³C but slightly different ¹H shifts,
    matching real HSQC behaviour for chiral molecules.

    Returns:
        [{"h_ppm": float, "c_ppm": float, "n_h": int}, ...]
        Sorted by h_ppm ascending.
    """
    sr       = apply_diastereotopic_splits(shift_result, random_seed)
    mol      = sr["mol_with_hs"]
    h_shifts = sr["h_shifts"]
    c_shifts = sr["c_shifts"]
    split_atoms = sr.get("_split_atoms", set())

    dist     = _get_graph_distances(mol)
    h_groups = _get_h_groups_with_splits(mol, split_atoms)

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


def build_hmbc(shift_result: dict, random_seed: int = 42) -> list:
    """
    Build HMBC peak list (2–3 bond H–C correlations).

    For each unique H equivalence group, find all C atoms at graph distance
    2 or 3. Exclude the 1-bond partner (HSQC peak).

    Diastereotopic CH₂ protons appear at their individually split H shifts
    (same offsets as in build_hsqc for the same random_seed).

    Returns:
        [{"h_ppm": float, "c_ppm": float}, ...]
        Sorted by h_ppm ascending, then c_ppm ascending.
    """
    sr       = apply_diastereotopic_splits(shift_result, random_seed)
    mol      = sr["mol_with_hs"]
    h_shifts = sr["h_shifts"]
    c_shifts = sr["c_shifts"]
    split_atoms = sr.get("_split_atoms", set())

    dist     = _get_graph_distances(mol)
    h_groups = _get_h_groups_with_splits(mol, split_atoms)

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


def build_cosy(shift_result: dict, random_seed: int = 42) -> list:
    """
    Build COSY peak list (2–3 bond H–H correlations).

    For each pair of H equivalence groups at distance 2 or 3, emit both
    (h1_ppm, h2_ppm) and (h2_ppm, h1_ppm). Diagonal peaks excluded.
    Pairs within the same equivalence group excluded.

    Diastereotopic CH₂ protons receive the same per-atom ±δ offsets as in
    build_hsqc / build_hmbc (same random_seed → same shifts).

    Returns:
        [{"h1_ppm": float, "h2_ppm": float}, ...]
        Sorted by h1_ppm ascending, then h2_ppm ascending.
    """
    sr       = apply_diastereotopic_splits(shift_result, random_seed)
    mol      = sr["mol_with_hs"]
    h_shifts = sr["h_shifts"]
    split_atoms = sr.get("_split_atoms", set())

    dist     = _get_graph_distances(mol)
    h_groups = _get_h_groups_with_splits(mol, split_atoms)

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
