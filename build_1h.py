"""
build_1h.py — Stage 3: 1H 1D NMR multiplet table (Phase 2).

See NMR_SKILL.md Section 4 for the full specification.
"""

import random
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdmolops


# ---------------------------------------------------------------------------
# J-coupling table (NMR_SKILL.md Section 4.2)
# ---------------------------------------------------------------------------

J_COUPLING_TABLE = {
    # Aromatic systems
    "aromatic_ortho":       {"mu": 8.0,  "sigma": 0.3},
    "aromatic_meta":        {"mu": 2.0,  "sigma": 0.2},
    "aromatic_para":        {"mu": 0.5,  "sigma": 0.1},
    "aromatic_5ring_34":    {"mu": 3.4,  "sigma": 0.3},
    "aromatic_5ring_23":    {"mu": 1.8,  "sigma": 0.2},

    # Alkene (vinyl)
    "vinyl_cis":            {"mu": 11.5, "sigma": 0.5},
    "vinyl_trans":          {"mu": 17.0, "sigma": 0.5},
    "vinyl_geminal":        {"mu": 2.0,  "sigma": 0.3},
    "vinyl_avg":            {"mu": 14.0, "sigma": 3.0},   # cis/trans unknown without 3D

    # Saturated (sp3)
    "vicinal_acyclic":      {"mu": 7.0,  "sigma": 0.5},
    "vicinal_ring5":        {"mu": 6.5,  "sigma": 1.0},
    "vicinal_ring6_ax_ax":  {"mu": 10.0, "sigma": 0.5},
    "vicinal_ring6_ax_eq":  {"mu": 4.0,  "sigma": 0.5},
    "vicinal_ring6_eq_eq":  {"mu": 3.5,  "sigma": 0.5},
    "geminal_ch2":          {"mu": 12.0, "sigma": 1.0},
    "geminal_ch2_ring":     {"mu": 9.0,  "sigma": 1.5},

    # Allylic and heteroatom-adjacent
    "allylic":              {"mu": 1.5,  "sigma": 0.3},
    "vicinal_nh":           {"mu": 6.0,  "sigma": 1.0},
}

_FALLBACK_TYPE = "vicinal_acyclic"

_MULT_LABELS = {1: "d", 2: "t", 3: "q", 4: "quint", 5: "sext", 6: "sept"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_exchangeable(mol, h_idx: int) -> bool:
    """Return True if H is attached to O or N (exchangeable in CDCl₃)."""
    for neighbor in mol.GetAtomWithIdx(h_idx).GetNeighbors():
        if neighbor.GetAtomicNum() in (7, 8):
            return True
    return False


def _get_h_groups(mol) -> dict:
    """
    Group non-exchangeable H atoms by canonical rank.
    Returns dict: rank -> [h_idx, ...].
    """
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    groups: dict = defaultdict(list)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            idx = atom.GetIdx()
            if not _is_exchangeable(mol, idx):
                groups[ranks[idx]].append(idx)
    return dict(groups)


def _multiplet_label(j_values: list) -> str:
    """
    Compute first-order multiplet label from coupling constants (Hz).

    Groups equal J values (within 0.5 Hz tolerance) and concatenates
    labels from largest to smallest coupling: e.g. [8.0, 8.0, 2.0] → "td".
    """
    if not j_values:
        return "s"
    j_sorted = sorted(j_values, reverse=True)
    groups = []
    for j in j_sorted:
        if groups and abs(j - groups[-1][0]) < 0.5:
            groups[-1][1] += 1
        else:
            groups.append([j, 1])
    return "".join(_MULT_LABELS.get(n, "m") for _, n in groups)


def _merge_overlapping(signals: list, tol_ppm: float) -> list:
    """Merge consecutive signals within tol_ppm into a single 'm' signal."""
    if not signals:
        return []
    merged = []
    cluster = [signals[0]]
    for sig in signals[1:]:
        if sig["shift_ppm"] - cluster[-1]["shift_ppm"] <= tol_ppm:
            cluster.append(sig)
        else:
            merged.append(_collapse_cluster(cluster))
            cluster = [sig]
    merged.append(_collapse_cluster(cluster))
    return merged


def _collapse_cluster(cluster: list) -> dict:
    if len(cluster) == 1:
        return cluster[0]
    mean_shift = sum(s["shift_ppm"] for s in cluster) / len(cluster)
    return {
        "shift_ppm":    round(mean_shift, 3),
        "multiplicity": "m",
        "n_protons":    sum(s["n_protons"] for s in cluster),
        "j_hz":         [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_j_coupling(mol_with_hs, h_a_idx: int, h_b_idx: int) -> str:
    """
    Classify the J-coupling type between two H atoms.

    Returns a key from J_COUPLING_TABLE in NMR_SKILL.md Section 4.2.
    Falls back to "vicinal_acyclic" for unrecognized types.
    """
    dist_matrix = rdmolops.GetDistanceMatrix(mol_with_hs).astype(int)
    d = int(dist_matrix[h_a_idx][h_b_idx])

    if d == 2:
        # Geminal: H_a – C – H_b
        h_a_nbrs = {n.GetIdx() for n in mol_with_hs.GetAtomWithIdx(h_a_idx).GetNeighbors()}
        h_b_nbrs = {n.GetIdx() for n in mol_with_hs.GetAtomWithIdx(h_b_idx).GetNeighbors()}
        shared = h_a_nbrs & h_b_nbrs
        if not shared:
            return _FALLBACK_TYPE
        c_idx = next(iter(shared))
        if mol_with_hs.GetAtomWithIdx(c_idx).IsInRing():
            return "geminal_ch2_ring"
        return "geminal_ch2"

    if d == 3:
        # Vicinal: H_a – X_a – X_b – H_b
        path = Chem.GetShortestPath(mol_with_hs, h_a_idx, h_b_idx)
        if not path or len(path) != 4:
            return _FALLBACK_TYPE
        x_a_idx, x_b_idx = path[1], path[2]
        x_a = mol_with_hs.GetAtomWithIdx(x_a_idx)
        x_b = mol_with_hs.GetAtomWithIdx(x_b_idx)
        ring_info = mol_with_hs.GetRingInfo()

        # 1. Aromatic ring coupling
        if x_a.GetIsAromatic() and x_b.GetIsAromatic():
            for ring in ring_info.AtomRings():
                if x_a_idx in ring and x_b_idx in ring:
                    ring_list = list(ring)
                    pos_a = ring_list.index(x_a_idx)
                    pos_b = ring_list.index(x_b_idx)
                    diff = min(abs(pos_a - pos_b), len(ring) - abs(pos_a - pos_b))
                    if len(ring) == 6:
                        return {
                            1: "aromatic_ortho",
                            2: "aromatic_meta",
                            3: "aromatic_para",
                        }.get(diff, _FALLBACK_TYPE)
                    if len(ring) == 5:
                        return {
                            1: "aromatic_5ring_34",
                            2: "aromatic_5ring_23",
                        }.get(diff, _FALLBACK_TYPE)

        # 2. Vinyl double bond (cis/trans unknown without 3D → use average)
        bond = mol_with_hs.GetBondBetweenAtoms(x_a_idx, x_b_idx)
        if bond and bond.GetBondTypeAsDouble() == 2.0:
            return "vinyl_avg"

        # 3. Ring vicinal
        if x_a.IsInRing() and x_b.IsInRing():
            for ring in sorted(ring_info.AtomRings(), key=len):
                if x_a_idx in ring and x_b_idx in ring:
                    if len(ring) == 5:
                        return "vicinal_ring5"
                    if len(ring) == 6:
                        # Without 3D conformation, use ax/eq as representative
                        return "vicinal_ring6_ax_eq"
                    break

        return _FALLBACK_TYPE

    return _FALLBACK_TYPE


def get_spin_systems(mol_with_hs) -> list:
    """
    Partition non-exchangeable H atoms into independent coupling networks.

    Returns list of lists: each inner list is H atom indices forming one
    spin system. H atoms in different spin systems do not couple and can be
    simulated independently.

    Two H atoms are in the same spin system if connected via a coupling path
    (graph distance ≤ 3 bonds). Exchangeable H (OH, NH) are excluded.
    """
    h_indices = [
        atom.GetIdx()
        for atom in mol_with_hs.GetAtoms()
        if atom.GetAtomicNum() == 1
        and not _is_exchangeable(mol_with_hs, atom.GetIdx())
    ]
    if not h_indices:
        return []

    dist_matrix = rdmolops.GetDistanceMatrix(mol_with_hs).astype(int)

    # Union-Find with path compression
    parent = {h: h for h in h_indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(len(h_indices)):
        for j in range(i + 1, len(h_indices)):
            if dist_matrix[h_indices[i]][h_indices[j]] <= 3:
                union(h_indices[i], h_indices[j])

    systems: dict = defaultdict(list)
    for h in h_indices:
        systems[find(h)].append(h)
    return list(systems.values())


def build_1h_multiplets(
    shift_result: dict,
    random_seed: int = 42,
    overlap_tol_ppm: float = 0.05,
    field_mhz: float = 400.0,
) -> list:
    """
    Build 1H 1D NMR multiplet table.

    Returns list of signal records:
    [
      {
        "shift_ppm":    float,
        "multiplicity": str,   # "s","d","t","q","dd","dt","ddd","m", etc.
        "n_protons":    int,
        "j_hz":         list[float],   # coupling constants, largest first
      },
      ...
    ]

    Assumptions:
    - First-order multiplets only (Δν/J > 5 at field_mhz)
    - Fixed J values with Gaussian randomisation (seed=random_seed)
    - Peaks within overlap_tol_ppm are merged to "m"
    - > 3 distinct coupling partner groups → reported as "m"

    See NMR_SKILL.md Section 4 for full specification.
    """
    mol = shift_result["mol_with_hs"]
    h_shifts = shift_result["h_shifts"]  # {h_atom_idx: ppm}
    rng = random.Random(random_seed)

    h_groups = _get_h_groups(mol)  # rank -> [h_idx, ...]
    if not h_groups:
        return []

    dist_matrix = rdmolops.GetDistanceMatrix(mol).astype(int)

    # Cache sampled J values: (min_idx, max_idx) -> float (Hz)
    j_cache: dict = {}

    def _sample_j(rep_a: int, rep_b: int) -> float:
        key = (min(rep_a, rep_b), max(rep_a, rep_b))
        if key not in j_cache:
            coupling_type = classify_j_coupling(mol, rep_a, rep_b)
            entry = J_COUPLING_TABLE.get(coupling_type, J_COUPLING_TABLE[_FALLBACK_TYPE])
            j_cache[key] = max(0.0, rng.gauss(entry["mu"], entry["sigma"]))
        return j_cache[key]

    # Build group list: (rank, rep_h, h_idxs, shift_ppm)
    # Use representative = first H in sorted group that has a shift
    group_list = []
    for rank, h_idxs in h_groups.items():
        rep, shift = None, None
        for h in sorted(h_idxs):
            if h in h_shifts:
                rep, shift = h, h_shifts[h]
                break
        if rep is None:
            continue
        group_list.append((rank, rep, h_idxs, shift))

    # Sort by shift_ppm so RNG calls are in a deterministic order
    group_list.sort(key=lambda x: x[3])

    signals = []
    for rank_a, rep_a, h_idxs_a, shift_ppm in group_list:
        coupling_j_list: list = []
        n_partner_groups = 0

        for rank_b, rep_b, h_idxs_b, _ in group_list:
            if rank_b == rank_a:
                continue  # same equivalence group — no self-coupling

            # Count H atoms in group_b at coupling distance (2–3 bonds) from rep_a
            n_coupling_h = sum(
                1 for h_b in h_idxs_b
                if 2 <= dist_matrix[rep_a][h_b] <= 3
            )
            if n_coupling_h == 0:
                continue

            n_partner_groups += 1
            j_val = _sample_j(rep_a, rep_b)
            coupling_j_list.extend([j_val] * n_coupling_h)

        # > 3 distinct coupling partner groups → complex multiplet, report "m"
        if n_partner_groups > 3:
            mult = "m"
            coupling_j_list = []
        elif not coupling_j_list:
            mult = "s"
        else:
            mult = _multiplet_label(coupling_j_list)

        signals.append({
            "shift_ppm":    round(shift_ppm, 3),
            "multiplicity": mult,
            "n_protons":    len(h_idxs_a),
            "j_hz":         [round(j, 1) for j in sorted(coupling_j_list, reverse=True)],
        })

    signals.sort(key=lambda x: x["shift_ppm"])
    return _merge_overlapping(signals, overlap_tol_ppm)
