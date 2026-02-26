"""
predict_shifts.py — Stage 1: per-atom 1H and 13C shift predictions.

Backends:
  "nmrdb"      → Zakodium/nmrdb.org REST API (per-atom, primary)
  "nmrshiftdb" → NMRShiftDB2 HTTP API (spectrum-level, fallback)
  "sgnn"       → AstraZeneca SGNN local model (stub)
"""

import re
import time
import urllib.parse
import warnings   # used by nmrshiftdb backend (SSL warning suppression)
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from collections import defaultdict


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BackendError(Exception):
    """Raised when a shift-prediction backend fails."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_shifts(smiles: str, backend: str = "nmrdb") -> dict:
    """
    Predict 1H and 13C chemical shifts for all atoms in a molecule.

    Args:
        smiles: Input SMILES string (will be canonicalized internally).
        backend: One of "nmrdb" | "nmrshiftdb" | "sgnn".

    Returns:
        {
          "canonical_smiles": str,
          "h_shifts":   {atom_idx: float, ...},   # atom idx in mol_with_hs
          "c_shifts":   {atom_idx: float, ...},   # atom idx in mol_with_hs
          "backend":    str,
          "mol_with_hs": rdkit_mol_object,
        }

    Raises:
        BackendError if the backend fails and no fallback is available.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise BackendError(f"Invalid SMILES: {smiles}")
    canonical_smiles = Chem.MolToSmiles(mol)

    if backend == "nmrdb":
        return _fetch_nmrdb(canonical_smiles)
    elif backend == "nmrshiftdb":
        return _fetch_nmrshiftdb(canonical_smiles)
    elif backend == "sgnn":
        return _fetch_sgnn(canonical_smiles)
    else:
        raise BackendError(f"Unknown backend: {backend!r}. "
                           f"Choose from 'nmrdb', 'nmrshiftdb', 'sgnn'.")


# ---------------------------------------------------------------------------
# Backend: nmrdb (Zakodium REST API)
# ---------------------------------------------------------------------------

_ZAKODIUM_13C = "https://nmr-prediction.service.zakodium.com/v1/predict/carbon"
_ZAKODIUM_1H  = "https://nmr-prediction.service.zakodium.com/v1/predict/proton"


def _fetch_nmrdb(canonical_smiles: str) -> dict:
    """
    Call the Zakodium/nmrdb.org prediction service.

    Key insight: both 1H and 13C APIs use the input SMILES atom ordering
    (i.e. the order of atoms as they appear in the canonical SMILES string,
    same as RDKit's canonical ordering for the same canonical SMILES).

    - 1H signal atoms: indices into mol_h (RDKit canonical mol with explicit H).
      Heavy atoms occupy indices 0..n_heavy-1; H atoms occupy n_heavy..total-1.
    - 13C signal atoms: indices into mol (RDKit canonical mol, heavy atoms only,
      0..n_heavy-1), which are also the heavy atom indices in mol_h.

    No molfile parsing or substructure matching is needed.
    """
    try:
        r1h = requests.post(_ZAKODIUM_1H, json={"smiles": canonical_smiles},
                            timeout=30)
        r1h.raise_for_status()
        d1h = r1h.json()["data"]
    except Exception as exc:
        raise BackendError(
            f"nmrdb 1H request failed for SMILES '{canonical_smiles}': {exc}"
        ) from exc

    time.sleep(1)  # rate-limit

    try:
        r13 = requests.post(_ZAKODIUM_13C, json={"smiles": canonical_smiles},
                            timeout=30)
        r13.raise_for_status()
        d13 = r13.json()["data"]
    except Exception as exc:
        raise BackendError(
            f"nmrdb 13C request failed for SMILES '{canonical_smiles}': {exc}"
        ) from exc

    # --- Build canonical RDKit mol_h ----------------------------------------
    # Signal atom indices from the API match this mol's atom ordering.
    mol = Chem.MolFromSmiles(canonical_smiles)
    mol_h = Chem.AddHs(mol)

    # --- Build h_shifts: {mol_h_atom_idx: ppm} for H atoms -----------------
    # 1H signal atoms are indices into mol_h (H atoms at positions n_heavy+)
    h_shifts = {}
    for sig in d1h.get("signals", []):
        ppm = float(sig["delta"])
        for ai in sig["atoms"]:
            if 0 <= ai < mol_h.GetNumAtoms():
                h_shifts[ai] = ppm

    # --- Build c_shifts: {mol_h_atom_idx: ppm} for C atoms -----------------
    # 13C signal atoms are indices into mol (heavy atoms, 0..n_heavy-1),
    # which are the same as mol_h heavy atom indices.
    n_heavy = mol.GetNumAtoms()
    c_shifts = {}
    for sig in d13.get("signals", []):
        ppm = float(sig["delta"])
        for ai in sig["atoms"]:
            if 0 <= ai < n_heavy:
                atom = mol.GetAtomWithIdx(ai)
                if atom.GetAtomicNum() == 6:   # only store for carbon
                    c_shifts[ai] = ppm

    return {
        "canonical_smiles": canonical_smiles,
        "h_shifts": h_shifts,
        "c_shifts": c_shifts,
        "backend": "nmrdb",
        "mol_with_hs": mol_h,
    }


# ---------------------------------------------------------------------------
# Backend: nmrshiftdb (spectrum-level, fallback)
# ---------------------------------------------------------------------------

_NMRSHIFTDB_BASE = "https://nmrshiftdb.nmr.uni-koeln.de/NmrshiftdbServlet/nmrshiftdbaction/searchorpredict/smiles"


def _parse_jcamp(text: str) -> list:
    """Parse a JCAMP-DX PEAKTABLE and return list of x (ppm) values."""
    peaks = []
    in_table = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("##PEAKTABLE="):
            in_table = True
            continue
        if line.startswith("##END"):
            break
        if in_table and line and not line.startswith("##"):
            # Format: x,y (or x,y x,y ...)
            for token in re.split(r"[\s;]+", line):
                if "," in token:
                    x_str = token.split(",")[0]
                    try:
                        peaks.append(float(x_str))
                    except ValueError:
                        pass
    return sorted(peaks)


def _estimated_c_shift(atom, mol) -> float:
    """
    Rough expected 13C chemical shift for ordering in NMRShiftDB assignment.
    Not accurate — used only to sort equivalence groups.
    """
    if atom.GetIsAromatic():
        return 125.0
    hyb = atom.GetHybridization()
    neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]
    heavy_neighbors = [n for n in neighbors if n.GetAtomicNum() != 1]
    # Check for C=O
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if (other.GetAtomicNum() == 8 and
                bond.GetBondTypeAsDouble() == 2.0):
            # C=O: check if ester/amide (another O/N neighbor)
            for n in heavy_neighbors:
                if n.GetAtomicNum() in (7, 8):
                    return 170.0
            return 200.0
    # sp2 (non-carbonyl, non-aromatic) = alkene
    from rdkit.Chem import rdchem
    if hyb == rdchem.HybridizationType.SP2:
        return 125.0
    # sp3 checks
    o_neighbors = sum(1 for n in heavy_neighbors if n.GetAtomicNum() == 8)
    n_neighbors = sum(1 for n in heavy_neighbors if n.GetAtomicNum() == 7)
    if o_neighbors >= 2:
        return 100.0
    if o_neighbors == 1:
        return 65.0
    if n_neighbors >= 1:
        return 45.0
    # plain alkyl — fewer H = higher ppm roughly
    n_h = atom.GetTotalNumHs()
    return 35.0 - n_h * 5.0


def _estimated_h_shift(h_atom, parent_c, mol) -> float:
    """Rough expected 1H shift for NMRShiftDB assignment ordering."""
    if parent_c.GetIsAromatic():
        return 7.0
    neighbors = [mol.GetAtomWithIdx(n.GetIdx())
                 for n in parent_c.GetNeighbors()
                 if n.GetIdx() != h_atom.GetIdx()]
    for bond in parent_c.GetBonds():
        other = bond.GetOtherAtom(parent_c)
        if other.GetAtomicNum() == 8 and bond.GetBondTypeAsDouble() == 2.0:
            return 9.5   # aldehyde
    heavy_n = [n for n in neighbors if n.GetAtomicNum() != 1]
    for n in heavy_n:
        if n.GetAtomicNum() == 8:
            return 3.5
        if n.GetAtomicNum() == 7:
            return 2.5
    from rdkit.Chem import rdchem
    if parent_c.GetHybridization() == rdchem.HybridizationType.SP2:
        return 5.5
    return 1.5


def _fetch_nmrshiftdb(canonical_smiles: str) -> dict:
    """
    Fetch spectrum-level 1H and 13C shifts from NMRShiftDB2 and assign to atoms.
    Result is approximate ('atom_resolved': False in returned dict).
    """
    enc = urllib.parse.quote(canonical_smiles, safe="")
    mol = Chem.MolFromSmiles(canonical_smiles)
    mol_h = Chem.AddHs(mol)

    # --- Fetch 13C -----------------------------------------------------------
    url_13c = f"{_NMRSHIFTDB_BASE}/{enc}/spectrumtype/13C/format/jcamp"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = requests.get(url_13c, timeout=20, verify=False)
        r.raise_for_status()
        peaks_13c = _parse_jcamp(r.text)
    except Exception as exc:
        raise BackendError(
            f"nmrshiftdb 13C request failed for '{canonical_smiles}': {exc}"
        ) from exc

    time.sleep(1)

    # --- Fetch 1H -----------------------------------------------------------
    url_1h = f"{_NMRSHIFTDB_BASE}/{enc}/spectrumtype/1H/format/jcamp"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = requests.get(url_1h, timeout=20, verify=False)
        r.raise_for_status()
        peaks_1h = _parse_jcamp(r.text)
    except Exception as exc:
        raise BackendError(
            f"nmrshiftdb 1H request failed for '{canonical_smiles}': {exc}"
        ) from exc

    # --- Assign 13C peaks to equivalence groups -----------------------------
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))
    c_groups: dict[int, list] = defaultdict(list)
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 6:
            c_groups[ranks[atom.GetIdx()]].append(atom.GetIdx())

    # Sort C groups by estimated shift
    c_group_list = sorted(
        c_groups.values(),
        key=lambda idxs: _estimated_c_shift(mol_h.GetAtomWithIdx(idxs[0]), mol_h)
    )

    c_shifts = {}
    for i, group_idxs in enumerate(c_group_list):
        if i < len(peaks_13c):
            ppm = peaks_13c[i]
        else:
            # More C groups than peaks — use last known peak
            ppm = peaks_13c[-1] if peaks_13c else 0.0
        for idx in group_idxs:
            c_shifts[idx] = ppm

    # --- Assign 1H peaks to equivalence groups -----------------------------
    h_groups: dict[int, list] = defaultdict(list)
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            h_groups[ranks[atom.GetIdx()]].append(atom.GetIdx())

    h_group_list = sorted(
        h_groups.values(),
        key=lambda idxs: _estimated_h_shift(
            mol_h.GetAtomWithIdx(idxs[0]),
            mol_h.GetAtomWithIdx(
                next(n.GetIdx() for n in mol_h.GetAtomWithIdx(idxs[0]).GetNeighbors())
            ),
            mol_h
        )
    )

    h_shifts = {}
    for i, group_idxs in enumerate(h_group_list):
        if i < len(peaks_1h):
            ppm = peaks_1h[i]
        else:
            ppm = peaks_1h[-1] if peaks_1h else 0.0
        for idx in group_idxs:
            h_shifts[idx] = ppm

    return {
        "canonical_smiles": canonical_smiles,
        "h_shifts": h_shifts,
        "c_shifts": c_shifts,
        "backend": "nmrshiftdb",
        "atom_resolved": False,  # approximate assignment
        "mol_with_hs": mol_h,
    }


# ---------------------------------------------------------------------------
# Backend: sgnn (AstraZeneca, stub)
# ---------------------------------------------------------------------------

def _fetch_sgnn(canonical_smiles: str) -> dict:
    """
    Use the AstraZeneca SGNN local model for per-atom shifts.
    Requires: git clone https://github.com/AstraZeneca/hsqc_structure_elucidation
              pip install -e <repo_dir>
    """
    try:
        import hsqc  # noqa: F401
    except ImportError:
        raise BackendError(
            f"sgnn backend not installed for SMILES '{canonical_smiles}'. "
            "Clone https://github.com/AstraZeneca/hsqc_structure_elucidation "
            "and run: pip install -e <repo_dir>"
        )
    raise NotImplementedError(
        "sgnn backend: import succeeded but integration not yet implemented. "
        f"SMILES: {canonical_smiles}"
    )
