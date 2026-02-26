"""
tests/test_shifts.py — Unit tests for predict_shifts.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from rdkit import Chem
from predict_shifts import predict_shifts, BackendError


def test_predict_shifts_nmrdb_benzene():
    """Benzene has 6 equivalent H → one H shift entry; 1 unique C shift."""
    result = predict_shifts("c1ccccc1", backend="nmrdb")
    assert result["canonical_smiles"] is not None
    assert result["backend"] == "nmrdb"
    mol = result["mol_with_hs"]
    assert mol is not None

    h_shifts = result["h_shifts"]
    c_shifts = result["c_shifts"]

    # All H on benzene are equivalent → may appear as 1 or 6 entries
    # (API returns per-H entries, but all with same ppm)
    assert len(h_shifts) >= 1
    # All H shifts should be around 7.2-7.4 ppm
    for ppm in h_shifts.values():
        assert 6.5 <= ppm <= 8.5, f"Unexpected benzene H shift: {ppm}"

    # All C shifts should be around 128 ppm
    assert len(c_shifts) >= 1
    for ppm in c_shifts.values():
        assert 120.0 <= ppm <= 145.0, f"Unexpected benzene C shift: {ppm}"


def test_predict_shifts_nmrdb_acetophenone():
    """Acetophenone: CH3 at ~2.6 ppm, carbonyl C at ~197 ppm."""
    result = predict_shifts("CC(=O)c1ccccc1", backend="nmrdb")
    h_shifts = result["h_shifts"]
    c_shifts = result["c_shifts"]

    # Should have H shifts for CH3 (~2.6 ppm) and aromatic H (~7.5–8.0 ppm)
    h_ppms = sorted(h_shifts.values())
    assert any(1.5 <= p <= 3.5 for p in h_ppms), \
        f"No CH3 H shift found in {h_ppms}"
    assert any(6.5 <= p <= 8.5 for p in h_ppms), \
        f"No aromatic H shift found in {h_ppms}"

    # Should have C shifts including carbonyl (~197 ppm)
    c_ppms = list(c_shifts.values())
    assert any(p >= 185.0 for p in c_ppms), \
        f"No carbonyl C shift found in {c_ppms}"


def test_invalid_smiles_raises():
    with pytest.raises(BackendError):
        predict_shifts("not_a_smiles", backend="nmrdb")


def test_unknown_backend_raises():
    with pytest.raises(BackendError):
        predict_shifts("CC", backend="nonexistent")


def test_mol_with_hs_has_explicit_h():
    """mol_with_hs must contain explicit H atoms."""
    result = predict_shifts("CC(=O)c1ccccc1", backend="nmrdb")
    mol = result["mol_with_hs"]
    h_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    assert len(h_atoms) > 0, "mol_with_hs should have explicit H atoms"


def test_shift_dict_keys_valid_atom_indices():
    """All keys in h_shifts/c_shifts must be valid atom indices."""
    result = predict_shifts("CC(=O)c1ccccc1", backend="nmrdb")
    mol    = result["mol_with_hs"]
    n      = mol.GetNumAtoms()

    for idx in result["h_shifts"]:
        assert 0 <= idx < n, f"h_shifts key {idx} out of range (n={n})"
        assert mol.GetAtomWithIdx(idx).GetAtomicNum() == 1, \
            f"Atom {idx} in h_shifts is not H"

    for idx in result["c_shifts"]:
        assert 0 <= idx < n, f"c_shifts key {idx} out of range (n={n})"
        assert mol.GetAtomWithIdx(idx).GetAtomicNum() == 6, \
            f"Atom {idx} in c_shifts is not C"
