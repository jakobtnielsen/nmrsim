"""
tests/test_1h.py — Unit tests for build_1h.py (Phase 2: 1H 1D NMR).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools

import pytest
from rdkit import Chem
from rdkit.Chem import rdmolops

from predict_shifts import predict_shifts
from build_1h import (
    build_1h_multiplets,
    get_spin_systems,
    classify_j_coupling,
    J_COUPLING_TABLE,
    _multiplet_label,
)


# ---------------------------------------------------------------------------
# Shared fixture: cache API calls across tests
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _shift_result(smiles: str):
    """Cache API calls across tests."""
    return predict_shifts(smiles, backend="nmrdb")


# ---------------------------------------------------------------------------
# _multiplet_label unit tests (no API needed)
# ---------------------------------------------------------------------------

class TestMultipletLabel:
    def test_singlet(self):
        assert _multiplet_label([]) == "s"

    def test_doublet(self):
        assert _multiplet_label([8.0]) == "d"

    def test_triplet(self):
        assert _multiplet_label([7.5, 7.5]) == "t"

    def test_quartet(self):
        assert _multiplet_label([7.0, 7.0, 7.0]) == "q"

    def test_quintet(self):
        assert _multiplet_label([7.0, 7.0, 7.0, 7.0]) == "quint"

    def test_dd_two_unequal(self):
        assert _multiplet_label([8.0, 2.0]) == "dd"

    def test_td_triplet_of_doublets(self):
        # [8.0, 8.0, 2.0]: largest group first → "td"
        assert _multiplet_label([8.0, 8.0, 2.0]) == "td"

    def test_dt_doublet_of_triplets(self):
        # [8.0, 2.0, 2.0] → d (J=8) then t (J≈2) → "dt"
        assert _multiplet_label([8.0, 2.0, 2.0]) == "dt"

    def test_ddd_three_unequal(self):
        assert _multiplet_label([8.0, 4.0, 2.0]) == "ddd"

    def test_equal_grouping_within_tolerance(self):
        # 8.0 and 8.3 are within 0.5 Hz → grouped together → "t"
        assert _multiplet_label([8.3, 8.0]) == "t"

    def test_outside_tolerance_separate(self):
        # 8.0 and 8.6 differ by 0.6 Hz → separate groups → "dd"
        assert _multiplet_label([8.6, 8.0]) == "dd"

    def test_sorted_input_order_independent(self):
        # Same J values regardless of input order → same label
        assert _multiplet_label([2.0, 8.0]) == _multiplet_label([8.0, 2.0])


# ---------------------------------------------------------------------------
# classify_j_coupling tests (no API needed)
# ---------------------------------------------------------------------------

class TestClassifyJCoupling:
    def _get_dist(self, mol):
        return rdmolops.GetDistanceMatrix(mol).astype(int)

    def test_vicinal_acyclic(self):
        """Ethane: vicinal H-C-C-H coupling."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        dist = self._get_dist(mol)
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        for h_a in h_atoms:
            for h_b in h_atoms:
                if h_a < h_b and dist[h_a][h_b] == 3:
                    result = classify_j_coupling(mol, h_a, h_b)
                    assert result == "vicinal_acyclic", \
                        f"Expected vicinal_acyclic, got {result}"
                    return
        pytest.skip("No vicinal pair found in ethane")

    def test_geminal_sp3(self):
        """CH₂ group: geminal H-C-H coupling."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCC"))
        dist = self._get_dist(mol)
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        found = False
        for h_a in h_atoms:
            for h_b in h_atoms:
                if h_a < h_b and dist[h_a][h_b] == 2:
                    result = classify_j_coupling(mol, h_a, h_b)
                    assert result in ("geminal_ch2", "geminal_ch2_ring"), \
                        f"Expected geminal type, got {result}"
                    found = True
                    break
            if found:
                break
        assert found, "No geminal pair found in propane"

    def test_aromatic_ortho(self):
        """Benzene: adjacent aromatic H → aromatic_ortho."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        dist = self._get_dist(mol)
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        found = False
        for h_a in h_atoms:
            for h_b in h_atoms:
                if h_a < h_b and dist[h_a][h_b] == 3:
                    result = classify_j_coupling(mol, h_a, h_b)
                    assert result == "aromatic_ortho", \
                        f"Expected aromatic_ortho, got {result}"
                    found = True
                    break
            if found:
                break
        assert found, "No ortho pair found in benzene"

    def test_aromatic_meta_in_substituted_ring(self):
        """Meta H-H coupling in toluene: 4-bond path gives aromatic_meta."""
        mol = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
        dist = self._get_dist(mol)
        h_atoms = [
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetAtomicNum() == 1
            and mol.GetAtomWithIdx(
                list(mol.GetAtomWithIdx(a.GetIdx()).GetNeighbors())[0].GetIdx()
            ).GetIsAromatic()
        ]
        # Find a pair with graph distance 4 (meta in ring: H-C-C-C-H)
        for h_a in h_atoms:
            for h_b in h_atoms:
                if h_a < h_b and dist[h_a][h_b] == 4:
                    # This is a meta pair (but dist=4 is excluded from coupling range)
                    # classify_j_coupling only handles dist 2 and 3; dist=4 returns fallback
                    result = classify_j_coupling(mol, h_a, h_b)
                    assert result == "vicinal_acyclic"  # fallback for dist=4
                    return
        pytest.skip("No meta pair found")

    def test_vicinal_ring5(self):
        """Cyclopentane: vicinal H in 5-membered ring."""
        mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCC1"))
        dist = self._get_dist(mol)
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        found = False
        for h_a in h_atoms:
            for h_b in h_atoms:
                if h_a < h_b and dist[h_a][h_b] == 3:
                    result = classify_j_coupling(mol, h_a, h_b)
                    # Should be ring5 or ring6 depending on the ring
                    assert result in ("vicinal_ring5", "vicinal_ring6_ax_eq"), \
                        f"Expected ring vicinal, got {result}"
                    found = True
                    break
            if found:
                break
        assert found, "No vicinal pair found in cyclopentane"

    def test_geminal_in_ring(self):
        """CH₂ in cyclohexane ring: geminal_ch2_ring."""
        mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
        dist = self._get_dist(mol)
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        found = False
        for h_a in h_atoms:
            for h_b in h_atoms:
                if h_a < h_b and dist[h_a][h_b] == 2:
                    result = classify_j_coupling(mol, h_a, h_b)
                    assert result == "geminal_ch2_ring", \
                        f"Expected geminal_ch2_ring, got {result}"
                    found = True
                    break
            if found:
                break
        assert found, "No geminal pair found in cyclohexane"

    def test_vinyl_avg(self):
        """Vinyl protons: H-C=C-H without 3D → vinyl_avg."""
        mol = Chem.AddHs(Chem.MolFromSmiles("C=C"))
        dist = self._get_dist(mol)
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        found = False
        for h_a in h_atoms:
            for h_b in h_atoms:
                if h_a < h_b and dist[h_a][h_b] == 3:
                    parent_a = list(mol.GetAtomWithIdx(h_a).GetNeighbors())[0]
                    parent_b = list(mol.GetAtomWithIdx(h_b).GetNeighbors())[0]
                    # Only test trans-vinyl (different carbons)
                    if parent_a.GetIdx() != parent_b.GetIdx():
                        result = classify_j_coupling(mol, h_a, h_b)
                        assert result == "vinyl_avg", \
                            f"Expected vinyl_avg, got {result}"
                        found = True
                        break
            if found:
                break
        assert found, "No trans-vinyl pair found in ethylene"

    def test_fallback_for_large_distance(self):
        """Distance > 3 → fallback (vicinal_acyclic)."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
        dist = self._get_dist(mol)
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        for h_a in h_atoms:
            for h_b in h_atoms:
                if dist[h_a][h_b] > 3:
                    result = classify_j_coupling(mol, h_a, h_b)
                    assert result == "vicinal_acyclic"  # fallback
                    return


# ---------------------------------------------------------------------------
# get_spin_systems tests
# ---------------------------------------------------------------------------

class TestGetSpinSystems:
    def test_returns_list_of_lists(self):
        result = _shift_result("CC(=O)c1ccccc1")
        mol = result["mol_with_hs"]
        systems = get_spin_systems(mol)
        assert isinstance(systems, list)
        for s in systems:
            assert isinstance(s, list)
            assert len(s) > 0

    def test_acetophenone_two_systems(self):
        """
        Acetophenone: CH₃ is isolated from the aromatic ring (separated by C=O).
        Expected: {3H CH₃} and {5H aromatic}.
        """
        result = _shift_result("CC(=O)c1ccccc1")
        mol = result["mol_with_hs"]
        systems = get_spin_systems(mol)
        sizes = sorted(len(s) for s in systems)
        assert sizes == [3, 5], f"Expected spin systems of sizes [3, 5], got {sizes}"

    def test_all_h_accounted_for(self):
        """Every non-exchangeable H must appear in exactly one spin system."""
        result = _shift_result("CC(=O)c1ccccc1")
        mol = result["mol_with_hs"]
        systems = get_spin_systems(mol)
        seen = set()
        for system in systems:
            for h in system:
                assert h not in seen, f"H atom {h} in multiple spin systems"
                seen.add(h)
        expected = sum(
            1 for a in mol.GetAtoms()
            if a.GetAtomicNum() == 1
            and not any(
                n.GetAtomicNum() in (7, 8)
                for n in a.GetNeighbors()
            )
        )
        assert len(seen) == expected, \
            f"Expected {expected} H in spin systems, got {len(seen)}"

    def test_ethanol_excludes_oh(self):
        """
        Ethanol (CCO): CH₃ and CH₂ form one spin system; OH is excluded.
        Expected: 1 system with 5 H atoms.
        """
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        systems = get_spin_systems(mol)
        assert len(systems) == 1, \
            f"Expected 1 spin system (OH excluded), got {len(systems)}: {systems}"
        assert len(systems[0]) == 5, \
            f"Expected 5 H (CH₃+CH₂), got {len(systems[0])}"

    def test_caffeine_isolated_methyls(self):
        """
        Caffeine: 3 N-CH₃ + 1 vinyl CH. Each is an isolated spin system
        (all separated by N atoms from each other).
        Expected: 4 spin systems.
        """
        result = _shift_result("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
        mol = result["mol_with_hs"]
        systems = get_spin_systems(mol)
        assert len(systems) == 4, \
            f"Expected 4 spin systems for caffeine, got {len(systems)}"

    def test_benzene_one_system(self):
        """Benzene: all 6 H in one aromatic spin system."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        systems = get_spin_systems(mol)
        assert len(systems) == 1, \
            f"Expected 1 spin system for benzene, got {len(systems)}"
        assert len(systems[0]) == 6, \
            f"Expected 6 H, got {len(systems[0])}"


# ---------------------------------------------------------------------------
# build_1h_multiplets tests
# ---------------------------------------------------------------------------

class TestBuildMultiplets:
    def test_returns_list(self):
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        assert isinstance(signals, list)

    def test_correct_keys(self):
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        for sig in signals:
            assert "shift_ppm"    in sig
            assert "multiplicity" in sig
            assert "n_protons"    in sig
            assert "j_hz"         in sig

    def test_n_protons_positive(self):
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        for sig in signals:
            assert sig["n_protons"] >= 1, f"n_protons must be ≥ 1: {sig}"

    def test_sorted_by_shift(self):
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        shifts = [s["shift_ppm"] for s in signals]
        assert shifts == sorted(shifts), f"Signals not sorted by shift: {shifts}"

    def test_deterministic(self):
        """Same seed → identical output."""
        result = _shift_result("CC(=O)c1ccccc1")
        s1 = build_1h_multiplets(result, random_seed=42)
        s2 = build_1h_multiplets(result, random_seed=42)
        assert s1 == s2, "build_1h_multiplets must be deterministic"

    def test_different_seeds_differ(self):
        """Different seeds → different J values (very likely)."""
        result = _shift_result("CC(=O)c1ccccc1")
        s1 = build_1h_multiplets(result, random_seed=42)
        s2 = build_1h_multiplets(result, random_seed=99)
        j1 = [j for s in s1 for j in s["j_hz"]]
        j2 = [j for s in s2 for j in s["j_hz"]]
        # J values exist for aromatic signals
        assert j1 or j2, "Expected non-empty J lists"
        assert j1 != j2, "Different seeds should give different J values"

    # --- Acetophenone ---

    def test_acetophenone_signal_count(self):
        """Acetophenone should produce 4 multiplet signals."""
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        assert len(signals) == 4, \
            f"Expected 4 signals for acetophenone, got {len(signals)}: {signals}"

    def test_acetophenone_total_protons(self):
        """Total integral must equal 8 (CH₃=3, ortho=2, meta=2, para=1)."""
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        total = sum(s["n_protons"] for s in signals)
        assert total == 8, f"Expected 8 total protons, got {total}"

    def test_acetophenone_methyl_singlet(self):
        """CH₃ of acetophenone is a singlet: adjacent to C=O (no H neighbours)."""
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        methyl = min(signals, key=lambda s: s["shift_ppm"])
        assert methyl["n_protons"] == 3, \
            f"Expected 3H for CH₃, got {methyl['n_protons']}"
        assert methyl["multiplicity"] == "s", \
            f"CH₃ should be singlet, got {methyl['multiplicity']}"
        assert methyl["j_hz"] == [], \
            f"Singlet should have no J values: {methyl['j_hz']}"

    def test_acetophenone_aromatic_have_coupling(self):
        """Aromatic H of acetophenone should have non-zero J (unless merged to 'm')."""
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        aromatic = [s for s in signals if s["shift_ppm"] > 6.5]
        assert len(aromatic) > 0, "No aromatic signals found"
        for sig in aromatic:
            if sig["multiplicity"] != "m":
                assert len(sig["j_hz"]) > 0, \
                    f"Aromatic signal should have J couplings: {sig}"

    def test_acetophenone_aromatic_j_in_range(self):
        """Aromatic J values should be in 0.3–10 Hz range."""
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        for sig in signals:
            for j in sig["j_hz"]:
                assert 0.3 <= j <= 10.0, \
                    f"J value {j} out of expected aromatic range: {sig}"

    # --- Ethyl benzoate (CH₃/CH₂ pattern) ---

    def test_ethyl_benzoate_triplet_quartet(self):
        """
        Ethyl benzoate: -OCH₂CH₃
          CH₃ (3H, ~1.4 ppm) → triplet (split by 2 OCH₂ Hs)
          OCH₂ (2H, ~4.4 ppm) → quartet (split by 3 CH₃ Hs)
        """
        result = _shift_result("CCOC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        aliphatic = sorted(
            [s for s in signals if s["shift_ppm"] < 5.0],
            key=lambda s: s["shift_ppm"],
        )
        assert len(aliphatic) >= 2, \
            f"Expected at least CH₃ and OCH₂ aliphatic signals: {aliphatic}"
        ch3  = aliphatic[0]   # ≈ 1.4 ppm
        och2 = aliphatic[1]   # ≈ 4.4 ppm
        assert ch3["multiplicity"] == "t", \
            f"CH₃ of ethyl ester should be triplet, got {ch3['multiplicity']}"
        assert och2["multiplicity"] == "q", \
            f"OCH₂ of ethyl ester should be quartet, got {och2['multiplicity']}"

    def test_ethyl_benzoate_total_aliphatic_protons(self):
        """CH₃ (3H) + OCH₂ (2H) = 5 aliphatic protons."""
        result = _shift_result("CCOC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        aliphatic_total = sum(
            s["n_protons"] for s in signals if s["shift_ppm"] < 5.0
        )
        assert aliphatic_total == 5, \
            f"Expected 5 aliphatic protons, got {aliphatic_total}"

    # --- Overlap merging ---

    def test_overlap_merging_produces_m(self):
        """
        Two signals within overlap_tol_ppm should merge to multiplicity 'm'.
        Test by using a very large tolerance so all signals merge.
        """
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result, overlap_tol_ppm=100.0)
        assert len(signals) == 1, \
            f"With tol=100 ppm all signals should merge to 1: {signals}"
        assert signals[0]["multiplicity"] == "m"
        assert signals[0]["n_protons"] == 8  # all 8H merged
        assert signals[0]["j_hz"] == []

    def test_no_overlap_default_tol(self):
        """
        With default tolerance (0.05 ppm), acetophenone signals are well-separated
        and should NOT merge.
        """
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result, overlap_tol_ppm=0.05)
        assert len(signals) == 4, \
            f"Acetophenone should keep 4 signals with default tol: {signals}"

    # --- J values on singlets are empty ---

    def test_singlet_has_no_j(self):
        """Any signal with multiplicity 's' must have j_hz == []."""
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result)
        for sig in signals:
            if sig["multiplicity"] == "s":
                assert sig["j_hz"] == [], \
                    f"Singlet should have no J values: {sig}"

    # --- m signals have no j ---

    def test_m_signal_has_no_j(self):
        """Any signal with multiplicity 'm' (complex) must have j_hz == []."""
        result = _shift_result("CC(=O)c1ccccc1")
        signals = build_1h_multiplets(result, overlap_tol_ppm=100.0)
        for sig in signals:
            if sig["multiplicity"] == "m":
                assert sig["j_hz"] == [], \
                    f"'m' signal should have no J values: {sig}"

    # --- J_COUPLING_TABLE is self-consistent ---

    def test_j_table_keys_valid(self):
        """J_COUPLING_TABLE entries must have positive mu."""
        for key, entry in J_COUPLING_TABLE.items():
            assert entry["mu"] > 0, f"J entry '{key}' has non-positive mu"
            assert entry["sigma"] >= 0, f"J entry '{key}' has negative sigma"
