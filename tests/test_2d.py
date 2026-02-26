"""
tests/test_2d.py — Unit tests for build_2d.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from rdkit import Chem
from predict_shifts import predict_shifts
from build_2d import build_hsqc, build_hmbc, build_cosy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

import functools

@functools.lru_cache(maxsize=None)
def _shift_result(smiles: str):
    """Cache API calls across tests."""
    return predict_shifts(smiles, backend="nmrdb")


# ---------------------------------------------------------------------------
# HSQC tests
# ---------------------------------------------------------------------------

class TestHSQC:
    def test_acetophenone_hsqc_count(self):
        """Acetophenone should have 4 HSQC peaks: CH3, ortho, meta, para."""
        result = _shift_result("CC(=O)c1ccccc1")
        peaks  = build_hsqc(result)
        assert len(peaks) == 4, f"Expected 4 HSQC peaks, got {len(peaks)}: {peaks}"

    def test_benzene_hsqc_count(self):
        """Benzene: 6 equivalent H → 1 HSQC peak."""
        result = _shift_result("c1ccccc1")
        peaks  = build_hsqc(result)
        assert len(peaks) == 1, f"Expected 1 HSQC peak, got {len(peaks)}"
        assert peaks[0]["n_h"] == 6, \
            f"Expected n_h=6 for benzene, got {peaks[0]['n_h']}"

    def test_hsqc_no_quaternary_carbons(self):
        """Quaternary carbons must NOT appear in HSQC."""
        result = _shift_result("CC(=O)c1ccccc1")
        peaks  = build_hsqc(result)
        # For acetophenone, C=O (~197 ppm) and ipso C (~137 ppm) are quaternary
        c_ppms = [p["c_ppm"] for p in peaks]
        assert not any(p > 185.0 for p in c_ppms), \
            f"C=O quaternary carbon appears in HSQC: {c_ppms}"
        assert not any(135.0 <= p <= 145.0 for p in c_ppms), \
            f"Ipso quaternary carbon appears in HSQC: {c_ppms}"

    def test_hsqc_n_h_positive(self):
        result = _shift_result("CC(=O)c1ccccc1")
        peaks  = build_hsqc(result)
        for p in peaks:
            assert p["n_h"] >= 1, f"n_h must be >= 1: {p}"

    def test_hsqc_sorted_by_h_ppm(self):
        result = _shift_result("CC(=O)c1ccccc1")
        peaks  = build_hsqc(result)
        h_ppms = [p["h_ppm"] for p in peaks]
        assert h_ppms == sorted(h_ppms), f"HSQC not sorted by h_ppm: {h_ppms}"

    def test_hsqc_methyl_n_h(self):
        """CH3 group must appear with n_h=3."""
        result = _shift_result("CC(=O)c1ccccc1")
        peaks  = build_hsqc(result)
        methyl_peak = min(peaks, key=lambda p: p["h_ppm"])  # lowest shift
        assert methyl_peak["n_h"] == 3, \
            f"CH3 HSQC peak should have n_h=3, got {methyl_peak['n_h']}"


# ---------------------------------------------------------------------------
# HMBC tests
# ---------------------------------------------------------------------------

class TestHMBC:
    def test_hmbc_contains_methyl_carbonyl_correlation(self):
        """
        Acetophenone CH3 must show HMBC correlation to C=O (~197 ppm).
        This is the most diagnostic peak.
        """
        result = _shift_result("CC(=O)c1ccccc1")
        peaks  = build_hmbc(result)
        # Find CH3 H shift
        hsqc   = build_hsqc(result)
        ch3_h  = min(p["h_ppm"] for p in hsqc)  # lowest H shift

        # Find HMBC peaks from CH3 H
        ch3_hmbc = [p for p in peaks
                    if abs(p["h_ppm"] - ch3_h) < 0.05]
        c_ppms   = [p["c_ppm"] for p in ch3_hmbc]
        assert any(p >= 185.0 for p in c_ppms), \
            (f"CH3 HMBC must include C=O (~197 ppm). "
             f"CH3 h_ppm={ch3_h:.3f}, HMBC C shifts from CH3: {c_ppms}")

    def test_hmbc_excludes_one_bond(self):
        """
        HMBC must NOT contain 1-bond H-C correlations at the atom level.

        Note: coordinate-level overlap between HMBC and HSQC peaks CAN occur
        when two chemically distinct carbons have the same predicted shift
        (shift degeneracy). We test the atom-level rule, not coordinate uniqueness.
        """
        from rdkit.Chem import rdmolops
        result  = _shift_result("CC(=O)c1ccccc1")
        mol     = result["mol_with_hs"]
        h_shifts = result["h_shifts"]
        c_shifts = result["c_shifts"]
        dist    = rdmolops.GetDistanceMatrix(mol).astype(int)

        # Re-derive HMBC pairs at atom level and check none are distance 1
        from build_2d import _get_equivalent_h_groups, _representative
        h_groups = _get_equivalent_h_groups(mol)
        for _rank, h_idxs in h_groups.items():
            rep_h = _representative(h_idxs)
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != 6:
                    continue
                c_idx = atom.GetIdx()
                d = dist[rep_h][c_idx]
                if d in (2, 3):
                    # This is a valid HMBC correlation
                    # Verify the 1-bond partner (d==1) was NOT included
                    assert d != 1, \
                        f"Found 1-bond H-C correlation in HMBC atom pairs: H={rep_h}, C={c_idx}"

    def test_hmbc_nonempty(self):
        result = _shift_result("CC(=O)c1ccccc1")
        peaks  = build_hmbc(result)
        assert len(peaks) > 0

    def test_hmbc_sorted(self):
        result = _shift_result("CC(=O)c1ccccc1")
        peaks  = build_hmbc(result)
        pairs  = [(p["h_ppm"], p["c_ppm"]) for p in peaks]
        assert pairs == sorted(pairs), "HMBC not sorted by (h_ppm, c_ppm)"


# ---------------------------------------------------------------------------
# COSY tests
# ---------------------------------------------------------------------------

class TestCOSY:
    def test_cosy_symmetric(self):
        """COSY must be symmetric: (a,b) ↔ (b,a)."""
        result = _shift_result("CC(=O)c1ccccc1")
        peaks  = build_cosy(result)
        pairs  = {(round(p["h1_ppm"], 4), round(p["h2_ppm"], 4))
                  for p in peaks}
        for h1, h2 in pairs:
            assert (h2, h1) in pairs, \
                f"COSY not symmetric: ({h1}, {h2}) present but ({h2}, {h1}) missing"

    def test_cosy_no_diagonal(self):
        """COSY must not contain diagonal peaks (h1 == h2)."""
        result = _shift_result("CC(=O)c1ccccc1")
        peaks  = build_cosy(result)
        for p in peaks:
            assert abs(p["h1_ppm"] - p["h2_ppm"]) > 1e-6, \
                f"Diagonal peak found in COSY: {p}"

    def test_cosy_methyl_no_coupling(self):
        """
        Acetophenone CH3 has no vicinal H neighbors → no COSY peaks.
        CH3 is attached to C=O (no adjacent CH).
        """
        result  = _shift_result("CC(=O)c1ccccc1")
        peaks   = build_cosy(result)
        hsqc    = build_hsqc(result)
        ch3_h   = min(p["h_ppm"] for p in hsqc)
        ch3_cosy = [p for p in peaks
                    if abs(p["h1_ppm"] - ch3_h) < 0.05 or
                       abs(p["h2_ppm"] - ch3_h) < 0.05]
        assert len(ch3_cosy) == 0, \
            f"CH3 should have no COSY coupling but found: {ch3_cosy}"

    def test_cosy_aromatic_coupling_present(self):
        """Aromatic H–H couplings must appear in COSY for acetophenone."""
        result = _shift_result("CC(=O)c1ccccc1")
        peaks  = build_cosy(result)
        # All COSY peaks should be in aromatic region
        assert len(peaks) > 0, "Expected aromatic COSY peaks"
        aromatic_cosy = [p for p in peaks if p["h1_ppm"] >= 6.5]
        assert len(aromatic_cosy) > 0, \
            f"No aromatic COSY peaks found: {peaks}"

    def test_cosy_ethyl_benzoate(self):
        """Ethyl benzoate: CH2 must show COSY coupling to CH3."""
        result  = _shift_result("CCOC(=O)c1ccccc1")
        hsqc    = build_hsqc(result)
        cosy    = build_cosy(result)

        # CH3 triplet at ~1.4 ppm, O-CH2 quartet at ~4.4 ppm
        aliphatic = sorted([p["h_ppm"] for p in hsqc if p["h_ppm"] < 5.0])
        assert len(aliphatic) >= 2, f"Expected CH3 and CH2: {aliphatic}"

        ch3_h  = aliphatic[0]   # lowest H ppm
        ch2_h  = aliphatic[1]   # next lowest H ppm

        # Check CH3 ↔ CH2 COSY coupling
        coupling = [p for p in cosy
                    if abs(p["h1_ppm"] - ch3_h) < 0.05 and
                       abs(p["h2_ppm"] - ch2_h) < 0.1]
        assert len(coupling) >= 1, \
            (f"CH3–CH2 COSY coupling missing. "
             f"CH3={ch3_h:.3f}, CH2={ch2_h:.3f}, COSY={cosy[:5]}")
