"""
tests/test_known_molecules.py — Regression tests against published NMR data.

These tests verify topology/count correctness, not exact ppm values
(predicted shifts have ±0.3 ppm for 1H, ±2 ppm for 13C).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
import pytest
from predict_shifts import predict_shifts
from build_2d import build_hsqc, build_hmbc, build_cosy
from generate_problem import generate_problem


# ---------------------------------------------------------------------------
# Cached fixtures
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _problem(smiles: str, pid: str):
    return generate_problem(smiles, pid, backend="nmrdb", collapse=True)


# ---------------------------------------------------------------------------
# Acetophenone
# ---------------------------------------------------------------------------

class TestAcetophenone:
    SMILES = "CC(=O)c1ccccc1"

    def test_hsqc_count(self):
        p = _problem(self.SMILES, "t_acetophenone")
        assert len(p["spectra"]["hsqc"]) == 4, \
            f"Expected 4 HSQC peaks: {p['spectra']['hsqc']}"

    def test_hmbc_has_carbonyl(self):
        p = _problem(self.SMILES, "t_acetophenone")
        carbonyl_peaks = [pk for pk in p["spectra"]["hmbc"]
                          if pk["c_ppm"] >= 185.0]
        assert len(carbonyl_peaks) >= 1, \
            "Must have at least one HMBC peak to C=O"

    def test_cosy_symmetric(self):
        p    = _problem(self.SMILES, "t_acetophenone")
        cosy = p["spectra"]["cosy"]
        pairs = {(round(pk["h1_ppm"], 3), round(pk["h2_ppm"], 3))
                 for pk in cosy}
        for h1, h2 in pairs:
            assert (h2, h1) in pairs, \
                f"COSY asymmetric: ({h1}, {h2}) but ({h2}, {h1}) missing"

    def test_formula_and_dou(self):
        p = _problem(self.SMILES, "t_acetophenone")
        assert p["molecular_formula"] == "C8H8O"
        assert p["degree_of_unsaturation"] == 5


# ---------------------------------------------------------------------------
# Ethyl benzoate
# ---------------------------------------------------------------------------

class TestEthylBenzoate:
    SMILES = "CCOC(=O)c1ccccc1"

    def test_hsqc_count(self):
        p = _problem(self.SMILES, "t_ethylbenzoate")
        # ortho, meta, para (3 ring environments) + O-CH2 + CH3 = 5
        assert len(p["spectra"]["hsqc"]) == 5, \
            f"Expected 5 HSQC peaks: {p['spectra']['hsqc']}"

    def test_cosy_ethyl_coupling(self):
        p    = _problem(self.SMILES, "t_ethylbenzoate")
        cosy = p["spectra"]["cosy"]
        hsqc = p["spectra"]["hsqc"]
        # Aliphatic HSQC peaks
        aliphatic = sorted([pk["h_ppm"] for pk in hsqc if pk["h_ppm"] < 5.0])
        assert len(aliphatic) >= 2, "Expected CH3 and O-CH2 in aliphatic region"
        ch3_h = aliphatic[0]
        ch2_h = aliphatic[1]
        # CH3 ↔ CH2 coupling
        coupling = any(
            abs(pk["h1_ppm"] - ch3_h) < 0.05 and
            abs(pk["h2_ppm"] - ch2_h) < 0.10
            for pk in cosy
        )
        assert coupling, \
            f"CH3–OCH2 COSY coupling missing. ch3={ch3_h:.3f}, ch2={ch2_h:.3f}"


# ---------------------------------------------------------------------------
# Benzene
# ---------------------------------------------------------------------------

class TestBenzene:
    SMILES = "c1ccccc1"

    def test_hsqc_count(self):
        result = predict_shifts(self.SMILES, backend="nmrdb")
        peaks  = build_hsqc(result)
        assert len(peaks) == 1, f"Benzene should have 1 HSQC peak, got {len(peaks)}"

    def test_hsqc_n_h_6(self):
        result = predict_shifts(self.SMILES, backend="nmrdb")
        peaks  = build_hsqc(result)
        assert peaks[0]["n_h"] == 6, \
            f"Benzene HSQC n_h should be 6, got {peaks[0]['n_h']}"

    def test_cosy_empty(self):
        """Benzene: all H equivalent → no COSY cross-peaks (same equivalence group)."""
        result = predict_shifts(self.SMILES, backend="nmrdb")
        peaks  = build_cosy(result)
        assert len(peaks) == 0, \
            f"Benzene has only one equivalence group → no COSY. Got: {peaks}"


# ---------------------------------------------------------------------------
# Caffeine
# ---------------------------------------------------------------------------

class TestCaffeine:
    SMILES = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"

    def test_hsqc_count(self):
        """Caffeine: 3 N-CH3 groups + 1 vinyl H = 4 HSQC peaks."""
        p = _problem(self.SMILES, "t_caffeine")
        assert len(p["spectra"]["hsqc"]) == 4, \
            f"Expected 4 HSQC peaks: {p['spectra']['hsqc']}"

    def test_no_oh_nh_in_hsqc(self):
        """Caffeine has no OH or NH protons → no exchangeable H in HSQC."""
        p = _problem(self.SMILES, "t_caffeine")
        for pk in p["spectra"]["hsqc"]:
            assert pk["h_ppm"] <= 10.0, \
                f"Unexpectedly high H shift for caffeine: {pk}"


# ---------------------------------------------------------------------------
# Vanillin
# ---------------------------------------------------------------------------

class TestVanillin:
    SMILES = "COc1cc(C=O)ccc1O"

    def test_has_aldehyde_h_in_hsqc(self):
        """Vanillin aldehyde H should appear in HSQC at ~9.8 ppm."""
        result = predict_shifts(self.SMILES, backend="nmrdb")
        peaks  = build_hsqc(result)
        aldehyde = [p for p in peaks if p["h_ppm"] >= 9.0]
        assert len(aldehyde) >= 1, \
            f"Expected aldehyde HSQC peak. Got: {[p['h_ppm'] for p in peaks]}"

    def test_hmbc_aldehyde_to_ring(self):
        """Aldehyde H must show HMBC correlation to aromatic ring carbons."""
        result = predict_shifts(self.SMILES, backend="nmrdb")
        hmbc   = build_hmbc(result)
        hsqc   = build_hsqc(result)
        ald_h  = max(p["h_ppm"] for p in hsqc)
        ald_hmbc = [p for p in hmbc
                    if abs(p["h_ppm"] - ald_h) < 0.05
                    and 110.0 <= p["c_ppm"] <= 145.0]
        assert len(ald_hmbc) >= 1, \
            f"Aldehyde H should have HMBC to ring C. ald_h={ald_h:.2f}"


# ---------------------------------------------------------------------------
# Score answer tests (no API needed)
# ---------------------------------------------------------------------------

class TestScoreAnswer:
    def test_exact_match(self):
        from score_answer import score_smiles
        r = score_smiles("CC(=O)c1ccccc1", "CC(=O)c1ccccc1")
        assert r["exact_match"]
        assert r["tanimoto"] == 1.0

    def test_different_smiles(self):
        from score_answer import score_smiles
        r = score_smiles("CC(=O)c1ccccc1", "c1ccccc1")
        assert not r["exact_match"]
        assert r["tanimoto"] < 1.0

    def test_invalid_smiles(self):
        from score_answer import score_smiles
        r = score_smiles("not_valid", "CC(=O)c1ccccc1")
        assert not r["valid_smiles"]
        assert not r["exact_match"]
        assert r["tanimoto"] == 0.0

    def test_formula_match(self):
        from score_answer import score_smiles
        # Same formula, different structure
        r = score_smiles("CCCCCC", "CC(CC)CC")  # both C6H14
        assert r["formula_match"]
        assert not r["exact_match"]

    def test_mw_delta_exact(self):
        from score_answer import score_smiles
        r = score_smiles("CC(=O)c1ccccc1", "CC(=O)c1ccccc1")
        assert r["mw_delta"] == 0.0
