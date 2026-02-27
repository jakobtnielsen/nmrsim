"""
tests/test_collapse.py — Unit tests for collapse_peaks.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from collapse_peaks import collapse_2d_peaks


class TestCollapseHSQC:
    def test_no_peaks(self):
        assert collapse_2d_peaks([], "hsqc") == []

    def test_single_peak_unchanged(self):
        peaks = [{"h_ppm": 7.32, "c_ppm": 128.5, "n_h": 2}]
        out   = collapse_2d_peaks(peaks, "hsqc")
        assert len(out) == 1
        assert out[0]["h_ppm"] == 7.32
        assert out[0]["c_ppm"] == 128.5
        assert out[0]["n_h"]   == 2

    def test_within_tolerance_merged(self):
        peaks = [
            {"h_ppm": 7.320, "c_ppm": 128.50, "n_h": 1},
            {"h_ppm": 7.330, "c_ppm": 128.60, "n_h": 1},  # Δh=0.01, Δc=0.1
        ]
        out = collapse_2d_peaks(peaks, "hsqc", tol_h=0.02, tol_c=0.50)
        assert len(out) == 1, f"Expected 1 merged peak, got {len(out)}"
        assert out[0]["n_h"] == 2

    def test_outside_tolerance_separate(self):
        peaks = [
            {"h_ppm": 7.32, "c_ppm": 128.5, "n_h": 1},
            {"h_ppm": 7.50, "c_ppm": 128.5, "n_h": 1},  # Δh=0.18 > tol
        ]
        out = collapse_2d_peaks(peaks, "hsqc", tol_h=0.02, tol_c=0.50)
        assert len(out) == 2

    def test_centroid_correct(self):
        peaks = [
            {"h_ppm": 7.30, "c_ppm": 128.0, "n_h": 1},
            {"h_ppm": 7.32, "c_ppm": 128.4, "n_h": 3},
        ]
        out = collapse_2d_peaks(peaks, "hsqc", tol_h=0.05, tol_c=1.0)
        assert len(out) == 1
        assert abs(out[0]["h_ppm"] - 7.31)  < 0.001
        assert abs(out[0]["c_ppm"] - 128.2) < 0.001
        assert out[0]["n_h"] == 4

    def test_output_sorted_by_h_ppm(self):
        peaks = [
            {"h_ppm": 8.00, "c_ppm": 130.0, "n_h": 1},
            {"h_ppm": 2.50, "c_ppm":  29.0, "n_h": 3},
            {"h_ppm": 7.40, "c_ppm": 128.0, "n_h": 1},
        ]
        out = collapse_2d_peaks(peaks, "hsqc")
        h_ppms = [p["h_ppm"] for p in out]
        assert h_ppms == sorted(h_ppms)


class TestCollapseHMBC:
    def test_hmbc_no_n_h_in_output(self):
        peaks = [
            {"h_ppm": 7.32, "c_ppm": 128.5},
            {"h_ppm": 7.34, "c_ppm": 128.6},  # within tol
        ]
        out = collapse_2d_peaks(peaks, "hmbc", tol_h=0.05, tol_c=0.50)
        for p in out:
            assert "n_h" not in p, "HMBC peaks should not have n_h field"

    def test_hmbc_two_peaks_far_apart(self):
        peaks = [
            {"h_ppm": 2.60, "c_ppm": 197.4},
            {"h_ppm": 7.40, "c_ppm": 128.5},
        ]
        out = collapse_2d_peaks(peaks, "hmbc")
        assert len(out) == 2


class TestCollapseCOSY:
    def test_cosy_merged_within_tolerance(self):
        # Two peaks within tolerance: (7.30,7.50) and (7.31,7.51).
        # After normalization (both already h1<h2), they cluster into one.
        # The symmetric counterpart (7.505,7.305) is also emitted → 2 peaks total.
        peaks = [
            {"h1_ppm": 7.30, "h2_ppm": 7.50},
            {"h1_ppm": 7.31, "h2_ppm": 7.51},  # Δh1=0.01, Δh2=0.01
        ]
        out = collapse_2d_peaks(peaks, "cosy", tol_h=0.02, tol_h2=0.02)
        assert len(out) == 2
        # One peak should have h1 ≈ 7.305
        h1_vals = sorted(p["h1_ppm"] for p in out)
        assert abs(h1_vals[0] - 7.305) < 0.001

    def test_cosy_not_merged_different_h2(self):
        # Two non-mergeable peaks; each emits its symmetric counterpart → 4 total.
        peaks = [
            {"h1_ppm": 7.30, "h2_ppm": 7.50},
            {"h1_ppm": 7.31, "h2_ppm": 8.00},  # h2 differs too much to merge
        ]
        out = collapse_2d_peaks(peaks, "cosy", tol_h=0.02, tol_h2=0.02)
        assert len(out) == 4

    def test_cosy_symmetric_input_preserved(self):
        # Real COSY: symmetric input (a,b) + (b,a) should collapse to 2 peaks.
        peaks = [
            {"h1_ppm": 1.24, "h2_ppm": 1.27},
            {"h1_ppm": 1.27, "h2_ppm": 1.24},
        ]
        out = collapse_2d_peaks(peaks, "cosy")
        assert len(out) == 2
        pairs = {(p["h1_ppm"], p["h2_ppm"]) for p in out}
        # Both directions must be present
        assert any(p["h1_ppm"] < p["h2_ppm"] for p in out)
        assert any(p["h1_ppm"] > p["h2_ppm"] for p in out)

    def test_invalid_experiment_raises(self):
        with pytest.raises(ValueError):
            collapse_2d_peaks([], "xyzzy")
