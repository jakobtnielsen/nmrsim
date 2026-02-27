"""
collapse_peaks.py — Stage 4: merge 2D peaks within instrument resolution limits.

Implements greedy clustering as specified in NMR_SKILL.md Section 5.
"""


def collapse_2d_peaks(
    peaks: list,
    experiment: str,
    tol_h: float = 0.02,
    tol_c: float = 0.50,
    tol_h2: float = 0.02,
) -> list:
    """
    Merge 2D peaks that fall within resolution tolerances.

    Args:
        peaks:      Peak list in the format returned by build_hsqc/build_hmbc/build_cosy.
        experiment: "hsqc" | "hmbc" | "cosy"
        tol_h:      Tolerance in ppm for the H dimension (default 0.02).
        tol_c:      Tolerance in ppm for the C dimension (default 0.50, HSQC/HMBC only).
        tol_h2:     Tolerance for the second H dimension in COSY (default 0.02).

    Returns:
        Collapsed peak list in the same format as input.

    Merge criteria:
        HSQC/HMBC: |Δh_ppm| < tol_h  AND  |Δc_ppm| < tol_c
        COSY:      |Δh1_ppm| < tol_h  AND  |Δh2_ppm| < tol_h2

    Centroid of merged cluster = mean of all peaks' coordinates.
    For HSQC: n_h of merged cluster = sum of constituent n_h.
    """
    if experiment not in ("hsqc", "hmbc", "cosy"):
        raise ValueError(f"Unknown experiment type: {experiment!r}. "
                         "Expected 'hsqc', 'hmbc', or 'cosy'.")

    if not peaks:
        return []

    if experiment in ("hsqc", "hmbc"):
        return _collapse_hc(peaks, experiment, tol_h, tol_c)
    else:  # cosy
        return _collapse_hh(peaks, tol_h, tol_h2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collapse_hc(peaks: list, experiment: str,
                 tol_h: float, tol_c: float) -> list:
    """Greedy clustering for H–C experiments (HSQC or HMBC)."""
    is_hsqc = (experiment == "hsqc")

    # Clusters: each cluster stores {h_ppm_sum, c_ppm_sum, n_peaks, n_h_sum}
    clusters = []

    for peak in peaks:
        h = peak["h_ppm"]
        c = peak["c_ppm"]
        n_h = peak.get("n_h", 1)

        # Check if this peak falls within any existing cluster's centroid
        merged = False
        for cl in clusters:
            cen_h = cl["h_sum"] / cl["n"]
            cen_c = cl["c_sum"] / cl["n"]
            if abs(h - cen_h) < tol_h and abs(c - cen_c) < tol_c:
                # Merge into this cluster
                cl["h_sum"] += h
                cl["c_sum"] += c
                cl["n"]     += 1
                cl["n_h"]   += n_h
                merged = True
                break

        if not merged:
            clusters.append({
                "h_sum": h,
                "c_sum": c,
                "n":     1,
                "n_h":   n_h,
            })

    # Emit one peak per cluster
    result = []
    for cl in clusters:
        n = cl["n"]
        peak = {
            "h_ppm": round(cl["h_sum"] / n, 4),
            "c_ppm": round(cl["c_sum"] / n, 4),
        }
        if is_hsqc:
            peak["n_h"] = cl["n_h"]
        result.append(peak)

    key = (lambda p: (p["h_ppm"], p["c_ppm"]))
    return sorted(result, key=key)


def _collapse_hh(peaks: list, tol_h: float, tol_h2: float) -> list:
    """
    Greedy clustering for H–H experiment (COSY).

    Normalise each peak so h1 ≤ h2 before clustering (symmetric pairs are
    identical in normalised form). After clustering, re-emit both (h1, h2) and
    (h2, h1) for off-diagonal peaks, preserving COSY symmetry.
    """
    clusters = []

    for peak in peaks:
        # Normalise: always store the smaller shift first
        a, b = peak["h1_ppm"], peak["h2_ppm"]
        h1, h2 = (a, b) if a <= b else (b, a)

        merged = False
        for cl in clusters:
            cen_h1 = cl["h1_sum"] / cl["n"]
            cen_h2 = cl["h2_sum"] / cl["n"]
            if abs(h1 - cen_h1) < tol_h and abs(h2 - cen_h2) < tol_h2:
                cl["h1_sum"] += h1
                cl["h2_sum"] += h2
                cl["n"]      += 1
                merged = True
                break

        if not merged:
            clusters.append({"h1_sum": h1, "h2_sum": h2, "n": 1})

    result = []
    for cl in clusters:
        n = cl["n"]
        h1 = round(cl["h1_sum"] / n, 4)
        h2 = round(cl["h2_sum"] / n, 4)
        result.append({"h1_ppm": h1, "h2_ppm": h2})
        if h1 != h2:
            result.append({"h1_ppm": h2, "h2_ppm": h1})

    return sorted(result, key=lambda p: (p["h1_ppm"], p["h2_ppm"]))
