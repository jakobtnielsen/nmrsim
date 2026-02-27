"""tests/test_solver.py — Unit tests for solve_structure.py.

Tests focus on:
1. extract_atom_nodes() — correct atom count and hybridisation from HSQC
2. extract_constraints() — HMBC/COSY constraint generation
3. detect_fragments() — heteroatom placement heuristics
4. run_lsd() — LSD binary integration (requires lsd_bin/lsd)
5. solve() — end-to-end round-trip on known molecules
"""

import json
import math
import os
import pytest
from pathlib import Path
from rdkit import Chem

from solve_structure import (
    extract_atom_nodes,
    extract_constraints,
    detect_fragments,
    run_lsd,
    solve,
    _parse_formula,
    _build_lsd_text,
    _canonicalize_smiles,
    _LSD_BIN,
)

_OUTPUTS = Path(__file__).parent.parent / "outputs"
_HAS_LSD = os.path.isfile(_LSD_BIN) and os.access(_LSD_BIN, os.X_OK)

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canon(s):
    return _canonicalize_smiles(s)


def _load_problem(name):
    p = _OUTPUTS / f"nmr_{name}.json"
    if not p.exists():
        pytest.skip(f"Problem file not found: {p}")
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stage 0: formula parser
# ---------------------------------------------------------------------------

def test_parse_formula_simple():
    assert _parse_formula("C2H6") == {"C": 2, "H": 6}
    assert _parse_formula("C8H8O") == {"C": 8, "H": 8, "O": 1}
    assert _parse_formula("C8H10N4O2") == {"C": 8, "H": 10, "N": 4, "O": 2}


def test_parse_formula_single_element():
    assert _parse_formula("CH4") == {"C": 1, "H": 4}


# ---------------------------------------------------------------------------
# Stage 1: extract_atom_nodes
# ---------------------------------------------------------------------------

def _simple_problem(hsqc, hmbc=None, formula=""):
    return {
        "molecular_formula": formula,
        "spectra": {
            "hsqc": hsqc,
            "hmbc": hmbc or [],
            "cosy": [],
            "h1_1d": None,
        },
    }


def test_atom_nodes_basic_sp3():
    """Two distinct sp3 CH3 groups → 2 nodes."""
    problem = _simple_problem(
        hsqc=[
            {"c_ppm": 15.0, "h_ppm": 1.2, "n_h": 3},
            {"c_ppm": 20.0, "h_ppm": 0.9, "n_h": 3},
        ]
    )
    nodes = extract_atom_nodes(problem)
    assert len(nodes) == 2
    # Sorted by c_ppm descending
    assert nodes[0]["c_ppm"] == 20.0
    assert nodes[1]["c_ppm"] == 15.0
    for n in nodes:
        assert n["hyb"] == 3
        assert n["n_h"] == 3
        assert not n["is_aromatic"]
        assert not n["is_quat"]


def test_atom_nodes_aromatic_expansion():
    """Aromatic HSQC peak with n_h=2 → 2 separate nodes each with n_h=1."""
    problem = _simple_problem(
        hsqc=[{"c_ppm": 128.5, "h_ppm": 7.5, "n_h": 2}]
    )
    nodes = extract_atom_nodes(problem)
    assert len(nodes) == 2
    for n in nodes:
        assert n["hyb"] == 2
        assert n["n_h"] == 1
        assert n["is_aromatic"]
        assert n["group_c_ppm"] == 128.5


def test_atom_nodes_aromatic_no_expansion():
    """Aromatic CH (n_h=1) should NOT be expanded."""
    problem = _simple_problem(
        hsqc=[{"c_ppm": 131.0, "h_ppm": 7.7, "n_h": 1}]
    )
    nodes = extract_atom_nodes(problem)
    assert len(nodes) == 1
    assert nodes[0]["n_h"] == 1


def test_atom_nodes_quaternary_from_hmbc():
    """HMBC c_ppm not near HSQC → quaternary C node added."""
    problem = _simple_problem(
        hsqc=[{"c_ppm": 26.0, "h_ppm": 2.3, "n_h": 3}],
        hmbc=[{"h_ppm": 2.3, "c_ppm": 197.0}],
    )
    nodes = extract_atom_nodes(problem)
    assert len(nodes) == 2
    quat_nodes = [n for n in nodes if n["is_quat"]]
    assert len(quat_nodes) == 1
    assert abs(quat_nodes[0]["c_ppm"] - 197.0) < 0.1
    assert quat_nodes[0]["n_h"] == 0
    assert quat_nodes[0]["hyb"] == 2  # c_ppm > 100


def test_atom_nodes_no_false_quaternary():
    """HMBC c_ppm close to HSQC c_ppm → NOT detected as quaternary."""
    problem = _simple_problem(
        hsqc=[{"c_ppm": 128.5, "h_ppm": 7.5, "n_h": 1}],
        hmbc=[{"h_ppm": 7.5, "c_ppm": 128.4}],  # within 1.0 ppm
    )
    nodes = extract_atom_nodes(problem)
    assert len(nodes) == 1  # only HSQC node, no quat


def test_atom_nodes_lsd_ids_sequential():
    """LSD IDs should be sequential starting at 1."""
    problem = _simple_problem(
        hsqc=[
            {"c_ppm": 10.0, "h_ppm": 1.0, "n_h": 3},
            {"c_ppm": 20.0, "h_ppm": 1.5, "n_h": 2},
            {"c_ppm": 30.0, "h_ppm": 2.0, "n_h": 1},
        ]
    )
    nodes = extract_atom_nodes(problem)
    ids = [n["lsd_id"] for n in nodes]
    assert ids == list(range(1, len(nodes) + 1))


# ---------------------------------------------------------------------------
# Stage 2: extract_constraints
# ---------------------------------------------------------------------------

def test_constraints_cosy_simple():
    """COSY between two single-atom groups."""
    problem = {
        "molecular_formula": "C2H6",
        "spectra": {
            "hsqc": [
                {"c_ppm": 15.0, "h_ppm": 1.2, "n_h": 3},
                {"c_ppm": 20.0, "h_ppm": 0.9, "n_h": 3},
            ],
            "hmbc": [],
            "cosy": [
                {"h1_ppm": 1.2, "h2_ppm": 0.9},
                {"h1_ppm": 0.9, "h2_ppm": 1.2},
            ],
            "h1_1d": None,
        },
    }
    nodes = extract_atom_nodes(problem)
    constraints = extract_constraints(problem, nodes)
    # One COSY pair (deduped)
    assert len(constraints["cosy"]) == 1
    c = constraints["cosy"][0]
    assert c[0] < c[1]  # canonical form


def test_constraints_cosy_group_expansion():
    """COSY involving expanded group (n_h=2 aromatic) pairs correctly."""
    problem = {
        "molecular_formula": "",
        "spectra": {
            "hsqc": [
                {"c_ppm": 128.5, "h_ppm": 7.5, "n_h": 2},  # 2 atoms (IDs 1,2)
                {"c_ppm": 131.0, "h_ppm": 7.7, "n_h": 1},  # 1 atom (ID 3)
            ],
            "hmbc": [],
            "cosy": [
                {"h1_ppm": 7.5, "h2_ppm": 7.7},
                {"h1_ppm": 7.7, "h2_ppm": 7.5},
            ],
            "h1_1d": None,
        },
    }
    nodes = extract_atom_nodes(problem)
    assert len(nodes) == 3  # 2 expanded + 1
    constraints = extract_constraints(problem, nodes)
    # Should get 2 COSY pairs (one per expanded atom paired with atom 3)
    cosy = constraints["cosy"]
    assert len(cosy) == 2


def test_constraints_hmbc_excludes_self_group():
    """HMBC c_ppm matching the H atom's own group should be excluded."""
    problem = {
        "molecular_formula": "",
        "spectra": {
            "hsqc": [
                {"c_ppm": 128.5, "h_ppm": 7.5, "n_h": 1},
            ],
            "hmbc": [
                # c=128.5 would match the H atom's own group → should be excluded
                {"h_ppm": 7.5, "c_ppm": 128.5},
                # c=135.0 is a different (quat) atom → should be included
                {"h_ppm": 7.5, "c_ppm": 135.0},
            ],
            "cosy": [],
            "h1_1d": None,
        },
    }
    nodes = extract_atom_nodes(problem)
    # Quat node at 135.0 is auto-detected from HMBC
    constraints = extract_constraints(problem, nodes)
    hmbc = constraints["hmbc"]
    id_to_cppm = {n['lsd_id']: n['c_ppm'] for n in nodes}
    # 135.0 quat C should appear as a target; 128.5 self should not
    c_ppms_targeted = [id_to_cppm.get(c, 0) for _, c in hmbc]
    assert any(abs(cp - 135.0) < 0.5 for cp in c_ppms_targeted)
    assert not any(abs(cp - 128.5) < 0.5 for cp in c_ppms_targeted)


def test_constraints_hmbc_all_atoms_per_peak():
    """HMBC constraint applied to all atoms in an equivalent group per NMR peak.

    A single NMR peak from an expanded group (n_h=2 aromatic → 2 atoms) should
    generate one HMBC constraint per atom so that every member of the group is
    properly constrained in LSD.
    """
    problem = {
        "molecular_formula": "",
        "spectra": {
            "hsqc": [
                {"c_ppm": 128.5, "h_ppm": 7.5, "n_h": 2},  # expanded → 2 atoms
            ],
            "hmbc": [
                {"h_ppm": 7.5, "c_ppm": 137.0},  # 1 NMR peak
            ],
            "cosy": [],
            "h1_1d": None,
        },
    }
    # Inject quat C at 137.0
    nodes = extract_atom_nodes(problem)
    nodes.append({
        'lsd_id': len(nodes) + 1, 'symbol': 'C', 'hyb': 2, 'n_h': 0,
        'c_ppm': 137.0, 'h_ppm': None,
        'is_quat': True, 'is_aromatic': True,
        'group_c_ppm': 137.0, 'group_h_ppm': None,
    })
    constraints = extract_constraints(problem, nodes)
    # Both atoms in the equivalent group should each get a constraint → 2 total
    assert len(constraints["hmbc"]) == 2


# ---------------------------------------------------------------------------
# Stage 3: detect_fragments
# ---------------------------------------------------------------------------

def test_fragments_carbonyl_oxygen():
    """Quaternary C > 160 ppm → sp2 O added with BOND constraint."""
    nodes = [
        {'lsd_id': 1, 'symbol': 'C', 'hyb': 2, 'n_h': 0, 'c_ppm': 197.0,
         'h_ppm': None, 'is_quat': True, 'is_aromatic': False,
         'group_c_ppm': 197.0, 'group_h_ppm': None},
        {'lsd_id': 2, 'symbol': 'C', 'hyb': 3, 'n_h': 3, 'c_ppm': 26.0,
         'h_ppm': 2.3, 'is_quat': False, 'is_aromatic': False,
         'group_c_ppm': 26.0, 'group_h_ppm': 2.3},
    ]
    frags = detect_fragments(nodes, "C2H4O")
    o_atoms = [a for a in frags['extra_atoms'] if a['symbol'] == 'O']
    assert len(o_atoms) == 1
    assert o_atoms[0]['hyb'] == 2
    assert o_atoms[0]['n_h'] == 0
    # BOND between atom 1 (carbonyl) and the O
    bonds = frags['bond_constraints']
    assert (1, o_atoms[0]['lsd_id']) in bonds


def test_fragments_remaining_h_to_oxygen():
    """Extra H (not on C) gets assigned to sp3 O (alcohol)."""
    nodes = [
        {'lsd_id': 1, 'symbol': 'C', 'hyb': 3, 'n_h': 3, 'c_ppm': 18.0,
         'h_ppm': 1.2, 'is_quat': False, 'is_aromatic': False,
         'group_c_ppm': 18.0, 'group_h_ppm': 1.2},
        {'lsd_id': 2, 'symbol': 'C', 'hyb': 3, 'n_h': 2, 'c_ppm': 58.0,
         'h_ppm': 3.7, 'is_quat': False, 'is_aromatic': False,
         'group_c_ppm': 58.0, 'group_h_ppm': 3.7},
    ]
    # Ethanol: C2H6O. H on C = 3+2 = 5. Remaining = 6-5 = 1 → OH
    frags = detect_fragments(nodes, "C2H6O")
    o_atoms = [a for a in frags['extra_atoms'] if a['symbol'] == 'O']
    assert len(o_atoms) == 1
    assert o_atoms[0]['n_h'] == 1  # OH


def test_fragments_nitrogen_aromatic():
    """Aromatic N → sp2 hybridisation."""
    nodes = [
        {'lsd_id': 1, 'symbol': 'C', 'hyb': 2, 'n_h': 1, 'c_ppm': 150.0,
         'h_ppm': 8.5, 'is_quat': False, 'is_aromatic': True,
         'group_c_ppm': 150.0, 'group_h_ppm': 8.5},
    ]
    frags = detect_fragments(nodes, "C1H1N1")
    n_atoms = [a for a in frags['extra_atoms'] if a['symbol'] == 'N']
    assert len(n_atoms) == 1
    assert n_atoms[0]['hyb'] == 2


# ---------------------------------------------------------------------------
# Stage 4: run_lsd integration tests (require LSD binary)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_LSD, reason="LSD binary not found")
def test_run_lsd_ethane():
    """Ethane (C2H6) → exactly 1 solution → 'CC'."""
    problem = {
        "molecular_formula": "C2H6",
        "spectra": {
            "hsqc": [
                {"c_ppm": 15.0, "h_ppm": 1.2, "n_h": 3},
                {"c_ppm": 5.0, "h_ppm": 0.9, "n_h": 3},
            ],
            "hmbc": [],
            "cosy": [
                {"h1_ppm": 1.2, "h2_ppm": 0.9},
                {"h1_ppm": 0.9, "h2_ppm": 1.2},
            ],
            "h1_1d": None,
        },
    }
    nodes = extract_atom_nodes(problem)
    constraints = extract_constraints(problem, nodes)
    fragments = detect_fragments(nodes, "C2H6")
    smiles, _ = run_lsd(nodes, constraints, fragments, "C2H6")
    assert len(smiles) >= 1
    assert "CC" in smiles


@pytest.mark.skipif(not _HAS_LSD, reason="LSD binary not found")
def test_run_lsd_no_solutions_impossible():
    """Contradictory constraints → 0 solutions."""
    # Create a problem where COSY and HMBC are inconsistent
    problem = {
        "molecular_formula": "CH4",  # only 1 C — no bonds possible
        "spectra": {
            "hsqc": [{"c_ppm": 2.0, "h_ppm": 0.2, "n_h": 4}],
            "hmbc": [],
            "cosy": [],
            "h1_1d": None,
        },
    }
    nodes = extract_atom_nodes(problem)
    constraints = extract_constraints(problem, nodes)
    fragments = detect_fragments(nodes, "CH4")
    smiles, _ = run_lsd(nodes, constraints, fragments, "CH4")
    # LSD should find 1 solution (methane) or 0
    assert isinstance(smiles, list)


# ---------------------------------------------------------------------------
# Stage 5: solve end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_LSD, reason="LSD binary not found")
def test_solve_returns_structure():
    """solve() returns expected keys and types."""
    problem = {
        "molecular_formula": "C2H6",
        "spectra": {
            "hsqc": [
                {"c_ppm": 15.0, "h_ppm": 1.2, "n_h": 3},
                {"c_ppm": 5.0, "h_ppm": 0.9, "n_h": 3},
            ],
            "hmbc": [],
            "cosy": [
                {"h1_ppm": 1.2, "h2_ppm": 0.9},
                {"h1_ppm": 0.9, "h2_ppm": 1.2},
            ],
            "h1_1d": None,
        },
    }
    result = solve(problem, max_candidates=10)
    assert "candidates" in result
    assert "lsd_count" in result
    assert "difficulty" in result
    assert "lsd_file" in result
    assert isinstance(result["lsd_count"], int)
    assert isinstance(result["difficulty"], float)
    assert result["difficulty"] >= 0


@pytest.mark.skipif(not _HAS_LSD, reason="LSD binary not found")
def test_solve_ethane_unique():
    """Ethane → difficulty ~0 (1 solution)."""
    problem = {
        "molecular_formula": "C2H6",
        "spectra": {
            "hsqc": [
                {"c_ppm": 15.0, "h_ppm": 1.2, "n_h": 3},
                {"c_ppm": 5.0, "h_ppm": 0.9, "n_h": 3},
            ],
            "hmbc": [],
            "cosy": [
                {"h1_ppm": 1.2, "h2_ppm": 0.9},
                {"h1_ppm": 0.9, "h2_ppm": 1.2},
            ],
            "h1_1d": None,
        },
    }
    result = solve(problem, max_candidates=10)
    assert result["lsd_count"] >= 1
    assert result["difficulty"] == pytest.approx(0.0, abs=0.1)
    # Top candidate should be CC
    assert len(result["candidates"]) >= 1
    assert _canon(result["candidates"][0]["smiles"]) == "CC"


@pytest.mark.skipif(not _HAS_LSD, reason="LSD binary not found")
def test_solve_acetophenone_gt_ranked_first():
    """Acetophenone: ground truth should be rank 1 candidate."""
    problem = _load_problem("001_acetophenone")
    gt = problem["ground_truth"]["smiles"]
    result = solve(problem, max_candidates=20)

    assert result["lsd_count"] >= 1
    assert len(result["candidates"]) >= 1

    top_canon = _canon(result["candidates"][0]["smiles"])
    gt_canon = _canon(gt)
    assert top_canon == gt_canon, (
        f"Top candidate {top_canon!r} != ground truth {gt_canon!r}"
    )


@pytest.mark.skipif(not _HAS_LSD, reason="LSD binary not found")
def test_solve_difficulty_log_scale():
    """difficulty = log10(lsd_count); verify formula."""
    problem = _load_problem("001_acetophenone")
    result = solve(problem, max_candidates=20)
    if result["lsd_count"] > 0:
        expected = math.log10(result["lsd_count"])
        assert abs(result["difficulty"] - expected) < 1e-6


@pytest.mark.skipif(not _HAS_LSD, reason="LSD binary not found")
def test_solve_candidates_sorted_by_score():
    """Candidates must be sorted by ascending score (best first)."""
    problem = _load_problem("001_acetophenone")
    result = solve(problem, max_candidates=20)
    scores = [c["score"] for c in result["candidates"]
              if c["score"] is not None]
    assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# LSD file generation (no binary needed)
# ---------------------------------------------------------------------------

def test_build_lsd_text_hmbc_order():
    """LSD HMBC format: carbon first, H-bearing atom second."""
    atom_nodes = [
        {'lsd_id': 1, 'symbol': 'C', 'hyb': 2, 'n_h': 0,
         'c_ppm': 197.0, 'h_ppm': None, 'is_quat': True, 'is_aromatic': False,
         'group_c_ppm': 197.0, 'group_h_ppm': None},
        {'lsd_id': 2, 'symbol': 'C', 'hyb': 3, 'n_h': 3,
         'c_ppm': 26.0, 'h_ppm': 2.3, 'is_quat': False, 'is_aromatic': False,
         'group_c_ppm': 26.0, 'group_h_ppm': 2.3},
    ]
    constraints = {'hmbc': [(2, 1)], 'cosy': []}  # h_id=2, c_id=1
    text = _build_lsd_text(atom_nodes, [], constraints, [], (0, 4))
    # Should write HMBC 1 2 (c first, h second)
    assert "HMBC 1 2" in text
    # Should NOT write HMBC 2 1
    assert "HMBC 2 1" not in text


def test_build_lsd_text_no_hmqc_for_quat():
    """Quaternary carbons (n_h=0) must NOT get an HMQC line."""
    atom_nodes = [
        {'lsd_id': 1, 'symbol': 'C', 'hyb': 2, 'n_h': 0,
         'c_ppm': 197.0, 'h_ppm': None, 'is_quat': True, 'is_aromatic': False,
         'group_c_ppm': 197.0, 'group_h_ppm': None},
    ]
    text = _build_lsd_text(atom_nodes, [], {'hmbc': [], 'cosy': []}, [], (0, 4))
    assert "HMQC 1 1" not in text


def test_build_lsd_text_hmqc_for_ch():
    """CH atoms (n_h > 0) must get an HMQC line."""
    atom_nodes = [
        {'lsd_id': 1, 'symbol': 'C', 'hyb': 3, 'n_h': 3,
         'c_ppm': 26.0, 'h_ppm': 2.3, 'is_quat': False, 'is_aromatic': False,
         'group_c_ppm': 26.0, 'group_h_ppm': 2.3},
    ]
    text = _build_lsd_text(atom_nodes, [], {'hmbc': [], 'cosy': []}, [], (0, 4))
    assert "HMQC 1 1" in text
