"""
Element properties and constants.

Contains periodic table data, van der Waals radii, covalent radii,
and element classifications used in structural biology.
"""

from __future__ import annotations

from typing import Final, FrozenSet

# =============================================================================
# Element Vocabulary
# =============================================================================

# Element types commonly found in biomolecules, ordered by frequency
# Follows AF3 convention with 128 element types (most unused)
ELEMENTS: Final[tuple[str, ...]] = (
    # Common organic elements (0-5)
    "C", "N", "O", "S", "P", "H",
    # Halogens (6-9)
    "F", "CL", "BR", "I",
    # Common metals (10-19)
    "FE", "ZN", "MG", "CA", "MN", "CO", "NI", "CU", "NA", "K",
    # Less common elements (20-31)
    "SE", "MO", "W", "V", "CR", "CD", "HG", "PB", "AS", "B", "SI", "AL",
    # Unknown (32)
    "UNK",
)

# Element to index mapping
ELEMENT_ORDER: Final[dict[str, int]] = {
    elem: i for i, elem in enumerate(ELEMENTS)
}

NUM_ELEMENTS: Final[int] = len(ELEMENTS)

# Mapping from element symbol to internal atom type
ELEMENT_TO_ATOM_TYPE: Final[dict[str, int]] = ELEMENT_ORDER.copy()

# =============================================================================
# Van der Waals Radii (Å)
# =============================================================================

# From Bondi (1964) and others, commonly used in clash detection
VDW_RADII: Final[dict[str, float]] = {
    # Non-metals
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.90,
    "B": 1.92,
    "SI": 2.10,
    "AS": 1.85,
    
    # Alkali metals
    "LI": 1.82,
    "NA": 2.27,
    "K": 2.75,
    "RB": 3.03,
    "CS": 3.43,
    
    # Alkaline earth metals
    "BE": 1.53,
    "MG": 1.73,
    "CA": 2.31,
    "SR": 2.49,
    "BA": 2.68,
    
    # Transition metals
    "FE": 2.04,  # Fe2+/Fe3+
    "ZN": 1.39,
    "MN": 2.05,
    "CO": 2.00,
    "NI": 1.63,
    "CU": 1.40,
    "MO": 2.17,
    "W": 2.18,
    "V": 2.07,
    "CR": 2.06,
    "CD": 1.58,
    "HG": 1.55,
    "PB": 2.02,
    "AL": 1.84,
    
    # Default for unknown
    "UNK": 1.70,
}

# =============================================================================
# Covalent Radii (Å)
# =============================================================================

# For determining covalent bonds based on distance
COVALENT_RADII: Final[dict[str, float]] = {
    # Non-metals
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
    "SE": 1.20,
    "B": 0.84,
    "SI": 1.11,
    "AS": 1.19,
    
    # Alkali metals
    "LI": 1.28,
    "NA": 1.66,
    "K": 2.03,
    "RB": 2.20,
    "CS": 2.44,
    
    # Alkaline earth metals
    "BE": 0.96,
    "MG": 1.41,
    "CA": 1.76,
    "SR": 1.95,
    "BA": 2.15,
    
    # Transition metals
    "FE": 1.32,
    "ZN": 1.22,
    "MN": 1.39,
    "CO": 1.26,
    "NI": 1.24,
    "CU": 1.32,
    "MO": 1.54,
    "W": 1.62,
    "V": 1.53,
    "CR": 1.39,
    "CD": 1.44,
    "HG": 1.32,
    "PB": 1.46,
    "AL": 1.21,
    
    # Default for unknown
    "UNK": 1.50,
}

# =============================================================================
# Electronegativity (Pauling scale)
# =============================================================================

ELECTRONEGATIVITIES: Final[dict[str, float]] = {
    "H": 2.20,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "P": 2.19,
    "S": 2.58,
    "CL": 3.16,
    "BR": 2.96,
    "I": 2.66,
    "SE": 2.55,
    "B": 2.04,
    "SI": 1.90,
    "AS": 2.18,
    "NA": 0.93,
    "K": 0.82,
    "MG": 1.31,
    "CA": 1.00,
    "FE": 1.83,
    "ZN": 1.65,
    "MN": 1.55,
    "CO": 1.88,
    "NI": 1.91,
    "CU": 1.90,
}

# =============================================================================
# Element Classifications
# =============================================================================

# Metals commonly found in protein structures
METALS: Final[FrozenSet[str]] = frozenset([
    # Alkali metals
    "LI", "NA", "K", "RB", "CS",
    # Alkaline earth metals
    "BE", "MG", "CA", "SR", "BA",
    # Transition metals (common in proteins)
    "FE", "ZN", "MN", "CO", "NI", "CU", "MO", "W", "V", "CR",
    # Post-transition metals
    "AL", "CD", "HG", "PB",
])

# Elements that commonly form coordination complexes in proteins
COORDINATING_METALS: Final[FrozenSet[str]] = frozenset([
    "FE", "ZN", "MN", "CO", "NI", "CU", "MO", "MG", "CA",
])

# Halogens
HALOGENS: Final[FrozenSet[str]] = frozenset(["F", "CL", "BR", "I"])

# Organic elements (CHONPS)
ORGANIC_ELEMENTS: Final[FrozenSet[str]] = frozenset([
    "C", "H", "O", "N", "P", "S",
])

# Elements that can form hydrogen bonds
H_BOND_DONORS: Final[FrozenSet[str]] = frozenset(["N", "O", "S"])
H_BOND_ACCEPTORS: Final[FrozenSet[str]] = frozenset(["N", "O", "S", "F"])

# =============================================================================
# Atomic Masses (Daltons)
# =============================================================================

ATOMIC_MASSES: Final[dict[str, float]] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.065,
    "CL": 35.453,
    "BR": 79.904,
    "I": 126.904,
    "SE": 78.971,
    "B": 10.811,
    "SI": 28.086,
    "AS": 74.922,
    "NA": 22.990,
    "K": 39.098,
    "MG": 24.305,
    "CA": 40.078,
    "FE": 55.845,
    "ZN": 65.380,
    "MN": 54.938,
    "CO": 58.933,
    "NI": 58.693,
    "CU": 63.546,
    "MO": 95.950,
    "W": 183.840,
    "V": 50.942,
    "CR": 51.996,
    "CD": 112.414,
    "HG": 200.592,
    "PB": 207.200,
    "AL": 26.982,
}

# =============================================================================
# Common Oxidation States
# =============================================================================

COMMON_OXIDATION_STATES: Final[dict[str, tuple[int, ...]]] = {
    "FE": (2, 3),
    "ZN": (2,),
    "MG": (2,),
    "CA": (2,),
    "MN": (2, 3, 4),
    "CO": (2, 3),
    "NI": (2,),
    "CU": (1, 2),
    "NA": (1,),
    "K": (1,),
    "MO": (4, 5, 6),
}
