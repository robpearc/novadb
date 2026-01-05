"""
Bond definitions and constants.

Contains bond type definitions, standard bond lengths, and
polymer backbone connectivity information.
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# Bond Types
# =============================================================================

# Bond order types (for ligand and small molecule handling)
BOND_TYPE_ORDER: Final[dict[str, int]] = {
    "SINGLE": 0,
    "SING": 0,  # CCD format
    "DOUBLE": 1,
    "DOUB": 1,  # CCD format
    "TRIPLE": 2,
    "TRIP": 2,  # CCD format
    "AROMATIC": 3,
    "AROM": 3,  # CCD format
    "UNKNOWN": 4,
}

NUM_BOND_TYPES: Final[int] = 5

# =============================================================================
# Protein Backbone Bonds
# =============================================================================

# Bonds within a protein residue (intra-residue)
PROTEIN_BACKBONE_BONDS: Final[tuple[tuple[str, str], ...]] = (
    ("N", "CA"),
    ("CA", "C"),
    ("C", "O"),
)

# Peptide bond connecting residues (C to next N)
PEPTIDE_BOND_ATOMS: Final[tuple[str, str]] = ("C", "N")

# All standard intra-residue bonds for amino acids
PROTEIN_RESIDUE_BONDS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "ALA": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
    ),
    "ARG": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"),
        ("CZ", "NH1"), ("CZ", "NH2"),
    ),
    "ASN": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2"),
    ),
    "ASP": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2"),
    ),
    "CYS": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "SG"),
    ),
    "GLN": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2"),
    ),
    "GLU": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2"),
    ),
    "GLY": (
        ("N", "CA"), ("CA", "C"), ("C", "O"),
    ),
    "HIS": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"),
        ("CD2", "NE2"), ("CE1", "NE2"),
    ),
    "ILE": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1"),
    ),
    "LEU": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
    ),
    "LYS": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ"),
    ),
    "MET": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "SD"), ("SD", "CE"),
    ),
    "PHE": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"),
        ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"),
    ),
    "PRO": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "CD"), ("CD", "N"),  # Proline ring
    ),
    "SER": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "OG"),
    ),
    "THR": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "OG1"), ("CB", "CG2"),
    ),
    "TRP": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"),
        ("NE1", "CE2"), ("CD2", "CE2"), ("CD2", "CE3"), ("CE2", "CZ2"),
        ("CE3", "CZ3"), ("CZ2", "CH2"), ("CZ3", "CH2"),
    ),
    "TYR": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"),
        ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH"),
    ),
    "VAL": (
        ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
        ("CB", "CG1"), ("CB", "CG2"),
    ),
}

# =============================================================================
# RNA Backbone Bonds
# =============================================================================

# Bonds within RNA backbone (intra-residue)
RNA_BACKBONE_BONDS: Final[tuple[tuple[str, str], ...]] = (
    ("P", "OP1"),
    ("P", "OP2"),
    ("P", "O5'"),
    ("O5'", "C5'"),
    ("C5'", "C4'"),
    ("C4'", "O4'"),
    ("C4'", "C3'"),
    ("C3'", "O3'"),
    ("C3'", "C2'"),
    ("C2'", "O2'"),
    ("C2'", "C1'"),
    ("C1'", "O4'"),
)

# Phosphodiester bond connecting RNA nucleotides
RNA_PHOSPHODIESTER_BOND: Final[tuple[str, str]] = ("O3'", "P")

# Base attachment point
RNA_BASE_BOND: Final[dict[str, tuple[str, str]]] = {
    "A": ("C1'", "N9"),
    "G": ("C1'", "N9"),
    "C": ("C1'", "N1"),
    "U": ("C1'", "N1"),
}

# =============================================================================
# DNA Backbone Bonds
# =============================================================================

# Bonds within DNA backbone (intra-residue) - no O2'
DNA_BACKBONE_BONDS: Final[tuple[tuple[str, str], ...]] = (
    ("P", "OP1"),
    ("P", "OP2"),
    ("P", "O5'"),
    ("O5'", "C5'"),
    ("C5'", "C4'"),
    ("C4'", "O4'"),
    ("C4'", "C3'"),
    ("C3'", "O3'"),
    ("C3'", "C2'"),
    ("C2'", "C1'"),
    ("C1'", "O4'"),
)

# Phosphodiester bond connecting DNA nucleotides
DNA_PHOSPHODIESTER_BOND: Final[tuple[str, str]] = ("O3'", "P")

# Base attachment point
DNA_BASE_BOND: Final[dict[str, tuple[str, str]]] = {
    "DA": ("C1'", "N9"),
    "DG": ("C1'", "N9"),
    "DC": ("C1'", "N1"),
    "DT": ("C1'", "N1"),
}

# =============================================================================
# Standard Bond Lengths (Å)
# =============================================================================

# Equilibrium bond lengths for common bond types
STANDARD_BOND_LENGTHS: Final[dict[tuple[str, str], float]] = {
    # Carbon-carbon
    ("C", "C"): 1.54,      # Single bond (sp3-sp3)
    ("C", "=C"): 1.34,     # Double bond
    ("C", "#C"): 1.20,     # Triple bond
    ("C", "C_AR"): 1.40,   # Aromatic
    
    # Carbon-nitrogen
    ("C", "N"): 1.47,      # Single bond
    ("C", "=N"): 1.29,     # Double bond (imine)
    ("C", "N_AR"): 1.34,   # Aromatic
    ("CA", "N"): 1.46,     # Peptide backbone
    
    # Carbon-oxygen
    ("C", "O"): 1.43,      # Single bond (ether, alcohol)
    ("C", "=O"): 1.23,     # Double bond (carbonyl)
    ("C", "O_AR"): 1.37,   # Aromatic
    
    # Carbon-sulfur
    ("C", "S"): 1.82,      # Single bond
    
    # Nitrogen-hydrogen
    ("N", "H"): 1.01,
    
    # Oxygen-hydrogen
    ("O", "H"): 0.96,
    
    # Sulfur-hydrogen
    ("S", "H"): 1.34,
    
    # Carbon-hydrogen
    ("C", "H"): 1.09,
    
    # Phosphorus-oxygen
    ("P", "O"): 1.61,      # Single bond
    ("P", "=O"): 1.48,     # Double bond
    
    # Specific backbone bonds
    ("N", "CA"): 1.458,    # N-Cα
    ("CA", "C"): 1.525,    # Cα-C
    ("C", "N"): 1.329,     # Peptide bond (partial double)
    ("C", "O"): 1.231,     # Carbonyl
    
    # Disulfide bond
    ("S", "S"): 2.03,
}

# =============================================================================
# Bond Detection Parameters
# =============================================================================

# Tolerance for detecting covalent bonds from distance
BOND_LENGTH_TOLERANCE: Final[float] = 0.4  # Å

# More permissive tolerance for edge cases
COVALENT_BOND_TOLERANCE: Final[float] = 0.56  # Å (based on covalent radii)

# Maximum distance to consider for any bond
MAX_BOND_DISTANCE: Final[float] = 2.5  # Å

# =============================================================================
# Special Bond Types
# =============================================================================

# Disulfide bond detection parameters
DISULFIDE_BOND_LENGTH: Final[float] = 2.03  # Å
DISULFIDE_BOND_TOLERANCE: Final[float] = 0.3  # Å

# Metal coordination bond parameters
METAL_COORDINATION_DISTANCE: Final[float] = 2.8  # Å (maximum)

# Common metal coordination geometries
METAL_COORDINATION_NUMBERS: Final[dict[str, tuple[int, ...]]] = {
    "ZN": (4,),           # Tetrahedral
    "FE": (4, 5, 6),      # Various (heme, etc.)
    "MG": (6,),           # Octahedral
    "CA": (6, 7, 8),      # Various
    "MN": (4, 5, 6),      # Various
    "CO": (4, 6),         # Tetrahedral or octahedral
    "NI": (4, 6),         # Square planar or octahedral
    "CU": (4, 5),         # Various
}
