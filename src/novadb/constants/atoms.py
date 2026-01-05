"""
Atom definitions for standard residue types.

Contains atom names, backbone definitions, and reference frame atoms
for proteins, RNA, and DNA based on AlphaFold3 specifications.
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# Backbone Atoms
# =============================================================================

# Protein backbone atoms (N-terminus to C-terminus)
PROTEIN_BACKBONE_ATOMS: Final[tuple[str, ...]] = ("N", "CA", "C", "O")

# RNA backbone atoms (5' to 3')
RNA_BACKBONE_ATOMS: Final[tuple[str, ...]] = (
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'",
    "C3'", "O3'", "C2'", "O2'", "C1'",
)

# DNA backbone atoms (5' to 3') - no O2'
DNA_BACKBONE_ATOMS: Final[tuple[str, ...]] = (
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'",
    "C3'", "O3'", "C2'", "C1'",
)

# =============================================================================
# Reference Frame Atoms (for FAPE loss computation)
# =============================================================================

# Protein: N-CA-C frame (AF3 Section 3.8)
PROTEIN_FRAME_ATOMS: Final[tuple[str, str, str]] = ("N", "CA", "C")

# RNA/DNA: C1'-C3'-C4' frame (AF3 Section 3.8)
RNA_FRAME_ATOMS: Final[tuple[str, str, str]] = ("C1'", "C3'", "C4'")
DNA_FRAME_ATOMS: Final[tuple[str, str, str]] = ("C1'", "C3'", "C4'")

# CB virtual position atoms (for computing CB from backbone)
CB_POSITION_ATOMS: Final[tuple[str, str, str]] = ("N", "CA", "C")

# =============================================================================
# Standard Atoms Per Residue Type
# =============================================================================

# Heavy atoms only (no hydrogens), ordered consistently
RESIDUE_ATOMS: Final[dict[str, tuple[str, ...]]] = {
    # Amino Acids
    "ALA": ("N", "CA", "C", "O", "CB"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    "GLY": ("N", "CA", "C", "O"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD"),
    "SER": ("N", "CA", "C", "O", "CB", "OG"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    
    # Unknown amino acid (GLY-like)
    "UNK": ("N", "CA", "C", "O"),
}

# Atom name to index mapping per residue
RESIDUE_ATOM_INDEX: Final[dict[str, dict[str, int]]] = {
    restype: {atom: i for i, atom in enumerate(atoms)}
    for restype, atoms in RESIDUE_ATOMS.items()
}

# =============================================================================
# RNA Nucleotide Atoms
# =============================================================================

RNA_ATOMS: Final[dict[str, tuple[str, ...]]] = {
    "A": (
        # Backbone
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        # Base (Adenine)
        "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4",
    ),
    "C": (
        # Backbone
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        # Base (Cytosine)
        "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6",
    ),
    "G": (
        # Backbone
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        # Base (Guanine)
        "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
    ),
    "U": (
        # Backbone
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        # Base (Uracil)
        "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6",
    ),
}

# =============================================================================
# DNA Nucleotide Atoms
# =============================================================================

DNA_ATOMS: Final[dict[str, tuple[str, ...]]] = {
    "DA": (
        # Backbone (no O2')
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
        # Base (Adenine)
        "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4",
    ),
    "DC": (
        # Backbone
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
        # Base (Cytosine)
        "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6",
    ),
    "DG": (
        # Backbone
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
        # Base (Guanine)
        "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
    ),
    "DT": (
        # Backbone
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
        # Base (Thymine)
        "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6",  # C7 is the methyl
    ),
}

# =============================================================================
# Maximum Atoms Per Residue Type
# =============================================================================

MAX_PROTEIN_ATOMS: Final[int] = 14   # TRP has the most
MAX_RNA_ATOMS: Final[int] = 23       # G has the most (12 backbone + 11 base)
MAX_DNA_ATOMS: Final[int] = 22       # DG has the most (11 backbone + 11 base)
MAX_ATOMS_PER_TOKEN: Final[int] = 24  # AF3 filter threshold for ligands

# =============================================================================
# Chi Angle Definitions (Side Chain Torsions)
# =============================================================================

# Atoms defining each chi angle (4 atoms per angle)
CHI_ANGLES_ATOMS: Final[dict[str, list[tuple[str, str, str, str]]]] = {
    "ALA": [],  # No chi angles
    "ARG": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ"),
    ],
    "ASN": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1"),
    ],
    "ASP": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1"),
    ],
    "CYS": [
        ("N", "CA", "CB", "SG"),
    ],
    "GLN": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "GLU": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "GLY": [],  # No chi angles
    "HIS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "ND1"),
    ],
    "ILE": [
        ("N", "CA", "CB", "CG1"),
        ("CA", "CB", "CG1", "CD1"),
    ],
    "LEU": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "LYS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ"),
    ],
    "MET": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "SD"),
        ("CB", "CG", "SD", "CE"),
    ],
    "PHE": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "PRO": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
    ],
    "SER": [
        ("N", "CA", "CB", "OG"),
    ],
    "THR": [
        ("N", "CA", "CB", "OG1"),
    ],
    "TRP": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "TYR": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "VAL": [
        ("N", "CA", "CB", "CG1"),
    ],
}

# Maximum number of chi angles for any residue
MAX_CHI_ANGLES: Final[int] = 4

# Mask indicating which chi angles exist for each residue (for vectorization)
CHI_ANGLES_MASK: Final[dict[str, tuple[bool, bool, bool, bool]]] = {
    restype: tuple(
        i < len(CHI_ANGLES_ATOMS.get(restype, []))
        for i in range(MAX_CHI_ANGLES)
    )
    for restype in RESIDUE_ATOMS
}

# =============================================================================
# Symmetric Atoms (atoms that can be swapped without changing structure)
# =============================================================================

# These atom pairs can be swapped during evaluation to handle naming ambiguity
SYMMETRIC_ATOMS: Final[dict[str, list[tuple[str, str]]]] = {
    "ARG": [("NH1", "NH2")],
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
}

# =============================================================================
# Atom Order for Dense Representations
# =============================================================================

# Canonical atom ordering for all protein residues (padded to MAX_PROTEIN_ATOMS)
# This allows efficient dense tensor representations
ATOM14_ORDER: Final[tuple[str, ...]] = (
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1",
    "SG", "CD", "CD1", "CD2",
)

# Mapping from ATOM14 order to per-residue atom indices
ATOM14_TO_RESIDUE: Final[dict[str, dict[str, int]]] = {
    restype: {
        atom: ATOM14_ORDER.index(atom) if atom in ATOM14_ORDER else -1
        for atom in atoms
    }
    for restype, atoms in RESIDUE_ATOMS.items()
}
