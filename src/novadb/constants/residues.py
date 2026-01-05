"""
Residue and token type definitions.

Contains standard amino acid, nucleotide, and token vocabulary definitions
based on AlphaFold3 supplement specifications.
"""

from __future__ import annotations

from typing import Final, FrozenSet

# =============================================================================
# Standard Amino Acids
# =============================================================================

# One-letter codes (canonical order)
AMINO_ACIDS_1: Final[str] = "ACDEFGHIKLMNPQRSTVWY"

# Three-letter codes (matching order)
AMINO_ACIDS_3: Final[tuple[str, ...]] = (
    "ALA", "CYS", "ASP", "GLU", "PHE",
    "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG",
    "SER", "THR", "VAL", "TRP", "TYR",
)

# Mappings between 1-letter and 3-letter codes
AA_1_TO_3: Final[dict[str, str]] = dict(zip(AMINO_ACIDS_1, AMINO_ACIDS_3, strict=True))
AA_3_TO_1: Final[dict[str, str]] = dict(zip(AMINO_ACIDS_3, AMINO_ACIDS_1, strict=True))

# =============================================================================
# Nucleotides
# =============================================================================

# RNA nucleotides (3-letter codes used in mmCIF)
RNA_NUCLEOTIDES: Final[tuple[str, ...]] = ("A", "C", "G", "U")
RNA_NUCLEOTIDES_1: Final[str] = "ACGU"

# DNA nucleotides (3-letter codes used in mmCIF)
DNA_NUCLEOTIDES: Final[tuple[str, ...]] = ("DA", "DC", "DG", "DT")
DNA_NUCLEOTIDES_1: Final[str] = "ACGT"

# =============================================================================
# Token Vocabulary (AlphaFold3 - 32 types)
# =============================================================================

# Full token vocabulary matching AF3 Table 5
# Order: 20 amino acids + 4 RNA + 4 DNA + 4 special tokens = 32
TOKEN_TYPES: Final[tuple[str, ...]] = (
    # Standard amino acids (0-19)
    "ALA", "CYS", "ASP", "GLU", "PHE",
    "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG",
    "SER", "THR", "VAL", "TRP", "TYR",
    # RNA nucleotides (20-23)
    "A", "C", "G", "U",
    # DNA nucleotides (24-27)
    "DA", "DC", "DG", "DT",
    # Special tokens (28-31)
    "UNK",     # Unknown residue
    "GAP",     # Gap in alignment
    "MASK",    # Masked token (for training)
    "LIGAND",  # Generic ligand token
)

# Residue type to index mapping
RESTYPE_ORDER: Final[dict[str, int]] = {
    restype: i for i, restype in enumerate(TOKEN_TYPES)
}

NUM_TOKEN_TYPES: Final[int] = 32
UNKNOWN_RESTYPE: Final[int] = RESTYPE_ORDER["UNK"]

# Standard residues (non-special tokens)
STANDARD_RESIDUES: Final[FrozenSet[str]] = frozenset(
    list(AMINO_ACIDS_3) + list(RNA_NUCLEOTIDES) + list(DNA_NUCLEOTIDES)
)

# =============================================================================
# Amino Acid Properties
# =============================================================================

HYDROPHOBIC_AAS: Final[FrozenSet[str]] = frozenset([
    "ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO",
])

POLAR_AAS: Final[FrozenSet[str]] = frozenset([
    "SER", "THR", "ASN", "GLN", "TYR", "CYS",
])

CHARGED_AAS: Final[FrozenSet[str]] = frozenset([
    "ASP", "GLU",  # Negative
    "LYS", "ARG", "HIS",  # Positive
])

AROMATIC_AAS: Final[FrozenSet[str]] = frozenset([
    "PHE", "TYR", "TRP", "HIS",
])

# =============================================================================
# Modified Residues (to standard mapping)
# =============================================================================

MODIFIED_RESIDUE_MAP: Final[dict[str, str]] = {
    # Selenomethionine
    "MSE": "MET",
    "MET": "MET",  # Identity for convenience
    
    # Modified lysines
    "MLY": "LYS",  # N-dimethyl-lysine
    "MLZ": "LYS",  # N-methyl-lysine
    "M3L": "LYS",  # N-trimethyl-lysine
    "ALY": "LYS",  # N-acetyl-lysine
    
    # Modified methionines
    "FME": "MET",  # N-formylmethionine
    "SME": "MET",  # Methionine sulfoxide
    
    # Modified prolines
    "HYP": "PRO",  # 4-hydroxyproline
    "DPR": "PRO",  # D-proline
    
    # Phosphorylated residues
    "TPO": "THR",  # Phosphothreonine
    "SEP": "SER",  # Phosphoserine
    "PTR": "TYR",  # Phosphotyrosine
    
    # Modified cysteines
    "CSO": "CYS",  # S-hydroxycysteine
    "OCS": "CYS",  # Cysteinesulfonic acid
    "CME": "CYS",  # S,S-(2-hydroxyethyl)thiocysteine
    "CSS": "CYS",  # S-mercaptocysteine
    "SEC": "CYS",  # Selenocysteine (technically different)
    "CSD": "CYS",  # S-cysteinesulfinic acid
    "CSX": "CYS",  # S-oxy cysteine
    "SCY": "CYS",  # S-acetyl-cysteine
    
    # Modified serines
    "SAC": "SER",  # N-acetyl-serine
    "SVA": "SER",  # Serine vanadate
    
    # Modified glutamates
    "PCA": "GLU",  # Pyroglutamic acid
    "CGU": "GLU",  # Gamma-carboxy-glutamic acid
    
    # Modified aspartates
    "BHD": "ASP",  # Beta-hydroxyaspartic acid
    
    # Modified arginines
    "AGM": "ARG",  # 4-guanidino-butanoic acid
    "MMA": "ARG",  # N-methyl-arginine
    
    # Modified histidines
    "HIC": "HIS",  # Histidine-methyl ester
    "NEP": "HIS",  # N1-phosphonohistidine
    
    # Modified tyrosines
    "TYS": "TYR",  # Sulfonated tyrosine
    "TPQ": "TYR",  # 2,4,5-trihydroxyphenylalanine
    
    # Modified phenylalanines
    "PHD": "PHE",  # 2-amino-4-phenyl-butyric acid
    
    # Modified tryptophans
    "TRO": "TRP",  # 2-hydroxy-tryptophan
    "HTR": "TRP",  # Beta-hydroxy-tryptophan
    
    # Modified asparagines
    "MEN": "ASN",  # N-methyl-asparagine
    
    # Modified glutamines
    "MEQ": "GLN",  # N-methyl-glutamine
    
    # Modified alanines
    "DAL": "ALA",  # D-alanine
    "ABA": "ALA",  # Alpha-aminobutyric acid
    "AIB": "ALA",  # Alpha-aminoisobutyric acid
    
    # Modified valines
    "DVA": "VAL",  # D-valine
    "MVA": "VAL",  # N-methyl-valine
    
    # Modified leucines
    "DLE": "LEU",  # D-leucine
    "MLE": "LEU",  # N-methyl-leucine
    "NLE": "LEU",  # Norleucine
    
    # Modified isoleucines
    "DIL": "ILE",  # D-isoleucine
    "IIL": "ILE",  # Iso-isoleucine
    
    # Modified glycines
    "SAR": "GLY",  # Sarcosine (N-methylglycine)
    "GL3": "GLY",  # Alpha-amino-glycine
    
    # D-amino acids (generic)
    "DAR": "ARG",
    "DAS": "ASP",
    "DCY": "CYS",
    "DGN": "GLN",
    "DGL": "GLU",
    "DHI": "HIS",
    "DLY": "LYS",
    "DME": "MET",  # D-methionine (not MSE)
    "DPN": "PHE",
    "DSN": "SER",
    "DTH": "THR",
    "DTR": "TRP",
    "DTY": "TYR",
}

# =============================================================================
# Crystallization Aids and Common Artifacts
# =============================================================================

# Small molecules commonly added during crystallization (to filter out)
# Based on AF3 Section 2.5.1 and common PDB artifacts
CRYSTALLIZATION_AIDS: Final[FrozenSet[str]] = frozenset([
    # Sulfates and phosphates
    "SO4", "PO4", "NO3", "CO3",
    
    # Polyethylene glycols
    "PEG", "PG4", "PE4", "P6G", "1PE", "2PE",
    "PGE", "PGR", "EPE",
    
    # Glycols and alcohols
    "GOL", "EDO", "MPD", "EOH", "MOH", "IPA",
    "PGO", "PDO", "12P", "15P",
    
    # Buffers
    "MES", "TRS", "HEP", "EPE", "TAM", "ACA",
    "CIT", "ACT", "ACY", "FMT", "MLI",
    
    # Cryoprotectants
    "DMS", "DMF", "NMF",
    
    # Detergents
    "LDA", "SDS", "OLC", "BOG", "LMT",
    "DDQ", "C8E", "C10",
    
    # Reducing agents
    "BME", "DTT", "TCE",
    
    # Other common additives
    "IMD",  # Imidazole
    "URE",  # Urea
    "SCN",  # Thiocyanate
    "AZI",  # Azide
    "IOD",  # Iodide
    "BR",   # Bromide
    "CL",   # Chloride
    "F",    # Fluoride
])

# Common ions (not always artifacts, but often)
COMMON_IONS: Final[FrozenSet[str]] = frozenset([
    # Monovalent
    "NA", "K", "LI", "RB", "CS",
    "CL", "BR", "F", "I",
    
    # Divalent
    "MG", "CA", "ZN", "FE", "MN", "CO", "NI", "CU",
    "CD", "BA", "SR",
    
    # Other
    "FE2", "FE3", "CU1", "CU2",
])
