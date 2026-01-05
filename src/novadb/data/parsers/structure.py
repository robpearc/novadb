"""Core structure data classes for representing biomolecular structures.

This module provides dataclasses for atoms, residues, chains, and complete
structures, following the conventions used in AlphaFold3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


class ChainType(Enum):
    """Types of molecular chains.
    
    From AF3 supplement Table 13 (standard residues) and Section 2.6 (tokenization).
    """
    PROTEIN = auto()
    RNA = auto()
    DNA = auto()
    LIGAND = auto()
    ION = auto()
    WATER = auto()
    UNKNOWN = auto()


# Standard amino acids (Table 13)
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK"  # Unknown amino acid
}

# Standard nucleotides (Table 13)
STANDARD_RNA_NUCLEOTIDES = {"A", "G", "C", "U", "N"}
STANDARD_DNA_NUCLEOTIDES = {"DA", "DG", "DC", "DT", "DN"}
STANDARD_NUCLEOTIDES = STANDARD_RNA_NUCLEOTIDES | STANDARD_DNA_NUCLEOTIDES

# One-letter codes for residues (32 classes as per AF3)
RESIDUE_TYPES = [
    # 20 standard amino acids + unknown
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "X",
    # 4 RNA nucleotides + unknown
    "RA", "RG", "RC", "RU", "RX",
    # 4 DNA nucleotides + unknown
    "DA", "DG", "DC", "DT", "DX",
    # Gap
    "-"
]


# Crystallization aids to remove (Table 9)
CRYSTALLIZATION_AIDS = {
    "SO4", "GOL", "EDO", "PO4", "ACT", "PEG", "DMS", "TRS", "PGE", "PG4",
    "FMT", "EPE", "MPD", "MES", "CD", "IOD"
}

# Ligand exclusion list (Table 10)
LIGAND_EXCLUSION_LIST = {
    "144", "15P", "1PE", "2F2", "2JC", "3HR", "3SY", "7N5", "7PE", "9JE",
    "AAE", "ABA", "ACE", "ACN", "ACT", "ACY", "AZI", "BAM", "BCN", "BCT",
    "BDN", "BEN", "BME", "BO3", "BTB", "BTC", "BU1", "C8E", "CAD", "CAQ",
    "CBM", "CCN", "CIT", "CL", "CLR", "CM", "CMO", "CO3", "CPT", "CXS",
    "D10", "DEP", "DIO", "DMS", "DN", "DOD", "DOX", "EDO", "EEE", "EGL",
    "EOH", "EOX", "EPE", "ETF", "FCY", "FJO", "FLC", "FMT", "FW5", "GOL",
    "GSH", "GTT", "GYF", "HED", "IHP", "IHS", "IMD", "IOD", "IPA", "IPH",
    "LDA", "MB3", "MEG", "MES", "MLA", "MLI", "MOH", "MPD", "MRD", "MSE",
    "MYR", "N", "NA", "NH2", "NH4", "NHE", "NO3", "O4B", "OHE", "OLA",
    "OLC", "OMB", "OME", "OXA", "P6G", "PE3", "PE4", "PEG", "PEO", "PEP",
    "PG0", "PG4", "PGE", "PGR", "PLM", "PO4", "POL", "POP", "PVO", "SAR",
    "SCN", "SEO", "SEP", "SIN", "SO4", "SPD", "SPM", "SR", "STE", "STO",
    "STU", "TAR", "TBU", "TME", "TPO", "TRS", "UNK", "UNL", "UNX", "UPL", "URE"
}

# Ion CCD codes (Table 12) - partial list
ION_CCD_CODES = {
    "AG", "AL", "AU", "BA", "BR", "CA", "CD", "CE", "CL", "CO", "CR", "CS",
    "CU", "DY", "EU", "F", "FE", "GA", "GD", "HG", "IN", "IR", "K", "LA",
    "LI", "LU", "MG", "MN", "MO", "NA", "NI", "OS", "PB", "PD", "PR", "PT",
    "RB", "RU", "SB", "SM", "SR", "TB", "TH", "TL", "V", "W", "Y", "YB", "ZN", "ZR"
}


@dataclass
class Atom:
    """Represents a single atom in a structure.
    
    Attributes:
        name: Atom name (e.g., 'CA', 'N', 'C1')
        element: Element symbol (e.g., 'C', 'N', 'O')
        coords: 3D coordinates in Angstroms
        occupancy: Occupancy factor (0-1)
        b_factor: Temperature factor
        charge: Formal charge
        is_hetero: Whether this is a HETATM
        alt_loc: Alternative location indicator
        serial: Atom serial number
    """
    name: str
    element: str
    coords: np.ndarray  # Shape (3,)
    occupancy: float = 1.0
    b_factor: float = 0.0
    charge: int = 0
    is_hetero: bool = False
    alt_loc: str = ""
    serial: int = 0

    def __post_init__(self):
        if not isinstance(self.coords, np.ndarray):
            self.coords = np.array(self.coords, dtype=np.float32)
        assert self.coords.shape == (3,), f"Coords must be shape (3,), got {self.coords.shape}"

    @property
    def is_hydrogen(self) -> bool:
        """Check if this is a hydrogen atom."""
        return self.element == "H"

    @property
    def is_heavy(self) -> bool:
        """Check if this is a heavy (non-hydrogen) atom."""
        return self.element != "H"

    def distance_to(self, other: "Atom") -> float:
        """Calculate Euclidean distance to another atom."""
        return float(np.linalg.norm(self.coords - other.coords))


@dataclass
class Residue:
    """Represents a residue (amino acid, nucleotide, or ligand).
    
    Attributes:
        name: Residue name (3-letter code, e.g., 'ALA', 'DA')
        seq_id: Sequence position (residue number)
        insertion_code: PDB insertion code
        atoms: Dictionary of atom name to Atom
        is_standard: Whether this is a standard residue
    """
    name: str
    seq_id: int
    atoms: Dict[str, Atom] = field(default_factory=dict)
    insertion_code: str = ""
    is_standard: bool = True

    @property
    def num_atoms(self) -> int:
        """Number of atoms in this residue."""
        return len(self.atoms)

    @property
    def num_heavy_atoms(self) -> int:
        """Number of heavy (non-hydrogen) atoms."""
        return sum(1 for atom in self.atoms.values() if atom.is_heavy)

    @property
    def chain_type(self) -> ChainType:
        """Determine the chain type based on residue name."""
        if self.name in STANDARD_AMINO_ACIDS:
            return ChainType.PROTEIN
        elif self.name in STANDARD_RNA_NUCLEOTIDES or self.name in {"A", "G", "C", "U"}:
            return ChainType.RNA
        elif self.name in STANDARD_DNA_NUCLEOTIDES or self.name in {"DA", "DG", "DC", "DT"}:
            return ChainType.DNA
        elif self.name in ION_CCD_CODES:
            return ChainType.ION
        elif self.name == "HOH":
            return ChainType.WATER
        else:
            return ChainType.LIGAND

    def get_atom(self, name: str) -> Optional[Atom]:
        """Get an atom by name."""
        return self.atoms.get(name)

    def get_ca(self) -> Optional[Atom]:
        """Get the Cα atom (for proteins)."""
        return self.atoms.get("CA")

    def get_cb(self) -> Optional[Atom]:
        """Get the Cβ atom (or Cα for glycine)."""
        if self.name == "GLY":
            return self.get_ca()
        return self.atoms.get("CB")

    def get_c1_prime(self) -> Optional[Atom]:
        """Get the C1' atom (for nucleotides)."""
        return self.atoms.get("C1'")

    def get_center_atom(self) -> Optional[Atom]:
        """Get the token center atom as defined in AF3 Section 2.6.
        
        - Cα for standard amino acids
        - C1' for standard nucleotides
        - First atom for ligands (single atom per token)
        """
        chain_type = self.chain_type
        if chain_type == ChainType.PROTEIN:
            return self.get_ca()
        elif chain_type in (ChainType.RNA, ChainType.DNA):
            return self.get_c1_prime()
        else:
            # For ligands, return first atom
            if self.atoms:
                return next(iter(self.atoms.values()))
            return None

    def get_backbone_atoms(self) -> Dict[str, Atom]:
        """Get backbone atoms for frame construction.
        
        From AF3 Section 4.3.2:
        - Proteins: (N, Cα, C)
        - Nucleotides: (C1', C3', C4')
        """
        chain_type = self.chain_type
        if chain_type == ChainType.PROTEIN:
            names = ["N", "CA", "C"]
        elif chain_type in (ChainType.RNA, ChainType.DNA):
            names = ["C1'", "C3'", "C4'"]
        else:
            return {}

        return {name: self.atoms[name] for name in names if name in self.atoms}

    @property
    def heavy_atom_coords(self) -> np.ndarray:
        """Get coordinates of all heavy atoms as Nx3 array."""
        coords = [atom.coords for atom in self.atoms.values() if atom.is_heavy]
        if coords:
            return np.stack(coords)
        return np.zeros((0, 3), dtype=np.float32)


@dataclass
class Chain:
    """Represents a molecular chain.
    
    Attributes:
        chain_id: PDB chain identifier (e.g., 'A', 'B')
        residues: List of residues in sequence order
        entity_id: Entity identifier (same sequence = same entity)
        chain_type: Type of chain (protein, RNA, DNA, ligand)
    """
    chain_id: str
    residues: List[Residue] = field(default_factory=list)
    entity_id: int = 0
    chain_type: Optional[ChainType] = None

    def __post_init__(self):
        if self.chain_type is None and self.residues:
            self.chain_type = self._infer_chain_type()

    def _infer_chain_type(self) -> ChainType:
        """Infer chain type from residues."""
        if not self.residues:
            return ChainType.UNKNOWN

        # Count residue types
        type_counts = {}
        for res in self.residues:
            res_type = res.chain_type
            type_counts[res_type] = type_counts.get(res_type, 0) + 1

        # Return most common type
        if type_counts:
            return max(type_counts, key=type_counts.get)
        return ChainType.UNKNOWN

    @property
    def sequence(self) -> str:
        """Get the sequence as a string."""
        seq = []
        for res in self.residues:
            if self.chain_type == ChainType.PROTEIN:
                seq.append(self._aa_to_one_letter(res.name))
            elif self.chain_type in (ChainType.RNA, ChainType.DNA):
                seq.append(self._nucleotide_to_one_letter(res.name))
            else:
                seq.append("X")
        return "".join(seq)

    @staticmethod
    def _aa_to_one_letter(three_letter: str) -> str:
        """Convert 3-letter amino acid code to 1-letter."""
        mapping = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        }
        return mapping.get(three_letter, "X")

    @staticmethod
    def _nucleotide_to_one_letter(code: str) -> str:
        """Convert nucleotide code to 1-letter."""
        mapping = {
            "A": "A", "G": "G", "C": "C", "U": "U",
            "DA": "A", "DG": "G", "DC": "C", "DT": "T",
        }
        return mapping.get(code, "N")

    @property
    def num_residues(self) -> int:
        """Number of residues in chain."""
        return len(self.residues)

    @property
    def num_atoms(self) -> int:
        """Total number of atoms in chain."""
        return sum(res.num_atoms for res in self.residues)

    @property
    def num_heavy_atoms(self) -> int:
        """Number of heavy atoms in chain."""
        return sum(res.num_heavy_atoms for res in self.residues)

    def get_residue(self, seq_id: int, insertion_code: str = "") -> Optional[Residue]:
        """Get residue by sequence ID and insertion code."""
        for res in self.residues:
            if res.seq_id == seq_id and res.insertion_code == insertion_code:
                return res
        return None

    def get_ca_coords(self) -> np.ndarray:
        """Get Cα coordinates as Nx3 array."""
        coords = []
        for res in self.residues:
            ca = res.get_ca()
            if ca is not None:
                coords.append(ca.coords)
        if coords:
            return np.stack(coords)
        return np.zeros((0, 3), dtype=np.float32)

    def get_all_heavy_atom_coords(self) -> np.ndarray:
        """Get all heavy atom coordinates as Nx3 array."""
        coords = []
        for res in self.residues:
            for atom in res.atoms.values():
                if atom.is_heavy:
                    coords.append(atom.coords)
        if coords:
            return np.stack(coords)
        return np.zeros((0, 3), dtype=np.float32)

    @property
    def is_peptide(self) -> bool:
        """Check if this is a peptide (protein with <16 residues).

        From AF3 supplement Section 6.1.
        """
        return self.chain_type == ChainType.PROTEIN and self.num_residues < 16

    def get_sequence(self) -> str:
        """Get the sequence as a string (alias for sequence property)."""
        return self.sequence


@dataclass
class Bond:
    """Represents a covalent bond between atoms.
    
    Attributes:
        chain1_id: First chain ID
        res1_seq_id: First residue sequence ID
        atom1_name: First atom name
        chain2_id: Second chain ID
        res2_seq_id: Second residue sequence ID
        atom2_name: Second atom name
        bond_order: Bond order (1=single, 2=double, 3=triple)
    """
    chain1_id: str
    res1_seq_id: int
    atom1_name: str
    chain2_id: str
    res2_seq_id: int
    atom2_name: str
    bond_order: int = 1


@dataclass
class Structure:
    """Represents a complete biomolecular structure.
    
    Attributes:
        pdb_id: PDB identifier
        chains: Dictionary of chain ID to Chain
        resolution: Structure resolution in Angstroms
        method: Experimental method (X-RAY, NMR, CRYO-EM)
        release_date: PDB release date
        bonds: List of inter-residue/inter-chain bonds
    """
    pdb_id: str
    chains: Dict[str, Chain] = field(default_factory=dict)
    resolution: Optional[float] = None
    method: Optional[str] = None
    release_date: Optional[str] = None
    bonds: List[Bond] = field(default_factory=list)

    # Metadata
    title: Optional[str] = None
    authors: Optional[List[str]] = None

    @property
    def num_chains(self) -> int:
        """Number of chains."""
        return len(self.chains)

    @property
    def num_residues(self) -> int:
        """Total number of residues."""
        return sum(chain.num_residues for chain in self.chains.values())

    @property
    def num_atoms(self) -> int:
        """Total number of atoms."""
        return sum(chain.num_atoms for chain in self.chains.values())

    def get_chain(self, chain_id: str) -> Optional[Chain]:
        """Get chain by ID."""
        return self.chains.get(chain_id)

    def get_polymer_chains(self) -> List[Chain]:
        """Get all polymer (protein/RNA/DNA) chains."""
        return [
            chain for chain in self.chains.values()
            if chain.chain_type in (ChainType.PROTEIN, ChainType.RNA, ChainType.DNA)
        ]

    def get_protein_chains(self) -> List[Chain]:
        """Get all protein chains."""
        return [
            chain for chain in self.chains.values()
            if chain.chain_type == ChainType.PROTEIN
        ]

    def get_nucleic_acid_chains(self) -> List[Chain]:
        """Get all nucleic acid (RNA/DNA) chains."""
        return [
            chain for chain in self.chains.values()
            if chain.chain_type in (ChainType.RNA, ChainType.DNA)
        ]

    def get_ligand_chains(self) -> List[Chain]:
        """Get all ligand chains."""
        return [
            chain for chain in self.chains.values()
            if chain.chain_type == ChainType.LIGAND
        ]

    def remove_hydrogens(self) -> None:
        """Remove all hydrogen atoms from the structure.
        
        From AF3 supplement Section 2.5.4 - hydrogens are removed from bioassemblies.
        """
        for chain in self.chains.values():
            for residue in chain.residues:
                residue.atoms = {
                    name: atom for name, atom in residue.atoms.items()
                    if not atom.is_hydrogen
                }

    def remove_waters(self) -> None:
        """Remove all water molecules.
        
        From AF3 supplement Section 2.1 - waters are removed during parsing.
        """
        for chain_id in list(self.chains.keys()):
            chain = self.chains[chain_id]
            if chain.chain_type == ChainType.WATER:
                del self.chains[chain_id]
            else:
                chain.residues = [
                    res for res in chain.residues if res.name != "HOH"
                ]

    def remove_crystallization_aids(self) -> None:
        """Remove crystallization aids.
        
        From AF3 supplement Section 5.1 and Table 9.
        """
        for chain in self.chains.values():
            chain.residues = [
                res for res in chain.residues
                if res.name not in CRYSTALLIZATION_AIDS
            ]

    def get_interfaces(self, distance_threshold: float = 5.0) -> List[Tuple[str, str]]:
        """Find all chain-chain interfaces.
        
        From AF3 supplement Section 2.5.1:
        Interfaces are pairs of chains with minimum heavy atom separation < 5Å.
        
        Args:
            distance_threshold: Maximum distance for interface (Angstroms)
            
        Returns:
            List of (chain_id1, chain_id2) tuples for chains that form interfaces
        """
        interfaces = []
        chain_ids = list(self.chains.keys())

        for i, chain1_id in enumerate(chain_ids):
            chain1 = self.chains[chain1_id]
            coords1 = chain1.get_all_heavy_atom_coords()
            if len(coords1) == 0:
                continue

            for chain2_id in chain_ids[i + 1:]:
                chain2 = self.chains[chain2_id]
                coords2 = chain2.get_all_heavy_atom_coords()
                if len(coords2) == 0:
                    continue

                # Compute minimum distance
                # For efficiency, we could use scipy.spatial.distance.cdist
                diffs = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
                min_dist = np.min(distances)

                if min_dist < distance_threshold:
                    interfaces.append((chain1_id, chain2_id))

        return interfaces
