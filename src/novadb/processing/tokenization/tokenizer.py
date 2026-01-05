"""Tokenization for biomolecular structures.

This module implements the tokenization scheme from AlphaFold3 
supplement Section 2.6:
- Standard amino acid residue → 1 token
- Standard nucleotide residue → 1 token  
- Modified amino acid/nucleotide → 1 token per heavy atom
- Ligands → 1 token per heavy atom

Token center atoms:
- Cα for standard amino acids
- C1' for standard nucleotides
- Single heavy atom for per-atom tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np

from novadb.data.parsers.structure import (
    Structure,
    Chain,
    Residue,
    Atom,
    ChainType,
    STANDARD_AMINO_ACIDS,
    STANDARD_RNA_NUCLEOTIDES,
    STANDARD_DNA_NUCLEOTIDES,
)


class TokenType(Enum):
    """Type of token in the tokenization scheme."""
    STANDARD_AMINO_ACID = auto()
    STANDARD_NUCLEOTIDE = auto()
    MODIFIED_RESIDUE = auto()
    LIGAND_ATOM = auto()
    ION = auto()


@dataclass
class Token:
    """Represents a single token in the tokenized structure.
    
    From AF3 Section 2.6, a token is either:
    - A complete standard residue (amino acid or nucleotide)
    - A single heavy atom (for modified residues or ligands)
    
    Attributes:
        token_index: Global token index
        residue_index: Residue position in original chain
        chain_id: Parent chain identifier
        chain_index: Index of chain in structure (asym_id)
        entity_id: Entity identifier (same sequence = same entity)
        token_type: Type of token
        residue_name: 3-letter residue/compound code
        atoms: Dict of atom name to Atom for this token
        center_atom_name: Name of the center atom for this token
    """
    token_index: int
    residue_index: int
    chain_id: str
    chain_index: int
    entity_id: int
    token_type: TokenType
    residue_name: str
    atoms: Dict[str, Atom] = field(default_factory=dict)
    center_atom_name: str = ""

    @property
    def num_atoms(self) -> int:
        """Number of atoms in this token."""
        return len(self.atoms)

    @property
    def center_atom(self) -> Optional[Atom]:
        """Get the center atom for this token."""
        return self.atoms.get(self.center_atom_name)

    @property
    def center_coords(self) -> Optional[np.ndarray]:
        """Get coordinates of the center atom."""
        center = self.center_atom
        if center:
            return center.coords
        return None

    @property
    def is_standard_residue(self) -> bool:
        """Check if this is a standard (non-atom-level) token."""
        return self.token_type in (
            TokenType.STANDARD_AMINO_ACID,
            TokenType.STANDARD_NUCLEOTIDE,
        )

    @property
    def is_protein(self) -> bool:
        """Check if this is a protein token."""
        return self.token_type == TokenType.STANDARD_AMINO_ACID

    @property
    def is_nucleic_acid(self) -> bool:
        """Check if this is a nucleic acid token."""
        return self.token_type == TokenType.STANDARD_NUCLEOTIDE

    def get_atom_coords(self) -> np.ndarray:
        """Get coordinates of all atoms as Nx3 array."""
        if not self.atoms:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([atom.coords for atom in self.atoms.values()])

    def get_atom_elements(self) -> List[str]:
        """Get element symbols for all atoms."""
        return [atom.element for atom in self.atoms.values()]


@dataclass
class TokenizedStructure:
    """Represents a tokenized structure.
    
    Attributes:
        tokens: List of all tokens
        pdb_id: Structure identifier
        chain_id_to_index: Mapping from chain ID to chain index
        entity_id_map: Mapping from chain ID to entity ID
    """
    tokens: List[Token]
    pdb_id: str
    chain_id_to_index: Dict[str, int] = field(default_factory=dict)
    entity_id_map: Dict[str, int] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        """Total number of tokens."""
        return len(self.tokens)

    @property
    def num_atoms(self) -> int:
        """Total number of atoms across all tokens."""
        return sum(token.num_atoms for token in self.tokens)

    def get_chain_tokens(self, chain_id: str) -> List[Token]:
        """Get all tokens for a specific chain."""
        return [t for t in self.tokens if t.chain_id == chain_id]

    def get_token_indices(self) -> np.ndarray:
        """Get token indices as array."""
        return np.array([t.token_index for t in self.tokens], dtype=np.int32)

    def get_residue_indices(self) -> np.ndarray:
        """Get residue indices as array."""
        return np.array([t.residue_index for t in self.tokens], dtype=np.int32)

    def get_chain_indices(self) -> np.ndarray:
        """Get chain indices (asym_id) as array."""
        return np.array([t.chain_index for t in self.tokens], dtype=np.int32)

    def get_entity_ids(self) -> np.ndarray:
        """Get entity IDs as array."""
        return np.array([t.entity_id for t in self.tokens], dtype=np.int32)

    def get_token_center_coords(self) -> np.ndarray:
        """Get center atom coordinates for all tokens as Nx3 array."""
        coords = []
        for token in self.tokens:
            center = token.center_coords
            if center is not None:
                coords.append(center)
            else:
                coords.append(np.zeros(3, dtype=np.float32))
        return np.stack(coords)

    def get_flat_atom_coords(self) -> np.ndarray:
        """Get all atom coordinates as flat Mx3 array."""
        coords = []
        for token in self.tokens:
            for atom in token.atoms.values():
                coords.append(atom.coords)
        if coords:
            return np.stack(coords)
        return np.zeros((0, 3), dtype=np.float32)

    def get_token_to_atom_mapping(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get mapping from flat atom indices to token indices.
        
        Returns:
            Tuple of (atom_to_token, token_atom_idx)
            - atom_to_token: Token index for each atom
            - token_atom_idx: Within-token atom index for each atom
        """
        atom_to_token = []
        token_atom_idx = []

        for token in self.tokens:
            for i, _ in enumerate(token.atoms.values()):
                atom_to_token.append(token.token_index)
                token_atom_idx.append(i)

        return (
            np.array(atom_to_token, dtype=np.int32),
            np.array(token_atom_idx, dtype=np.int32),
        )


class Tokenizer:
    """Tokenizes structures according to AF3 tokenization scheme.
    
    From AF3 supplement Section 2.6:
    - Standard amino acid residue → 1 token
    - Standard nucleotide residue → 1 token
    - Modified amino acid/nucleotide → 1 token per heavy atom
    - All ligands → 1 token per heavy atom
    
    Token center atoms:
    - Cα for standard amino acids
    - C1' for standard nucleotides
    - The single heavy atom for per-atom tokens
    """

    # Standard residue mappings
    STANDARD_AA = STANDARD_AMINO_ACIDS - {"UNK"}
    STANDARD_RNA = {"A", "G", "C", "U"}
    STANDARD_DNA = {"DA", "DG", "DC", "DT"}

    def __init__(
        self,
        max_atoms_per_token: int = 23,  # Largest standard residue (Trp)
    ):
        """Initialize the tokenizer.
        
        Args:
            max_atoms_per_token: Maximum atoms per standard residue token
        """
        self.max_atoms_per_token = max_atoms_per_token

    def tokenize(self, structure: Structure) -> TokenizedStructure:
        """Tokenize a structure.
        
        Args:
            structure: Structure to tokenize
            
        Returns:
            TokenizedStructure with all tokens
        """
        tokens = []
        token_index = 0
        chain_id_to_index = {}
        entity_id_map = {}

        # Assign chain indices and entity IDs
        for chain_index, chain_id in enumerate(sorted(structure.chains.keys())):
            chain = structure.chains[chain_id]
            chain_id_to_index[chain_id] = chain_index
            entity_id_map[chain_id] = chain.entity_id

        # Process each chain
        for chain_id in sorted(structure.chains.keys()):
            chain = structure.chains[chain_id]
            chain_index = chain_id_to_index[chain_id]
            entity_id = entity_id_map[chain_id]

            for residue in chain.residues:
                new_tokens = self._tokenize_residue(
                    residue=residue,
                    chain_id=chain_id,
                    chain_index=chain_index,
                    entity_id=entity_id,
                    start_token_index=token_index,
                    chain_type=chain.chain_type,
                )

                tokens.extend(new_tokens)
                token_index += len(new_tokens)

        return TokenizedStructure(
            tokens=tokens,
            pdb_id=structure.pdb_id,
            chain_id_to_index=chain_id_to_index,
            entity_id_map=entity_id_map,
        )

    def _tokenize_residue(
        self,
        residue: Residue,
        chain_id: str,
        chain_index: int,
        entity_id: int,
        start_token_index: int,
        chain_type: Optional[ChainType],
    ) -> List[Token]:
        """Tokenize a single residue.
        
        Returns either:
        - Single token for standard residues
        - Multiple tokens (one per heavy atom) for modified residues/ligands
        """
        # Determine if this is a standard residue
        if self._is_standard_amino_acid(residue):
            return [self._create_standard_aa_token(
                residue, chain_id, chain_index, entity_id, start_token_index
            )]

        if self._is_standard_nucleotide(residue):
            return [self._create_standard_nucleotide_token(
                residue, chain_id, chain_index, entity_id, start_token_index
            )]

        # Non-standard: tokenize per heavy atom
        return self._create_per_atom_tokens(
            residue, chain_id, chain_index, entity_id, start_token_index, chain_type
        )

    def _is_standard_amino_acid(self, residue: Residue) -> bool:
        """Check if residue is a standard amino acid."""
        return residue.name in self.STANDARD_AA

    def _is_standard_nucleotide(self, residue: Residue) -> bool:
        """Check if residue is a standard nucleotide."""
        return residue.name in self.STANDARD_RNA or residue.name in self.STANDARD_DNA

    def _create_standard_aa_token(
        self,
        residue: Residue,
        chain_id: str,
        chain_index: int,
        entity_id: int,
        token_index: int,
    ) -> Token:
        """Create token for standard amino acid."""
        return Token(
            token_index=token_index,
            residue_index=residue.seq_id,
            chain_id=chain_id,
            chain_index=chain_index,
            entity_id=entity_id,
            token_type=TokenType.STANDARD_AMINO_ACID,
            residue_name=residue.name,
            atoms=residue.atoms.copy(),
            center_atom_name="CA",
        )

    def _create_standard_nucleotide_token(
        self,
        residue: Residue,
        chain_id: str,
        chain_index: int,
        entity_id: int,
        token_index: int,
    ) -> Token:
        """Create token for standard nucleotide."""
        return Token(
            token_index=token_index,
            residue_index=residue.seq_id,
            chain_id=chain_id,
            chain_index=chain_index,
            entity_id=entity_id,
            token_type=TokenType.STANDARD_NUCLEOTIDE,
            residue_name=residue.name,
            atoms=residue.atoms.copy(),
            center_atom_name="C1'",
        )

    def _create_per_atom_tokens(
        self,
        residue: Residue,
        chain_id: str,
        chain_index: int,
        entity_id: int,
        start_token_index: int,
        chain_type: Optional[ChainType],
    ) -> List[Token]:
        """Create per-atom tokens for modified residue or ligand."""
        tokens = []

        # Determine token type
        if chain_type == ChainType.ION:
            token_type = TokenType.ION
        elif chain_type in (ChainType.PROTEIN, ChainType.RNA, ChainType.DNA):
            token_type = TokenType.MODIFIED_RESIDUE
        else:
            token_type = TokenType.LIGAND_ATOM

        # Create one token per heavy atom
        for i, (atom_name, atom) in enumerate(residue.atoms.items()):
            if atom.is_hydrogen:
                continue

            token = Token(
                token_index=start_token_index + len(tokens),
                residue_index=residue.seq_id,
                chain_id=chain_id,
                chain_index=chain_index,
                entity_id=entity_id,
                token_type=token_type,
                residue_name=residue.name,
                atoms={atom_name: atom},
                center_atom_name=atom_name,
            )
            tokens.append(token)

        return tokens

    def get_restype_encoding(self, tokens: List[Token]) -> np.ndarray:
        """Get one-hot restype encoding for tokens.
        
        From AF3 Table 5: 32 possible values
        - 20 amino acids + unknown (0-20)
        - 4 RNA nucleotides + unknown (21-25)
        - 4 DNA nucleotides + unknown (26-30)
        - Gap (31)
        """
        num_classes = 32

        # Mapping for amino acids
        aa_map = {
            "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
            "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
            "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
            "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
        }

        # Mapping for nucleotides
        rna_map = {"A": 21, "G": 22, "C": 23, "U": 24}
        dna_map = {"DA": 26, "DG": 27, "DC": 28, "DT": 29}

        encoding = np.zeros((len(tokens), num_classes), dtype=np.float32)

        for i, token in enumerate(tokens):
            if token.token_type == TokenType.STANDARD_AMINO_ACID:
                idx = aa_map.get(token.residue_name, 20)  # 20 = unknown AA
            elif token.token_type == TokenType.STANDARD_NUCLEOTIDE:
                idx = rna_map.get(token.residue_name, 
                                  dna_map.get(token.residue_name, 25))
            else:
                # Ligands represented as "unknown amino acid"
                idx = 20

            encoding[i, idx] = 1.0

        return encoding

    def get_chain_type_masks(self, tokens: List[Token]) -> Dict[str, np.ndarray]:
        """Get per-token chain type masks.
        
        From AF3 Table 5: is_protein, is_rna, is_dna, is_ligand
        """
        n = len(tokens)
        masks = {
            "is_protein": np.zeros(n, dtype=np.float32),
            "is_rna": np.zeros(n, dtype=np.float32),
            "is_dna": np.zeros(n, dtype=np.float32),
            "is_ligand": np.zeros(n, dtype=np.float32),
        }

        for i, token in enumerate(tokens):
            if token.token_type == TokenType.STANDARD_AMINO_ACID:
                masks["is_protein"][i] = 1.0
            elif token.token_type == TokenType.STANDARD_NUCLEOTIDE:
                if token.residue_name in self.STANDARD_RNA:
                    masks["is_rna"][i] = 1.0
                else:
                    masks["is_dna"][i] = 1.0
            elif token.token_type == TokenType.MODIFIED_RESIDUE:
                # Determine based on residue type (simplified)
                masks["is_ligand"][i] = 1.0
            else:
                masks["is_ligand"][i] = 1.0

        return masks
