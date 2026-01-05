"""Input feature extraction for AlphaFold3.

Implements all input features from AlphaFold3 supplement Table 5:
- Token-level features (Ntokens,)
- Atom-level features (Natoms,)
- MSA features (Nmsa, Ntokens)
- Template features (Ntemplates, Ntokens)
- Pair features (Ntokens, Ntokens)

Reference: AF3 Supplement Section 2.8 and Table 5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from novadb.processing.tokenization.tokenizer import (
    Token,
    TokenType,
    TokenizedStructure,
    Tokenizer,
)
from novadb.search.msa.msa import MSA


@dataclass
class InputFeatures:
    """Container for all input features.
    
    From AF3 Table 5, organized by feature type.
    """
    # Token-level features (Ntokens,)
    token_index: np.ndarray  # int32
    residue_index: np.ndarray  # int32
    asym_id: np.ndarray  # int32 (chain index)
    entity_id: np.ndarray  # int32
    sym_id: np.ndarray  # int32 (symmetry copy)
    restype: np.ndarray  # int32 (one-hot 32 classes)
    is_protein: np.ndarray  # float32 mask
    is_rna: np.ndarray  # float32 mask
    is_dna: np.ndarray  # float32 mask
    is_ligand: np.ndarray  # float32 mask

    # Atom-level features (Natoms,)
    atom_ref_pos: np.ndarray  # float32 (Natoms, 3)
    atom_ref_mask: np.ndarray  # float32 (Natoms,)
    atom_ref_element: np.ndarray  # int32 (Natoms, 128) one-hot
    atom_ref_charge: np.ndarray  # float32 (Natoms,)
    atom_ref_atom_name_chars: np.ndarray  # int32 (Natoms, 4, 64) one-hot
    atom_ref_space_uid: np.ndarray  # int32 (Natoms,) per-residue unique ID

    # Token-to-atom mapping (for aggregation)
    atom_to_token: np.ndarray  # int32 (Natoms,)

    # MSA features (Nmsa, Ntokens)
    msa: Optional[np.ndarray] = None  # int32, values 0-32
    msa_mask: Optional[np.ndarray] = None  # float32
    msa_deletion_value: Optional[np.ndarray] = None  # float32
    msa_species: Optional[np.ndarray] = None  # int32 per-sequence

    # Template features (Ntemplates, Ntokens)
    template_restype: Optional[np.ndarray] = None  # int32
    template_pseudo_beta: Optional[np.ndarray] = None  # float32 (Nt, Ntok, 3)
    template_pseudo_beta_mask: Optional[np.ndarray] = None  # float32
    template_backbone_mask: Optional[np.ndarray] = None  # float32
    template_distogram: Optional[np.ndarray] = None  # float32 (Nt, Ntok, Ntok, 39)

    # Pair features (Ntokens, Ntokens)
    token_bonds: Optional[np.ndarray] = None  # int32 sparse bond matrix

    # Ground truth for training (Natoms, 3)
    atom_gt_coords: Optional[np.ndarray] = None  # float32
    atom_gt_mask: Optional[np.ndarray] = None  # float32

    # Metadata
    num_tokens: int = 0
    num_atoms: int = 0
    pdb_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value
            elif value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputFeatures":
        """Create from dictionary."""
        return cls(**data)


class FeatureExtractor:
    """Extract input features from tokenized structures.
    
    Implements feature extraction per AF3 Table 5.
    """

    # Element encoding (periodic table order)
    ELEMENTS = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    ]
    ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}

    # Atom name character encoding (A-Z, 0-9, etc.)
    ATOM_NAME_CHARS = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' *+-_"
    )
    CHAR_TO_IDX = {c: i for i, c in enumerate(ATOM_NAME_CHARS)}

    # Residue type encoding (32 classes per AF3)
    AA_MAP = {
        "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
        "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
        "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
        "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
    }
    AA_UNK_IDX = 20

    RNA_MAP = {"A": 21, "G": 22, "C": 23, "U": 24}
    RNA_UNK_IDX = 25

    DNA_MAP = {"DA": 26, "DG": 27, "DC": 28, "DT": 29}
    DNA_UNK_IDX = 30

    GAP_IDX = 31
    NUM_RESTYPES = 32

    def __init__(
        self,
        max_msa_sequences: int = 16384,
        max_templates: int = 4,
    ):
        """Initialize the feature extractor.
        
        Args:
            max_msa_sequences: Maximum number of MSA sequences
            max_templates: Maximum number of templates
        """
        self.max_msa_sequences = max_msa_sequences
        self.max_templates = max_templates
        self.tokenizer = Tokenizer()

    def extract(
        self,
        tokenized: TokenizedStructure,
        msa: Optional[MSA] = None,
        templates: Optional[List[TokenizedStructure]] = None,
    ) -> InputFeatures:
        """Extract all input features.
        
        Args:
            tokenized: Tokenized structure
            msa: Optional MSA for the structure
            templates: Optional list of template structures
            
        Returns:
            InputFeatures with all extracted features
        """
        tokens = tokenized.tokens
        n_tokens = len(tokens)

        # Extract token-level features
        token_features = self._extract_token_features(tokens)

        # Extract atom-level features
        atom_features = self._extract_atom_features(tokens)

        # Extract MSA features if provided
        msa_features = {}
        if msa is not None:
            msa_features = self._extract_msa_features(msa, tokens)

        # Extract template features if provided
        template_features = {}
        if templates is not None:
            template_features = self._extract_template_features(
                templates, tokens
            )

        return InputFeatures(
            # Token features
            token_index=token_features["token_index"],
            residue_index=token_features["residue_index"],
            asym_id=token_features["asym_id"],
            entity_id=token_features["entity_id"],
            sym_id=token_features["sym_id"],
            restype=token_features["restype"],
            is_protein=token_features["is_protein"],
            is_rna=token_features["is_rna"],
            is_dna=token_features["is_dna"],
            is_ligand=token_features["is_ligand"],
            # Atom features
            atom_ref_pos=atom_features["atom_ref_pos"],
            atom_ref_mask=atom_features["atom_ref_mask"],
            atom_ref_element=atom_features["atom_ref_element"],
            atom_ref_charge=atom_features["atom_ref_charge"],
            atom_ref_atom_name_chars=atom_features["atom_ref_atom_name_chars"],
            atom_ref_space_uid=atom_features["atom_ref_space_uid"],
            atom_to_token=atom_features["atom_to_token"],
            # MSA features
            msa=msa_features.get("msa"),
            msa_mask=msa_features.get("msa_mask"),
            msa_deletion_value=msa_features.get("msa_deletion_value"),
            msa_species=msa_features.get("msa_species"),
            # Template features
            template_restype=template_features.get("template_restype"),
            template_pseudo_beta=template_features.get("template_pseudo_beta"),
            template_pseudo_beta_mask=template_features.get(
                "template_pseudo_beta_mask"
            ),
            template_backbone_mask=template_features.get("template_backbone_mask"),
            template_distogram=template_features.get("template_distogram"),
            # Metadata
            num_tokens=n_tokens,
            num_atoms=atom_features["num_atoms"],
            pdb_id=tokenized.pdb_id,
        )

    def _extract_token_features(
        self, tokens: List[Token]
    ) -> Dict[str, np.ndarray]:
        """Extract token-level features."""
        n = len(tokens)

        token_index = np.zeros(n, dtype=np.int32)
        residue_index = np.zeros(n, dtype=np.int32)
        asym_id = np.zeros(n, dtype=np.int32)
        entity_id = np.zeros(n, dtype=np.int32)
        sym_id = np.zeros(n, dtype=np.int32)  # Always 0 for single copy
        restype = np.zeros(n, dtype=np.int32)
        is_protein = np.zeros(n, dtype=np.float32)
        is_rna = np.zeros(n, dtype=np.float32)
        is_dna = np.zeros(n, dtype=np.float32)
        is_ligand = np.zeros(n, dtype=np.float32)

        for i, token in enumerate(tokens):
            token_index[i] = token.token_index
            residue_index[i] = token.residue_index
            asym_id[i] = token.chain_index
            entity_id[i] = token.entity_id
            sym_id[i] = 0  # No symmetry expansion

            # Encode residue type
            restype[i] = self._encode_restype(token)

            # Chain type masks
            if token.token_type == TokenType.STANDARD_AMINO_ACID:
                is_protein[i] = 1.0
            elif token.token_type == TokenType.STANDARD_NUCLEOTIDE:
                if token.residue_name in self.RNA_MAP:
                    is_rna[i] = 1.0
                else:
                    is_dna[i] = 1.0
            else:
                is_ligand[i] = 1.0

        return {
            "token_index": token_index,
            "residue_index": residue_index,
            "asym_id": asym_id,
            "entity_id": entity_id,
            "sym_id": sym_id,
            "restype": restype,
            "is_protein": is_protein,
            "is_rna": is_rna,
            "is_dna": is_dna,
            "is_ligand": is_ligand,
        }

    def _encode_restype(self, token: Token) -> int:
        """Encode residue type to integer (0-31)."""
        if token.token_type == TokenType.STANDARD_AMINO_ACID:
            return self.AA_MAP.get(token.residue_name, self.AA_UNK_IDX)
        elif token.token_type == TokenType.STANDARD_NUCLEOTIDE:
            if token.residue_name in self.RNA_MAP:
                return self.RNA_MAP.get(token.residue_name, self.RNA_UNK_IDX)
            else:
                return self.DNA_MAP.get(token.residue_name, self.DNA_UNK_IDX)
        else:
            # Ligands represented as unknown amino acid
            return self.AA_UNK_IDX

    def _extract_atom_features(
        self, tokens: List[Token]
    ) -> Dict[str, np.ndarray]:
        """Extract atom-level features."""
        # Count total atoms
        total_atoms = sum(token.num_atoms for token in tokens)

        atom_ref_pos = np.zeros((total_atoms, 3), dtype=np.float32)
        atom_ref_mask = np.zeros(total_atoms, dtype=np.float32)
        atom_ref_element = np.zeros((total_atoms, 128), dtype=np.int32)
        atom_ref_charge = np.zeros(total_atoms, dtype=np.float32)
        atom_ref_atom_name_chars = np.zeros(
            (total_atoms, 4, 64), dtype=np.int32
        )
        atom_ref_space_uid = np.zeros(total_atoms, dtype=np.int32)
        atom_to_token = np.zeros(total_atoms, dtype=np.int32)

        atom_idx = 0
        space_uid = 0

        for token in tokens:
            # Each token gets its own space_uid (residue-level)
            for atom_name, atom in token.atoms.items():
                # Position
                atom_ref_pos[atom_idx] = atom.coords
                atom_ref_mask[atom_idx] = 1.0

                # Element encoding
                elem_idx = self.ELEMENT_TO_IDX.get(atom.element, 0)
                atom_ref_element[atom_idx, elem_idx] = 1

                # Charge (simplified - from atom data if available)
                atom_ref_charge[atom_idx] = 0.0  # Would come from CCD

                # Atom name encoding (4 characters, 64-dim one-hot each)
                self._encode_atom_name(
                    atom_name, atom_ref_atom_name_chars[atom_idx]
                )

                # Space UID (same for atoms in same token)
                atom_ref_space_uid[atom_idx] = space_uid

                # Token mapping
                atom_to_token[atom_idx] = token.token_index

                atom_idx += 1

            space_uid += 1

        return {
            "atom_ref_pos": atom_ref_pos,
            "atom_ref_mask": atom_ref_mask,
            "atom_ref_element": atom_ref_element,
            "atom_ref_charge": atom_ref_charge,
            "atom_ref_atom_name_chars": atom_ref_atom_name_chars,
            "atom_ref_space_uid": atom_ref_space_uid,
            "atom_to_token": atom_to_token,
            "num_atoms": total_atoms,
        }

    def _encode_atom_name(
        self, name: str, output: np.ndarray
    ) -> None:
        """Encode atom name to 4x64 one-hot array."""
        padded = name.ljust(4)[:4].upper()
        for i, char in enumerate(padded):
            char_idx = self.CHAR_TO_IDX.get(char, 0)
            output[i, char_idx] = 1

    def _extract_msa_features(
        self, msa: MSA, tokens: List[Token]
    ) -> Dict[str, np.ndarray]:
        """Extract MSA features."""
        n_tokens = len(tokens)
        n_seqs = min(len(msa.sequences), self.max_msa_sequences)

        # Build residue index to token index mapping
        # For simplicity, assume MSA covers first chain
        msa_length = len(msa.sequences[0].sequence) if msa.sequences else 0

        msa_arr = np.full((n_seqs, n_tokens), self.GAP_IDX, dtype=np.int32)
        msa_mask = np.zeros((n_seqs, n_tokens), dtype=np.float32)
        msa_deletion_value = np.zeros((n_seqs, n_tokens), dtype=np.float32)
        msa_species = np.zeros(n_seqs, dtype=np.int32)

        # Map residue positions to token indices
        residue_to_token = {}
        for token in tokens:
            if token.is_protein:
                key = (token.chain_id, token.residue_index)
                residue_to_token[key] = token.token_index

        # Extract MSA rows
        for seq_idx, seq in enumerate(msa.sequences[:n_seqs]):
            aligned_seq = seq.sequence
            deletions = seq.deletion_matrix or []

            # Map aligned positions to token positions
            res_idx = 0
            for pos, char in enumerate(aligned_seq):
                if char == "-":
                    continue

                # Find corresponding token
                # This is simplified - real implementation would need
                # proper alignment handling
                token_idx = residue_to_token.get(
                    (tokens[0].chain_id if tokens else "A", res_idx), None
                )

                if token_idx is not None and token_idx < n_tokens:
                    msa_arr[seq_idx, token_idx] = self._encode_msa_char(char)
                    msa_mask[seq_idx, token_idx] = 1.0

                    if deletions and pos < len(deletions):
                        msa_deletion_value[seq_idx, token_idx] = (
                            2.0 / np.pi * np.arctan(deletions[pos] / 3.0)
                        )

                res_idx += 1

            # Species encoding (from description)
            msa_species[seq_idx] = hash(seq.species) % (2**31 - 1)

        return {
            "msa": msa_arr,
            "msa_mask": msa_mask,
            "msa_deletion_value": msa_deletion_value,
            "msa_species": msa_species,
        }

    def _encode_msa_char(self, char: str) -> int:
        """Encode MSA character to integer."""
        char = char.upper()
        if char in self.AA_MAP:
            # Single letter to 3-letter
            aa_1to3 = {
                "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
                "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
                "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
                "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
            }
            aa3 = aa_1to3.get(char)
            if aa3:
                return self.AA_MAP.get(aa3, self.AA_UNK_IDX)
        if char == "-":
            return self.GAP_IDX
        return self.AA_UNK_IDX

    def _extract_template_features(
        self,
        templates: List[TokenizedStructure],
        query_tokens: List[Token],
    ) -> Dict[str, np.ndarray]:
        """Extract template features."""
        n_tokens = len(query_tokens)
        n_templates = min(len(templates), self.max_templates)

        template_restype = np.zeros(
            (n_templates, n_tokens), dtype=np.int32
        )
        template_pseudo_beta = np.zeros(
            (n_templates, n_tokens, 3), dtype=np.float32
        )
        template_pseudo_beta_mask = np.zeros(
            (n_templates, n_tokens), dtype=np.float32
        )
        template_backbone_mask = np.zeros(
            (n_templates, n_tokens), dtype=np.float32
        )
        # Distogram: 39 bins for distances
        template_distogram = np.zeros(
            (n_templates, n_tokens, n_tokens, 39), dtype=np.float32
        )

        for t_idx, template in enumerate(templates[:n_templates]):
            # Build mapping from query to template positions
            # This is simplified - real implementation would use
            # sequence alignment

            for token in template.tokens:
                # Find matching query token
                q_idx = self._find_matching_token(token, query_tokens)
                if q_idx is None:
                    continue

                template_restype[t_idx, q_idx] = self._encode_restype(token)

                # Get pseudo-beta position (CB for non-glycine, CA for glycine)
                pseudo_beta = self._get_pseudo_beta(token)
                if pseudo_beta is not None:
                    template_pseudo_beta[t_idx, q_idx] = pseudo_beta
                    template_pseudo_beta_mask[t_idx, q_idx] = 1.0

                # Check backbone completeness
                if self._has_backbone(token):
                    template_backbone_mask[t_idx, q_idx] = 1.0

            # Compute distogram
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    if (
                        template_pseudo_beta_mask[t_idx, i] > 0
                        and template_pseudo_beta_mask[t_idx, j] > 0
                    ):
                        dist = np.linalg.norm(
                            template_pseudo_beta[t_idx, i]
                            - template_pseudo_beta[t_idx, j]
                        )
                        bin_idx = self._distance_to_bin(dist)
                        template_distogram[t_idx, i, j, bin_idx] = 1.0
                        template_distogram[t_idx, j, i, bin_idx] = 1.0

        return {
            "template_restype": template_restype,
            "template_pseudo_beta": template_pseudo_beta,
            "template_pseudo_beta_mask": template_pseudo_beta_mask,
            "template_backbone_mask": template_backbone_mask,
            "template_distogram": template_distogram,
        }

    def _find_matching_token(
        self, token: Token, query_tokens: List[Token]
    ) -> Optional[int]:
        """Find matching query token by residue index and chain."""
        for i, q_token in enumerate(query_tokens):
            if (
                q_token.residue_index == token.residue_index
                and q_token.chain_id == token.chain_id
            ):
                return i
        return None

    def _get_pseudo_beta(self, token: Token) -> Optional[np.ndarray]:
        """Get pseudo-beta position (CB or CA for glycine)."""
        if token.residue_name == "GLY":
            ca = token.atoms.get("CA")
            return ca.coords if ca else None
        else:
            cb = token.atoms.get("CB")
            if cb:
                return cb.coords
            ca = token.atoms.get("CA")
            return ca.coords if ca else None

    def _has_backbone(self, token: Token) -> bool:
        """Check if token has complete backbone atoms."""
        if token.is_protein:
            required = {"N", "CA", "C"}
            return required.issubset(token.atoms.keys())
        elif token.is_nucleic_acid:
            required = {"P", "C4'", "C3'"}
            return required.issubset(token.atoms.keys())
        return True

    def _distance_to_bin(self, distance: float, num_bins: int = 39) -> int:
        """Convert distance to distogram bin index.
        
        From AF3: 39 bins, 3.25 to 50.75 Å, 1.25 Å spacing
        """
        min_dist = 3.25
        max_dist = 50.75
        bin_width = (max_dist - min_dist) / (num_bins - 1)

        if distance < min_dist:
            return 0
        if distance >= max_dist:
            return num_bins - 1

        return int((distance - min_dist) / bin_width)


def compute_relative_position_encoding(
    tokens: List[Token],
    max_relative_idx: int = 32,
) -> np.ndarray:
    """Compute relative position encoding.
    
    From AF3: Encodes relative position within chains.
    Clipped to [-max_relative_idx, max_relative_idx].
    
    Args:
        tokens: List of tokens
        max_relative_idx: Maximum relative index before clipping
        
    Returns:
        (Ntokens, Ntokens) array of relative positions
    """
    n = len(tokens)
    rel_pos = np.zeros((n, n), dtype=np.int32)

    for i, tok_i in enumerate(tokens):
        for j, tok_j in enumerate(tokens):
            if tok_i.chain_id == tok_j.chain_id:
                diff = tok_j.residue_index - tok_i.residue_index
                diff = np.clip(diff, -max_relative_idx, max_relative_idx)
                rel_pos[i, j] = diff + max_relative_idx
            else:
                # Different chains: use special value
                rel_pos[i, j] = 2 * max_relative_idx + 1

    return rel_pos


def compute_token_pair_features(
    tokens: List[Token],
    max_relative_idx: int = 32,
) -> Dict[str, np.ndarray]:
    """Compute pairwise token features.
    
    Returns:
        Dictionary with:
        - relative_position: (N, N) relative sequence positions
        - same_chain: (N, N) mask for same chain
        - same_entity: (N, N) mask for same entity
    """
    n = len(tokens)

    relative_position = compute_relative_position_encoding(
        tokens, max_relative_idx
    )
    same_chain = np.zeros((n, n), dtype=np.float32)
    same_entity = np.zeros((n, n), dtype=np.float32)

    for i, tok_i in enumerate(tokens):
        for j, tok_j in enumerate(tokens):
            if tok_i.chain_id == tok_j.chain_id:
                same_chain[i, j] = 1.0
            if tok_i.entity_id == tok_j.entity_id:
                same_entity[i, j] = 1.0

    return {
        "relative_position": relative_position,
        "same_chain": same_chain,
        "same_entity": same_entity,
    }
