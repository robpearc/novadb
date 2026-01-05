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
from novadb.processing.optimized import (
    compute_relative_position_vectorized,
    compute_pair_masks_vectorized,
    compute_msa_profile_vectorized,
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
    restype: np.ndarray  # float32 (Ntokens, 32) one-hot encoding
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

    # Atom existence and ambiguity features (Natoms,) - Training targets
    atom_exists: Optional[np.ndarray] = None  # float32, mask for atoms to predict
    atom_is_ambiguous: Optional[np.ndarray] = None  # float32, symmetric/ambiguous atoms

    # Frame atom mask (Natoms,) - Atoms defining local coordinate frames
    frame_atom_mask: Optional[np.ndarray] = None  # float32, atoms used in frame construction

    # Target features for training (Ntokens,)
    target_feat: Optional[np.ndarray] = None  # int32, one-hot target residue type

    # Query structure pseudo-beta positions (Ntokens, 3)
    pseudo_beta: Optional[np.ndarray] = None  # float32, CB (or CA for GLY) positions
    pseudo_beta_mask: Optional[np.ndarray] = None  # float32, validity mask

    # Backbone rigid body representation (Ntokens,)
    backbone_rigid_tensor: Optional[np.ndarray] = None  # float32 (Ntokens, 4, 4)
    backbone_rigid_mask: Optional[np.ndarray] = None  # float32 (Ntokens,)

    # MSA features (Nmsa, Ntokens)
    msa: Optional[np.ndarray] = None  # int32, values 0-32
    msa_mask: Optional[np.ndarray] = None  # float32
    msa_deletion_value: Optional[np.ndarray] = None  # float32
    msa_species: Optional[np.ndarray] = None  # int32 per-sequence

    # MSA profile features (Ntokens,)
    msa_profile: Optional[np.ndarray] = None  # float32 (Ntokens, 32)
    deletion_mean: Optional[np.ndarray] = None  # float32 (Ntokens,)
    has_deletion: Optional[np.ndarray] = None  # float32 (Ntokens,)

    # Template features (Ntemplates, Ntokens)
    template_restype: Optional[np.ndarray] = None  # int32
    template_pseudo_beta: Optional[np.ndarray] = None  # float32 (Nt, Ntok, 3)
    template_pseudo_beta_mask: Optional[np.ndarray] = None  # float32
    template_backbone_mask: Optional[np.ndarray] = None  # float32
    template_distogram: Optional[np.ndarray] = None  # float32 (Nt, Ntok, Ntok, 39)
    template_backbone_frame: Optional[np.ndarray] = None  # float32 (Nt, Ntok, 4, 4)
    template_backbone_frame_mask: Optional[np.ndarray] = None  # float32 (Nt, Ntok)
    template_unit_vector: Optional[np.ndarray] = None  # float32 (Nt, Ntok, Ntok, 3)

    # Pair features (Ntokens, Ntokens)
    token_bonds: Optional[np.ndarray] = None  # int32 sparse bond matrix
    relative_position: Optional[np.ndarray] = None  # int32 (Ntokens, Ntokens)
    same_chain: Optional[np.ndarray] = None  # float32 (Ntokens, Ntokens)
    same_entity: Optional[np.ndarray] = None  # float32 (Ntokens, Ntokens)

    # Ground truth for training (Natoms, 3)
    atom_gt_coords: Optional[np.ndarray] = None  # float32
    atom_gt_mask: Optional[np.ndarray] = None  # float32

    # Metadata features
    resolution: Optional[float] = None  # Experimental resolution in Angstroms
    is_distillation: bool = False  # Whether data comes from distillation

    # Metadata
    num_tokens: int = 0
    num_atoms: int = 0
    pdb_id: str = ""

    @property
    def ref_pos(self) -> Optional[np.ndarray]:
        """Reference positions (alias for atom_ref_pos)."""
        return self.atom_ref_pos

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
        resolution: Optional[float] = None,
        is_distillation: bool = False,
    ) -> InputFeatures:
        """Extract all input features.

        Args:
            tokenized: Tokenized structure
            msa: Optional MSA for the structure
            templates: Optional list of template structures
            resolution: Optional experimental resolution in Angstroms
            is_distillation: Whether data comes from distillation

        Returns:
            InputFeatures with all extracted features
        """
        tokens = tokenized.tokens
        n_tokens = len(tokens)

        # Extract token-level features
        token_features = self._extract_token_features(tokens)

        # Extract atom-level features
        atom_features = self._extract_atom_features(tokens)

        # Extract atom existence and ambiguity features
        atom_exists_features = self._extract_atom_exists_features(tokens)

        # Extract frame atom mask
        frame_features = self._extract_frame_atom_mask(tokens)

        # Extract target features
        target_features = self._extract_target_features(tokens)

        # Extract pseudo-beta features
        pseudo_beta_features = self._extract_pseudo_beta_features(tokens)

        # Extract backbone rigid features
        rigid_features = self._extract_backbone_rigid_features(tokens)

        # Extract pair features
        pair_features = self._extract_pair_features(tokens)

        # Extract MSA features if provided
        msa_features = {}
        msa_profile_features = {}
        if msa is not None:
            msa_features = self._extract_msa_features(msa, tokens)
            # Extract MSA profile features
            if msa_features.get("msa") is not None:
                msa_profile_features = self._extract_msa_profile_features(
                    msa_features["msa"],
                    msa_features.get("msa_deletion_value", np.zeros((1, n_tokens))),
                )

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
            # Atom existence and ambiguity features
            atom_exists=atom_exists_features["atom_exists"],
            atom_is_ambiguous=atom_exists_features["atom_is_ambiguous"],
            # Frame atom mask
            frame_atom_mask=frame_features["frame_atom_mask"],
            # Target features
            target_feat=target_features["target_feat"],
            # Pseudo-beta features
            pseudo_beta=pseudo_beta_features["pseudo_beta"],
            pseudo_beta_mask=pseudo_beta_features["pseudo_beta_mask"],
            # Backbone rigid features
            backbone_rigid_tensor=rigid_features["backbone_rigid_tensor"],
            backbone_rigid_mask=rigid_features["backbone_rigid_mask"],
            # MSA features
            msa=msa_features.get("msa"),
            msa_mask=msa_features.get("msa_mask"),
            msa_deletion_value=msa_features.get("msa_deletion_value"),
            msa_species=msa_features.get("msa_species"),
            # MSA profile features
            msa_profile=msa_profile_features.get("msa_profile"),
            deletion_mean=msa_profile_features.get("deletion_mean"),
            has_deletion=msa_profile_features.get("has_deletion"),
            # Template features
            template_restype=template_features.get("template_restype"),
            template_pseudo_beta=template_features.get("template_pseudo_beta"),
            template_pseudo_beta_mask=template_features.get(
                "template_pseudo_beta_mask"
            ),
            template_backbone_mask=template_features.get("template_backbone_mask"),
            template_distogram=template_features.get("template_distogram"),
            template_backbone_frame=template_features.get("template_backbone_frame"),
            template_backbone_frame_mask=template_features.get(
                "template_backbone_frame_mask"
            ),
            template_unit_vector=template_features.get("template_unit_vector"),
            # Pair features
            token_bonds=pair_features.get("token_bonds"),
            relative_position=pair_features.get("relative_position"),
            same_chain=pair_features.get("same_chain"),
            same_entity=pair_features.get("same_entity"),
            # Metadata features
            resolution=resolution,
            is_distillation=is_distillation,
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
        restype = np.zeros((n, self.NUM_RESTYPES), dtype=np.float32)  # One-hot encoding
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

            # Encode residue type as one-hot
            restype_idx = self._encode_restype(token)
            restype[i, restype_idx] = 1.0

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
        # Backbone frames: 4x4 transformation matrices
        template_backbone_frame = np.zeros(
            (n_templates, n_tokens, 4, 4), dtype=np.float32
        )
        template_backbone_frame_mask = np.zeros(
            (n_templates, n_tokens), dtype=np.float32
        )
        # Unit vectors between residue pairs
        template_unit_vector = np.zeros(
            (n_templates, n_tokens, n_tokens, 3), dtype=np.float32
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

                # Compute backbone frame
                frame = self._compute_backbone_frame(token)
                if frame is not None:
                    template_backbone_frame[t_idx, q_idx] = frame
                    template_backbone_frame_mask[t_idx, q_idx] = 1.0
                else:
                    # Identity matrix as default
                    template_backbone_frame[t_idx, q_idx] = np.eye(4, dtype=np.float32)

            # Compute distogram and unit vectors
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    if (
                        template_pseudo_beta_mask[t_idx, i] > 0
                        and template_pseudo_beta_mask[t_idx, j] > 0
                    ):
                        diff = (
                            template_pseudo_beta[t_idx, j]
                            - template_pseudo_beta[t_idx, i]
                        )
                        dist = np.linalg.norm(diff)

                        # Distogram bin
                        bin_idx = self._distance_to_bin(dist)
                        template_distogram[t_idx, i, j, bin_idx] = 1.0
                        template_distogram[t_idx, j, i, bin_idx] = 1.0

                        # Unit vector (direction from i to j)
                        if dist > 1e-8:
                            unit_vec = diff / dist
                            template_unit_vector[t_idx, i, j] = unit_vec
                            template_unit_vector[t_idx, j, i] = -unit_vec

        return {
            "template_restype": template_restype,
            "template_pseudo_beta": template_pseudo_beta,
            "template_pseudo_beta_mask": template_pseudo_beta_mask,
            "template_backbone_mask": template_backbone_mask,
            "template_distogram": template_distogram,
            "template_backbone_frame": template_backbone_frame,
            "template_backbone_frame_mask": template_backbone_frame_mask,
            "template_unit_vector": template_unit_vector,
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

    def _extract_atom_exists_features(
        self, tokens: List[Token]
    ) -> Dict[str, np.ndarray]:
        """Extract atom existence and ambiguity features.

        From AF3 Table 5:
        - atom_exists: Mask indicating which atoms should be predicted
        - atom_is_ambiguous: Mask for symmetric/ambiguous atom positions

        For training, atom_exists indicates ground truth atom positions.
        For ambiguous atoms (e.g., symmetric sidechains), special handling
        is needed for loss computation.
        """
        total_atoms = sum(token.num_atoms for token in tokens)

        atom_exists = np.zeros(total_atoms, dtype=np.float32)
        atom_is_ambiguous = np.zeros(total_atoms, dtype=np.float32)

        # Ambiguous atom sets for symmetric sidechains
        # From AF3: these atoms can be swapped without changing the structure
        AMBIGUOUS_ATOMS = {
            "ARG": {"NH1", "NH2"},  # Guanidinium
            "ASP": {"OD1", "OD2"},  # Carboxylate
            "GLU": {"OE1", "OE2"},  # Carboxylate
            "LEU": {"CD1", "CD2"},  # Isopropyl
            "PHE": {"CD1", "CD2", "CE1", "CE2"},  # Phenyl ring
            "TYR": {"CD1", "CD2", "CE1", "CE2"},  # Phenyl ring
            "VAL": {"CG1", "CG2"},  # Isopropyl
        }

        atom_idx = 0
        for token in tokens:
            ambiguous_set = AMBIGUOUS_ATOMS.get(token.residue_name, set())

            for atom_name, atom in token.atoms.items():
                # Atom exists if it has valid coordinates
                if not np.any(np.isnan(atom.coords)):
                    atom_exists[atom_idx] = 1.0

                # Mark ambiguous atoms
                if atom_name in ambiguous_set:
                    atom_is_ambiguous[atom_idx] = 1.0

                atom_idx += 1

        return {
            "atom_exists": atom_exists,
            "atom_is_ambiguous": atom_is_ambiguous,
        }

    def _extract_frame_atom_mask(
        self, tokens: List[Token]
    ) -> Dict[str, np.ndarray]:
        """Extract frame atom mask.

        From AF3: Indicates which atoms are used to construct
        local coordinate frames for each residue.

        For proteins: N, CA, C atoms define the backbone frame
        For nucleotides: C1', C3', C4' atoms define the frame
        For ligands: first 3 atoms or centroid-based frame
        """
        total_atoms = sum(token.num_atoms for token in tokens)
        frame_atom_mask = np.zeros(total_atoms, dtype=np.float32)

        # Frame atoms for each molecule type
        PROTEIN_FRAME_ATOMS = {"N", "CA", "C"}
        NUCLEIC_FRAME_ATOMS = {"C1'", "C3'", "C4'"}

        atom_idx = 0
        for token in tokens:
            if token.token_type == TokenType.STANDARD_AMINO_ACID:
                frame_atoms = PROTEIN_FRAME_ATOMS
            elif token.token_type == TokenType.STANDARD_NUCLEOTIDE:
                frame_atoms = NUCLEIC_FRAME_ATOMS
            else:
                # For ligands, use first 3 atoms as frame
                frame_atoms = set(list(token.atoms.keys())[:3])

            for atom_name in token.atoms.keys():
                if atom_name in frame_atoms:
                    frame_atom_mask[atom_idx] = 1.0
                atom_idx += 1

        return {"frame_atom_mask": frame_atom_mask}

    def _extract_target_features(
        self, tokens: List[Token]
    ) -> Dict[str, np.ndarray]:
        """Extract target features for training.

        From AF3 Table 5:
        - target_feat: One-hot encoding of target residue type

        This is used during training to provide the ground truth
        residue identity for structure prediction.
        """
        n_tokens = len(tokens)
        target_feat = np.zeros((n_tokens, self.NUM_RESTYPES), dtype=np.float32)

        for i, token in enumerate(tokens):
            restype_idx = self._encode_restype(token)
            target_feat[i, restype_idx] = 1.0

        return {"target_feat": target_feat}

    def _extract_pseudo_beta_features(
        self, tokens: List[Token]
    ) -> Dict[str, np.ndarray]:
        """Extract pseudo-beta features for query structure.

        From AF3 Table 5:
        - pseudo_beta: CB position (CA for GLY) for each token
        - pseudo_beta_mask: Validity mask for pseudo-beta positions

        These are used for distance-based features and contact prediction.
        """
        n_tokens = len(tokens)
        pseudo_beta = np.zeros((n_tokens, 3), dtype=np.float32)
        pseudo_beta_mask = np.zeros(n_tokens, dtype=np.float32)

        for i, token in enumerate(tokens):
            pb_pos = self._get_pseudo_beta(token)
            if pb_pos is not None:
                pseudo_beta[i] = pb_pos
                pseudo_beta_mask[i] = 1.0

        return {
            "pseudo_beta": pseudo_beta,
            "pseudo_beta_mask": pseudo_beta_mask,
        }

    def _extract_backbone_rigid_features(
        self, tokens: List[Token]
    ) -> Dict[str, np.ndarray]:
        """Extract backbone rigid body representation.

        From AF3 Section 4.3.2:
        - backbone_rigid_tensor: 4x4 transformation matrices for backbone frames
        - backbone_rigid_mask: Validity mask for frames

        The rigid body representation defines local coordinate systems
        at each residue for the diffusion-based structure module.
        """
        n_tokens = len(tokens)
        backbone_rigid_tensor = np.zeros((n_tokens, 4, 4), dtype=np.float32)
        backbone_rigid_mask = np.zeros(n_tokens, dtype=np.float32)

        for i, token in enumerate(tokens):
            frame = self._compute_backbone_frame(token)
            if frame is not None:
                backbone_rigid_tensor[i] = frame
                backbone_rigid_mask[i] = 1.0
            else:
                # Identity matrix as default
                backbone_rigid_tensor[i] = np.eye(4, dtype=np.float32)

        return {
            "backbone_rigid_tensor": backbone_rigid_tensor,
            "backbone_rigid_mask": backbone_rigid_mask,
        }

    def _compute_backbone_frame(self, token: Token) -> Optional[np.ndarray]:
        """Compute 4x4 backbone frame transformation matrix.

        For proteins: Frame defined by N, CA, C atoms
        For nucleotides: Frame defined by C1', C3', C4' atoms
        For ligands: Frame defined by first 3 atoms

        Returns:
            4x4 transformation matrix or None if cannot be computed
        """
        if token.token_type == TokenType.STANDARD_AMINO_ACID:
            n_atom = token.atoms.get("N")
            ca_atom = token.atoms.get("CA")
            c_atom = token.atoms.get("C")

            if n_atom is None or ca_atom is None or c_atom is None:
                return None

            n_pos = n_atom.coords
            ca_pos = ca_atom.coords
            c_pos = c_atom.coords

        elif token.token_type == TokenType.STANDARD_NUCLEOTIDE:
            c1_atom = token.atoms.get("C1'")
            c3_atom = token.atoms.get("C3'")
            c4_atom = token.atoms.get("C4'")

            if c1_atom is None or c3_atom is None or c4_atom is None:
                return None

            # Use C4' as origin, C1' and C3' to define axes
            n_pos = c3_atom.coords  # Analogous to N
            ca_pos = c4_atom.coords  # Analogous to CA (origin)
            c_pos = c1_atom.coords  # Analogous to C

        else:
            # For ligands, use first 3 atoms if available
            atom_list = list(token.atoms.values())
            if len(atom_list) < 3:
                return None

            n_pos = atom_list[0].coords
            ca_pos = atom_list[1].coords
            c_pos = atom_list[2].coords

        # Build local coordinate frame at CA/C4'/atom2
        # X-axis: CA -> C direction
        x_axis = c_pos - ca_pos
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            return None
        x_axis = x_axis / x_norm

        # Y-axis: perpendicular in N-CA-C plane
        n_to_ca = ca_pos - n_pos
        y_axis = n_to_ca - np.dot(n_to_ca, x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            return None
        y_axis = y_axis / y_norm

        # Z-axis: cross product (right-handed)
        z_axis = np.cross(x_axis, y_axis)

        # Build 4x4 transformation matrix
        frame = np.eye(4, dtype=np.float32)
        frame[:3, 0] = x_axis
        frame[:3, 1] = y_axis
        frame[:3, 2] = z_axis
        frame[:3, 3] = ca_pos  # Translation (origin at CA)

        return frame

    def _extract_pair_features(
        self, tokens: List[Token]
    ) -> Dict[str, np.ndarray]:
        """Extract pairwise token features.

        From AF3 Table 5:
        - relative_position: Relative sequence positions (clipped)
        - same_chain: Mask for tokens in same chain
        - same_entity: Mask for tokens in same entity
        """
        return compute_token_pair_features(tokens)

    def _extract_msa_profile_features(
        self, msa_arr: np.ndarray, msa_deletion_value: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Extract MSA profile features.

        From AF3 Table 5:
        - msa_profile: Distribution over residue types (Ntokens, 32)
        - deletion_mean: Mean deletion count per position
        - has_deletion: Binary deletion indicator

        Uses vectorized profile computation for O(n_tokens) instead of
        O(n_tokens * NUM_RESTYPES * n_seqs).
        """
        # Use vectorized MSA profile computation
        msa_profile = compute_msa_profile_vectorized(
            msa_arr, num_classes=self.NUM_RESTYPES
        )

        # Compute deletion features (already vectorized)
        deletion_mean = msa_deletion_value.mean(axis=0).astype(np.float32)
        has_deletion = (msa_deletion_value > 0).any(axis=0).astype(np.float32)

        return {
            "msa_profile": msa_profile,
            "deletion_mean": deletion_mean,
            "has_deletion": has_deletion,
        }


def compute_relative_position_encoding(
    tokens: List[Token],
    max_relative_idx: int = 32,
) -> np.ndarray:
    """Compute relative position encoding.

    From AF3: Encodes relative position within chains.
    Clipped to [-max_relative_idx, max_relative_idx].

    Uses vectorized implementation for O(n) instead of O(n²) Python loops.

    Args:
        tokens: List of tokens
        max_relative_idx: Maximum relative index before clipping

    Returns:
        (Ntokens, Ntokens) array of relative positions
    """
    if not tokens:
        return np.zeros((0, 0), dtype=np.int32)

    n = len(tokens)

    # Extract residue indices and chain IDs as arrays
    residue_indices = np.array([tok.residue_index for tok in tokens], dtype=np.int32)

    # Convert chain_ids to integer indices for vectorized comparison
    chain_id_map = {}
    chain_idx = 0
    chain_ids = np.zeros(n, dtype=np.int32)
    for i, tok in enumerate(tokens):
        if tok.chain_id not in chain_id_map:
            chain_id_map[tok.chain_id] = chain_idx
            chain_idx += 1
        chain_ids[i] = chain_id_map[tok.chain_id]

    # Use vectorized computation
    return compute_relative_position_vectorized(
        residue_indices, chain_ids, max_relative_idx
    )


def compute_token_pair_features(
    tokens: List[Token],
    max_relative_idx: int = 32,
) -> Dict[str, np.ndarray]:
    """Compute pairwise token features.

    Uses vectorized implementations for O(n) array construction instead of
    O(n²) nested Python loops.

    Returns:
        Dictionary with:
        - relative_position: (N, N) relative sequence positions
        - same_chain: (N, N) mask for same chain
        - same_entity: (N, N) mask for same entity
    """
    if not tokens:
        return {
            "relative_position": np.zeros((0, 0), dtype=np.int32),
            "same_chain": np.zeros((0, 0), dtype=np.float32),
            "same_entity": np.zeros((0, 0), dtype=np.float32),
        }

    n = len(tokens)

    # Extract chain and entity IDs as integer arrays for vectorized comparison
    chain_id_map = {}
    entity_id_map = {}
    chain_idx = 0
    entity_idx = 0
    chain_ids = np.zeros(n, dtype=np.int32)
    entity_ids = np.zeros(n, dtype=np.int32)
    residue_indices = np.zeros(n, dtype=np.int32)

    for i, tok in enumerate(tokens):
        # Map chain IDs to integers
        if tok.chain_id not in chain_id_map:
            chain_id_map[tok.chain_id] = chain_idx
            chain_idx += 1
        chain_ids[i] = chain_id_map[tok.chain_id]

        # Map entity IDs to integers
        if tok.entity_id not in entity_id_map:
            entity_id_map[tok.entity_id] = entity_idx
            entity_idx += 1
        entity_ids[i] = entity_id_map[tok.entity_id]

        residue_indices[i] = tok.residue_index

    # Use vectorized computations
    relative_position = compute_relative_position_vectorized(
        residue_indices, chain_ids, max_relative_idx
    )
    same_chain, same_entity = compute_pair_masks_vectorized(chain_ids, entity_ids)

    return {
        "relative_position": relative_position,
        "same_chain": same_chain,
        "same_entity": same_entity,
    }
