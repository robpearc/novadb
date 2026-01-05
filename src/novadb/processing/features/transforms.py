"""Composable feature transforms for AlphaFold3.

This module provides a composable transform architecture for feature extraction.
Transforms can be composed into pipelines and selectively enabled/disabled.

Transform Hierarchy (from AF3 Table 5):
- Per-Token: Single token features (residue type, token type, position)
- Per-Residue: Residue-level features (atoms, pseudo-beta, backbone)
- Per-Chain: Chain-level aggregations (chain tokens, chain atoms)
- Per-Sequence: Sequence features (MSA encoding)
- Pair: Pairwise features (relative position, distogram)
- Per-Structure: Full structure features (all tokens, all atoms)

Usage:
    # Create individual transforms
    transforms = [
        ResidueTypeEncodeTransform(),
        TokenTypeEncodeTransform(),
        PseudoBetaTransform(),
        RelativePositionTransform(),
    ]

    # Compose into pipeline
    pipeline = Pipeline(transforms)

    # Apply to data
    features = pipeline(input_data)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np

from novadb.processing.optimized import (
    compute_distogram_fast,
    compute_relative_position_vectorized,
    compute_pair_masks_vectorized,
    compute_unit_vectors_vectorized,
    compute_msa_profile_vectorized,
    compute_pairwise_distances_fast,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

# Feature dictionary type - the common data structure for transforms
FeatureDict = Dict[str, Any]

# Type variable for transform input/output
T = TypeVar("T", bound=FeatureDict)


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass(frozen=True)
class FeatureConfig:
    """Base configuration for feature transforms.

    Attributes:
        num_residue_types: Number of residue type classes (32 for AF3).
        num_element_types: Number of element types (128 for periodic table).
        num_atom_name_chars: Number of atom name character classes.
        atom_name_length: Length of atom name strings.
        max_relative_position: Maximum relative position encoding.
    """

    num_residue_types: int = 32
    num_element_types: int = 128
    num_atom_name_chars: int = 64
    atom_name_length: int = 4
    max_relative_position: int = 32


@dataclass(frozen=True)
class DistogramConfig:
    """Configuration for distogram computation.

    From AF3: 39 bins from 3.25 to 50.75 Angstroms.

    Attributes:
        num_bins: Number of distance bins.
        min_distance: Minimum distance (Angstroms).
        max_distance: Maximum distance (Angstroms).
    """

    num_bins: int = 39
    min_distance: float = 3.25
    max_distance: float = 50.75


@dataclass(frozen=True)
class MSAConfig:
    """Configuration for MSA feature extraction.

    Attributes:
        max_sequences: Maximum number of MSA sequences.
        num_classes: Number of residue classes in MSA.
        pseudocount: Pseudocount for profile normalization.
    """

    max_sequences: int = 16384
    num_classes: int = 32
    pseudocount: float = 1e-8


@dataclass(frozen=True)
class TemplateConfig:
    """Configuration for template feature extraction.

    Attributes:
        max_templates: Maximum number of templates.
        distogram_config: Distogram configuration.
    """

    max_templates: int = 4
    distogram_config: DistogramConfig = field(default_factory=DistogramConfig)


# =============================================================================
# Base Transform Protocol and Classes
# =============================================================================


@runtime_checkable
class Transform(Protocol):
    """Protocol for feature transforms.

    Transforms take a feature dictionary and return an updated dictionary.
    They should be stateless and composable.
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        """Apply the transform to input data.

        Args:
            data: Input feature dictionary.

        Returns:
            Updated feature dictionary with new features added.
        """
        ...


class BaseTransform(ABC):
    """Abstract base class for transforms with common functionality."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize transform with optional configuration.

        Args:
            config: Feature configuration. Uses defaults if None.
        """
        self.config = config or FeatureConfig()

    @abstractmethod
    def __call__(self, data: FeatureDict) -> FeatureDict:
        """Apply the transform."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Pipeline:
    """Compose multiple transforms into a single callable.

    Transforms are applied in order, with each transform receiving
    the output of the previous one.

    Example:
        pipeline = Pipeline([
            ResidueTypeEncodeTransform(),
            TokenTypeEncodeTransform(),
            RelativePositionTransform(),
        ])
        features = pipeline(input_data)
    """

    def __init__(
        self,
        transforms: Optional[Sequence[Transform]] = None,
        *,
        name: str = "FeaturePipeline",
    ):
        """Initialize pipeline with transforms.

        Args:
            transforms: Sequence of transforms to apply.
            name: Name for logging and debugging.
        """
        self.transforms: List[Transform] = list(transforms) if transforms else []
        self.name = name

    def __call__(self, data: FeatureDict) -> FeatureDict:
        """Apply all transforms in sequence.

        Args:
            data: Input feature dictionary.

        Returns:
            Feature dictionary after all transforms applied.
        """
        result = data.copy()
        for transform in self.transforms:
            result = transform(result)
        return result

    def add(self, transform: Transform) -> "Pipeline":
        """Add a transform to the pipeline.

        Args:
            transform: Transform to add.

        Returns:
            Self for chaining.
        """
        self.transforms.append(transform)
        return self

    def __repr__(self) -> str:
        transform_names = [t.__class__.__name__ for t in self.transforms]
        return f"Pipeline({self.name}, transforms={transform_names})"

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self):
        return iter(self.transforms)


# =============================================================================
# Residue Type Encoding Constants
# =============================================================================

# Standard amino acids (20)
AA_MAP = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
AA_UNK_IDX = 20

# RNA nucleotides
RNA_MAP = {"A": 21, "G": 22, "C": 23, "U": 24}
RNA_UNK_IDX = 25

# DNA nucleotides
DNA_MAP = {"DA": 26, "DG": 27, "DC": 28, "DT": 29}
DNA_UNK_IDX = 30

# Gap token
GAP_IDX = 31

# 1-letter to 3-letter amino acid mapping
AA_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

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

# Atom name character encoding
ATOM_NAME_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' *+-_"
CHAR_TO_IDX = {c: i for i, c in enumerate(ATOM_NAME_CHARS)}


# =============================================================================
# Per-Token Transforms
# =============================================================================


class ResidueTypeEncodeTransform(BaseTransform):
    """Encode residue types to integer indices.

    From AF3 Table 5: restype is a 32-class encoding.
    - 0-19: Standard amino acids
    - 20: Unknown amino acid
    - 21-24: RNA nucleotides (A, G, C, U)
    - 25: Unknown RNA
    - 26-29: DNA nucleotides (DA, DG, DC, DT)
    - 30: Unknown DNA
    - 31: Gap

    Input keys:
        - residue_names: List[str] of residue names
        - token_types: List[str] of token types ("protein", "rna", "dna", "ligand")

    Output keys:
        - restype: np.ndarray (Ntokens,) of int32
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        residue_names = data.get("residue_names", [])
        token_types = data.get("token_types", [])

        n = len(residue_names)
        restype = np.zeros(n, dtype=np.int32)

        for i, (name, ttype) in enumerate(zip(residue_names, token_types)):
            restype[i] = self._encode_restype(name, ttype)

        data["restype"] = restype
        return data

    def _encode_restype(self, name: str, token_type: str) -> int:
        """Encode a single residue type."""
        if token_type == "protein":
            return AA_MAP.get(name, AA_UNK_IDX)
        elif token_type == "rna":
            return RNA_MAP.get(name, RNA_UNK_IDX)
        elif token_type == "dna":
            return DNA_MAP.get(name, DNA_UNK_IDX)
        else:
            # Ligands use unknown amino acid encoding
            return AA_UNK_IDX


class TokenTypeEncodeTransform(BaseTransform):
    """Encode token types to boolean masks.

    From AF3 Table 5: is_protein, is_rna, is_dna, is_ligand masks.

    Input keys:
        - token_types: List[str] of token types

    Output keys:
        - is_protein: np.ndarray (Ntokens,) float32 mask
        - is_rna: np.ndarray (Ntokens,) float32 mask
        - is_dna: np.ndarray (Ntokens,) float32 mask
        - is_ligand: np.ndarray (Ntokens,) float32 mask
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        token_types = data.get("token_types", [])
        n = len(token_types)

        is_protein = np.zeros(n, dtype=np.float32)
        is_rna = np.zeros(n, dtype=np.float32)
        is_dna = np.zeros(n, dtype=np.float32)
        is_ligand = np.zeros(n, dtype=np.float32)

        for i, ttype in enumerate(token_types):
            if ttype == "protein":
                is_protein[i] = 1.0
            elif ttype == "rna":
                is_rna[i] = 1.0
            elif ttype == "dna":
                is_dna[i] = 1.0
            else:
                is_ligand[i] = 1.0

        data["is_protein"] = is_protein
        data["is_rna"] = is_rna
        data["is_dna"] = is_dna
        data["is_ligand"] = is_ligand
        return data


class TokenPositionTransform(BaseTransform):
    """Encode token position features.

    From AF3 Table 5: token_index, residue_index, asym_id, entity_id, sym_id.

    Input keys:
        - residue_indices: List[int] of residue indices
        - chain_ids: List[str] of chain identifiers
        - entity_ids: List[int] of entity identifiers

    Output keys:
        - token_index: np.ndarray (Ntokens,) int32
        - residue_index: np.ndarray (Ntokens,) int32
        - asym_id: np.ndarray (Ntokens,) int32 (chain index)
        - entity_id: np.ndarray (Ntokens,) int32
        - sym_id: np.ndarray (Ntokens,) int32 (always 0 for single copy)
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        residue_indices = data.get("residue_indices", [])
        chain_ids = data.get("chain_ids", [])
        entity_ids = data.get("entity_ids", [])

        n = len(residue_indices)

        # Token index is sequential
        token_index = np.arange(n, dtype=np.int32)

        # Residue index from input
        residue_index = np.array(residue_indices, dtype=np.int32)

        # Map chain IDs to integer indices
        chain_id_map = {}
        asym_id = np.zeros(n, dtype=np.int32)
        for i, cid in enumerate(chain_ids):
            if cid not in chain_id_map:
                chain_id_map[cid] = len(chain_id_map)
            asym_id[i] = chain_id_map[cid]

        # Entity IDs
        entity_id = np.array(entity_ids, dtype=np.int32) if entity_ids else np.zeros(n, dtype=np.int32)

        # Symmetry ID (always 0 for non-symmetry-expanded structures)
        sym_id = np.zeros(n, dtype=np.int32)

        data["token_index"] = token_index
        data["residue_index"] = residue_index
        data["asym_id"] = asym_id
        data["entity_id"] = entity_id
        data["sym_id"] = sym_id
        data["_chain_id_map"] = chain_id_map  # Store for later transforms
        return data


class SingleTokenFeatureTransform(BaseTransform):
    """Combined transform for all single-token features.

    Convenience transform that combines ResidueTypeEncode, TokenTypeEncode,
    and TokenPosition transforms.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config)
        self._transforms = [
            ResidueTypeEncodeTransform(config),
            TokenTypeEncodeTransform(config),
            TokenPositionTransform(config),
        ]

    def __call__(self, data: FeatureDict) -> FeatureDict:
        for transform in self._transforms:
            data = transform(data)
        return data


# =============================================================================
# Per-Residue Transforms
# =============================================================================


class ResidueAtomFeatureTransform(BaseTransform):
    """Extract atom-level features from residue atoms.

    From AF3 Table 5: atom_ref_pos, atom_ref_mask, atom_ref_element,
    atom_ref_charge, atom_ref_atom_name_chars, atom_ref_space_uid.

    Input keys:
        - atoms: List[Dict[str, AtomData]] per token
        - token_index: np.ndarray (Ntokens,)

    Output keys:
        - atom_ref_pos: np.ndarray (Natoms, 3) float32
        - atom_ref_mask: np.ndarray (Natoms,) float32
        - atom_ref_element: np.ndarray (Natoms, 128) int32 one-hot
        - atom_ref_charge: np.ndarray (Natoms,) float32
        - atom_ref_atom_name_chars: np.ndarray (Natoms, 4, 64) int32 one-hot
        - atom_ref_space_uid: np.ndarray (Natoms,) int32
        - atom_to_token: np.ndarray (Natoms,) int32
        - num_atoms: int
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atoms_per_token = data.get("atoms", [])

        # Count total atoms
        total_atoms = sum(
            len(atoms) if isinstance(atoms, dict) else atoms.get("count", 0)
            for atoms in atoms_per_token
        )

        if total_atoms == 0:
            # Handle empty case
            data["atom_ref_pos"] = np.zeros((0, 3), dtype=np.float32)
            data["atom_ref_mask"] = np.zeros(0, dtype=np.float32)
            data["atom_ref_element"] = np.zeros((0, self.config.num_element_types), dtype=np.int32)
            data["atom_ref_charge"] = np.zeros(0, dtype=np.float32)
            data["atom_ref_atom_name_chars"] = np.zeros(
                (0, self.config.atom_name_length, self.config.num_atom_name_chars),
                dtype=np.int32
            )
            data["atom_ref_space_uid"] = np.zeros(0, dtype=np.int32)
            data["atom_to_token"] = np.zeros(0, dtype=np.int32)
            data["num_atoms"] = 0
            return data

        # Allocate arrays
        atom_ref_pos = np.zeros((total_atoms, 3), dtype=np.float32)
        atom_ref_mask = np.zeros(total_atoms, dtype=np.float32)
        atom_ref_element = np.zeros(
            (total_atoms, self.config.num_element_types), dtype=np.int32
        )
        atom_ref_charge = np.zeros(total_atoms, dtype=np.float32)
        atom_ref_atom_name_chars = np.zeros(
            (total_atoms, self.config.atom_name_length, self.config.num_atom_name_chars),
            dtype=np.int32
        )
        atom_ref_space_uid = np.zeros(total_atoms, dtype=np.int32)
        atom_to_token = np.zeros(total_atoms, dtype=np.int32)

        atom_idx = 0
        for token_idx, atoms in enumerate(atoms_per_token):
            if isinstance(atoms, dict):
                for atom_name, atom in atoms.items():
                    # Position
                    coords = atom.coords if hasattr(atom, "coords") else atom.get("coords", [0, 0, 0])
                    atom_ref_pos[atom_idx] = coords
                    atom_ref_mask[atom_idx] = 1.0

                    # Element
                    element = atom.element if hasattr(atom, "element") else atom.get("element", "C")
                    elem_idx = ELEMENT_TO_IDX.get(element, 0)
                    atom_ref_element[atom_idx, elem_idx] = 1

                    # Charge
                    charge = atom.charge if hasattr(atom, "charge") else atom.get("charge", 0.0)
                    atom_ref_charge[atom_idx] = charge

                    # Atom name encoding
                    self._encode_atom_name(atom_name, atom_ref_atom_name_chars[atom_idx])

                    # Space UID (same for all atoms in token)
                    atom_ref_space_uid[atom_idx] = token_idx

                    # Token mapping
                    atom_to_token[atom_idx] = token_idx

                    atom_idx += 1

        data["atom_ref_pos"] = atom_ref_pos
        data["atom_ref_mask"] = atom_ref_mask
        data["atom_ref_element"] = atom_ref_element
        data["atom_ref_charge"] = atom_ref_charge
        data["atom_ref_atom_name_chars"] = atom_ref_atom_name_chars
        data["atom_ref_space_uid"] = atom_ref_space_uid
        data["atom_to_token"] = atom_to_token
        data["num_atoms"] = total_atoms
        return data

    def _encode_atom_name(self, name: str, output: np.ndarray) -> None:
        """Encode atom name to 4x64 one-hot array."""
        padded = name.ljust(4)[:4].upper()
        for i, char in enumerate(padded):
            char_idx = CHAR_TO_IDX.get(char, 0)
            output[i, char_idx] = 1


class PseudoBetaTransform(BaseTransform):
    """Compute pseudo-beta positions.

    From AF3 Table 5: pseudo_beta is CB position (CA for glycine).

    Input keys:
        - atoms: List[Dict[str, AtomData]] per token
        - residue_names: List[str] of residue names

    Output keys:
        - pseudo_beta: np.ndarray (Ntokens, 3) float32
        - pseudo_beta_mask: np.ndarray (Ntokens,) float32
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atoms_per_token = data.get("atoms", [])
        residue_names = data.get("residue_names", [])

        n = len(atoms_per_token)
        pseudo_beta = np.zeros((n, 3), dtype=np.float32)
        pseudo_beta_mask = np.zeros(n, dtype=np.float32)

        for i, (atoms, name) in enumerate(zip(atoms_per_token, residue_names)):
            if not isinstance(atoms, dict):
                continue

            # For glycine, use CA; otherwise use CB (fallback to CA)
            if name == "GLY":
                atom = atoms.get("CA")
            else:
                atom = atoms.get("CB") or atoms.get("CA")

            if atom is not None:
                coords = atom.coords if hasattr(atom, "coords") else atom.get("coords")
                if coords is not None:
                    pseudo_beta[i] = coords
                    pseudo_beta_mask[i] = 1.0

        data["pseudo_beta"] = pseudo_beta
        data["pseudo_beta_mask"] = pseudo_beta_mask
        return data


class BackboneCompleteTransform(BaseTransform):
    """Check backbone completeness for each residue.

    From AF3: Backbone is complete if N, CA, C present (protein)
    or C1', C3', C4' present (nucleic acid).

    Input keys:
        - atoms: List[Dict[str, AtomData]] per token
        - token_types: List[str] of token types

    Output keys:
        - backbone_mask: np.ndarray (Ntokens,) float32
    """

    PROTEIN_BACKBONE = {"N", "CA", "C"}
    NUCLEIC_BACKBONE = {"C1'", "C3'", "C4'"}

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atoms_per_token = data.get("atoms", [])
        token_types = data.get("token_types", [])

        n = len(atoms_per_token)
        backbone_mask = np.zeros(n, dtype=np.float32)

        for i, (atoms, ttype) in enumerate(zip(atoms_per_token, token_types)):
            if not isinstance(atoms, dict):
                continue

            atom_names = set(atoms.keys())

            if ttype == "protein":
                if self.PROTEIN_BACKBONE.issubset(atom_names):
                    backbone_mask[i] = 1.0
            elif ttype in ("rna", "dna"):
                if self.NUCLEIC_BACKBONE.issubset(atom_names):
                    backbone_mask[i] = 1.0
            else:
                # Ligands: consider complete if has atoms
                if len(atoms) >= 3:
                    backbone_mask[i] = 1.0

        data["backbone_mask"] = backbone_mask
        return data


# =============================================================================
# Frame Construction Transforms (AF3 Section 4.3.2)
# =============================================================================


@dataclass(frozen=True)
class FrameConfig:
    """Configuration for frame construction.

    From AF3 Section 4.3.2: Frames are constructed from three atoms.

    Attributes:
        colinearity_threshold_low: Minimum angle (degrees) before frame invalid.
        colinearity_threshold_high: Maximum angle (degrees) before frame invalid.
        use_kdtree: Whether to use KDTree for ligand frame construction.
    """

    colinearity_threshold_low: float = 25.0
    colinearity_threshold_high: float = 155.0
    use_kdtree: bool = True


class TokenFrameTransform(BaseTransform):
    """Construct local coordinate frames for each token.

    From AF3 Section 4.3.2:
    - Protein tokens use backbone atoms [N, CA, C]
    - DNA/RNA tokens use sugar atoms [C1', C3', C4']
    - Ligand tokens use center atom + two nearest neighbors from reference conformer

    The frame is invalid if:
    - Required atoms are missing
    - Three atoms are colinear (angle < 25 or > 155 degrees)
    - Fewer than 3 atoms exist in the residue

    Input keys:
        - atoms: List[Dict[str, AtomData]] per token
        - token_types: List[str] of token types
        - atom_ref_pos: np.ndarray (Natoms, 3) reference positions
        - atom_ref_mask: np.ndarray (Natoms,) reference mask
        - atom_to_token: np.ndarray (Natoms,) atom to token mapping

    Output keys:
        - has_frame: np.ndarray (Ntokens,) int32, 1 if valid frame
        - frame_atom_index: np.ndarray (Ntokens, 3) int32, indices of [a, b, c] atoms
    """

    # Backbone atoms for frame construction
    PROTEIN_FRAME_ATOMS = ["N", "CA", "C"]
    NUCLEIC_FRAME_ATOMS = ["C1'", "C3'", "C4'"]

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        frame_config: Optional[FrameConfig] = None,
    ):
        super().__init__(config)
        self.frame_config = frame_config or FrameConfig()

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atoms_per_token = data.get("atoms", [])
        token_types = data.get("token_types", [])
        atom_ref_pos = data.get("atom_ref_pos", np.zeros((0, 3)))
        atom_ref_mask = data.get("atom_ref_mask", np.array([]))
        atom_to_token = data.get("atom_to_token", np.array([]))

        n_tokens = len(atoms_per_token)

        has_frame = np.zeros(n_tokens, dtype=np.int32)
        frame_atom_index = np.full((n_tokens, 3), -1, dtype=np.int32)

        if n_tokens == 0:
            data["has_frame"] = has_frame
            data["frame_atom_index"] = frame_atom_index
            return data

        # Build atom name to global index mapping per token
        token_atom_maps = self._build_token_atom_maps(atoms_per_token, atom_to_token)

        # Build KDTree for ligand frame construction if needed
        ligand_kdtrees = {}
        if self.frame_config.use_kdtree:
            ligand_kdtrees = self._build_ligand_kdtrees(
                atoms_per_token, token_types, atom_ref_pos, atom_ref_mask, atom_to_token
            )

        for i, (atoms, ttype) in enumerate(zip(atoms_per_token, token_types)):
            if not isinstance(atoms, dict) or len(atoms) == 0:
                continue

            atom_map = token_atom_maps[i]

            if ttype == "protein":
                valid, indices = self._get_polymer_frame(
                    atoms, atom_map, self.PROTEIN_FRAME_ATOMS
                )
            elif ttype in ("rna", "dna"):
                valid, indices = self._get_polymer_frame(
                    atoms, atom_map, self.NUCLEIC_FRAME_ATOMS
                )
            else:
                # Ligand frame
                valid, indices = self._get_ligand_frame(
                    i, atoms, atom_map, atom_ref_pos, atom_ref_mask,
                    ligand_kdtrees.get(i)
                )

            if valid:
                # Colinearity check
                if self._check_colinearity(atom_ref_pos, indices):
                    has_frame[i] = 1
                    frame_atom_index[i] = indices

        data["has_frame"] = has_frame
        data["frame_atom_index"] = frame_atom_index
        return data

    def _build_token_atom_maps(
        self,
        atoms_per_token: List[Dict],
        atom_to_token: np.ndarray,
    ) -> List[Dict[str, int]]:
        """Build mapping from atom name to global index for each token."""
        n_tokens = len(atoms_per_token)
        token_atom_maps = [{} for _ in range(n_tokens)]

        # Build reverse mapping from global index to token
        if len(atom_to_token) > 0:
            global_idx = 0
            for token_idx, atoms in enumerate(atoms_per_token):
                if isinstance(atoms, dict):
                    for atom_name in atoms.keys():
                        token_atom_maps[token_idx][atom_name] = global_idx
                        global_idx += 1

        return token_atom_maps

    def _build_ligand_kdtrees(
        self,
        atoms_per_token: List[Dict],
        token_types: List[str],
        atom_ref_pos: np.ndarray,
        atom_ref_mask: np.ndarray,
        atom_to_token: np.ndarray,
    ) -> Dict[int, Tuple[Any, List[int]]]:
        """Build KDTrees for ligand tokens for nearest neighbor queries."""
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return {}

        ligand_kdtrees = {}

        for token_idx, (atoms, ttype) in enumerate(zip(atoms_per_token, token_types)):
            if ttype not in ("ligand", "other") or not isinstance(atoms, dict):
                continue

            # Get atom indices for this token
            atom_indices = np.where(atom_to_token == token_idx)[0]

            if len(atom_indices) < 3:
                ligand_kdtrees[token_idx] = (None, list(atom_indices))
                continue

            # Get valid positions
            valid_mask = atom_ref_mask[atom_indices] > 0
            valid_indices = atom_indices[valid_mask]

            if len(valid_indices) < 3:
                ligand_kdtrees[token_idx] = (None, list(atom_indices))
                continue

            positions = atom_ref_pos[valid_indices]
            kdtree = cKDTree(positions)
            ligand_kdtrees[token_idx] = (kdtree, list(valid_indices))

        return ligand_kdtrees

    def _get_polymer_frame(
        self,
        atoms: Dict,
        atom_map: Dict[str, int],
        frame_atoms: List[str],
    ) -> Tuple[bool, np.ndarray]:
        """Get frame for protein/nucleic acid tokens."""
        indices = np.array([-1, -1, -1], dtype=np.int32)

        # Check all required atoms exist
        for atom_name in frame_atoms:
            if atom_name not in atoms or atom_name not in atom_map:
                return False, indices

        # Get indices in order [a, b, c]
        for i, atom_name in enumerate(frame_atoms):
            indices[i] = atom_map[atom_name]

        return True, indices

    def _get_ligand_frame(
        self,
        token_idx: int,
        atoms: Dict,
        atom_map: Dict[str, int],
        atom_ref_pos: np.ndarray,
        atom_ref_mask: np.ndarray,
        kdtree_data: Optional[Tuple],
    ) -> Tuple[bool, np.ndarray]:
        """Get frame for ligand tokens using nearest neighbors.

        From AF3 Section 4.3.2:
        - b is the center atom (first atom in token)
        - a is the closest atom to b
        - c is the second closest atom to b
        """
        indices = np.array([-1, -1, -1], dtype=np.int32)

        if len(atoms) < 3:
            return False, indices

        # Get center atom (first atom in token)
        atom_names = list(atoms.keys())
        center_atom_name = atom_names[0]

        if center_atom_name not in atom_map:
            return False, indices

        b_idx = atom_map[center_atom_name]
        indices[1] = b_idx  # b is center

        if kdtree_data is None:
            return False, indices

        kdtree, valid_indices = kdtree_data

        if kdtree is None or len(valid_indices) < 3:
            return False, indices

        # Query nearest neighbors (k=3 to get self + 2 neighbors)
        center_pos = atom_ref_pos[b_idx:b_idx+1]
        distances, neighbor_indices = kdtree.query(center_pos, k=3)

        # Map back to global indices
        neighbor_global = [valid_indices[i] for i in neighbor_indices[0]]

        # a is closest (excluding self), c is second closest
        neighbors_excluding_self = [
            idx for idx in neighbor_global if idx != b_idx
        ]

        if len(neighbors_excluding_self) < 2:
            return False, indices

        indices[0] = neighbors_excluding_self[0]  # a
        indices[2] = neighbors_excluding_self[1]  # c

        # Verify all positions are valid
        if not all(atom_ref_mask[idx] > 0 for idx in indices if idx >= 0):
            return False, indices

        return True, indices

    def _check_colinearity(
        self,
        positions: np.ndarray,
        indices: np.ndarray,
    ) -> bool:
        """Check if three atoms are not colinear.

        Returns True if the angle is within valid range (not colinear).
        """
        if np.any(indices < 0) or np.any(indices >= len(positions)):
            return False

        a_pos = positions[indices[0]]
        b_pos = positions[indices[1]]
        c_pos = positions[indices[2]]

        # Compute angle at b
        vec_ba = a_pos - b_pos
        vec_bc = c_pos - b_pos

        norm_ba = np.linalg.norm(vec_ba)
        norm_bc = np.linalg.norm(vec_bc)

        if norm_ba < 1e-8 or norm_bc < 1e-8:
            return False

        cos_angle = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_degrees = np.degrees(np.arccos(cos_angle))

        # Check if NOT colinear
        return (
            angle_degrees > self.frame_config.colinearity_threshold_low
            and angle_degrees < self.frame_config.colinearity_threshold_high
        )


class FrameFromPositionsTransform(BaseTransform):
    """Compute 4x4 transformation matrices from frame atoms.

    Given frame atom indices [a, b, c], construct the local coordinate frame:
    - Origin at b (center atom)
    - X-axis: b -> c direction
    - Y-axis: perpendicular to X in the a-b-c plane
    - Z-axis: cross product of X and Y

    Input keys:
        - atom_ref_pos: np.ndarray (Natoms, 3)
        - has_frame: np.ndarray (Ntokens,)
        - frame_atom_index: np.ndarray (Ntokens, 3)

    Output keys:
        - backbone_rigid_tensor: np.ndarray (Ntokens, 4, 4) transformation matrices
        - backbone_rigid_mask: np.ndarray (Ntokens,) validity mask
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atom_ref_pos = data.get("atom_ref_pos", np.zeros((0, 3)))
        has_frame = data.get("has_frame", np.array([]))
        frame_atom_index = data.get("frame_atom_index", np.zeros((0, 3), dtype=np.int32))

        n_tokens = len(has_frame) if len(has_frame) > 0 else 0

        backbone_rigid_tensor = np.zeros((n_tokens, 4, 4), dtype=np.float32)
        backbone_rigid_mask = np.zeros(n_tokens, dtype=np.float32)

        # Initialize with identity matrices
        for i in range(n_tokens):
            backbone_rigid_tensor[i] = np.eye(4, dtype=np.float32)

        for i in range(n_tokens):
            if has_frame[i] == 0:
                continue

            a_idx, b_idx, c_idx = frame_atom_index[i]

            if a_idx < 0 or b_idx < 0 or c_idx < 0:
                continue

            a_pos = atom_ref_pos[a_idx]
            b_pos = atom_ref_pos[b_idx]  # Origin
            c_pos = atom_ref_pos[c_idx]

            # X-axis: b -> c
            x_axis = c_pos - b_pos
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-6:
                continue
            x_axis = x_axis / x_norm

            # Y-axis: perpendicular in a-b-c plane
            a_to_b = b_pos - a_pos
            y_axis = a_to_b - np.dot(a_to_b, x_axis) * x_axis
            y_norm = np.linalg.norm(y_axis)
            if y_norm < 1e-6:
                continue
            y_axis = y_axis / y_norm

            # Z-axis: cross product (right-handed)
            z_axis = np.cross(x_axis, y_axis)

            # Build 4x4 transformation matrix
            backbone_rigid_tensor[i, :3, 0] = x_axis
            backbone_rigid_tensor[i, :3, 1] = y_axis
            backbone_rigid_tensor[i, :3, 2] = z_axis
            backbone_rigid_tensor[i, :3, 3] = b_pos  # Translation
            backbone_rigid_tensor[i, 3, 3] = 1.0

            backbone_rigid_mask[i] = 1.0

        data["backbone_rigid_tensor"] = backbone_rigid_tensor
        data["backbone_rigid_mask"] = backbone_rigid_mask
        return data


# =============================================================================
# Reference Conformer Augmentation Transforms
# =============================================================================


@dataclass(frozen=True)
class AugmentConfig:
    """Configuration for reference conformer augmentation.

    Attributes:
        apply_rotation: Whether to apply random rotation.
        apply_translation: Whether to apply random translation.
        translation_std: Standard deviation for translation noise.
        centralize: Whether to center positions at origin.
    """

    apply_rotation: bool = True
    apply_translation: bool = True
    translation_std: float = 1.0
    centralize: bool = True


class RefPosAugmentTransform(BaseTransform):
    """Apply random augmentation to reference conformer positions.

    From AF3: Reference positions are augmented with random rotation
    and translation to prevent overfitting to specific orientations.

    Each residue (ref_space_uid) is augmented independently.

    Input keys:
        - atom_ref_pos: np.ndarray (Natoms, 3)
        - atom_ref_space_uid: np.ndarray (Natoms,)

    Output keys:
        - atom_ref_pos: np.ndarray (Natoms, 3) augmented positions
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        augment_config: Optional[AugmentConfig] = None,
    ):
        super().__init__(config)
        self.augment_config = augment_config or AugmentConfig()

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atom_ref_pos = data.get("atom_ref_pos", np.zeros((0, 3)))
        atom_ref_space_uid = data.get("atom_ref_space_uid", np.array([]))

        if len(atom_ref_pos) == 0:
            return data

        augmented_pos = atom_ref_pos.copy()

        # Process each residue independently
        unique_uids = np.unique(atom_ref_space_uid)

        for uid in unique_uids:
            mask = atom_ref_space_uid == uid
            residue_pos = augmented_pos[mask]

            # Centralize
            if self.augment_config.centralize:
                center = residue_pos.mean(axis=0)
                residue_pos = residue_pos - center

            # Random rotation
            if self.augment_config.apply_rotation:
                rotation = self._random_rotation_matrix()
                residue_pos = residue_pos @ rotation.T

            # Random translation
            if self.augment_config.apply_translation:
                translation = np.random.randn(3) * self.augment_config.translation_std
                residue_pos = residue_pos + translation

            augmented_pos[mask] = residue_pos

        data["atom_ref_pos"] = augmented_pos
        return data

    def _random_rotation_matrix(self) -> np.ndarray:
        """Generate a random 3x3 rotation matrix using QR decomposition."""
        random_matrix = np.random.randn(3, 3)
        q, r = np.linalg.qr(random_matrix)
        # Ensure proper rotation (det = 1)
        d = np.diag(np.sign(np.diag(r)))
        rotation = q @ d
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        return rotation.astype(np.float32)


# =============================================================================
# Ligand Atom Renaming Transform
# =============================================================================


class LigandAtomRenameTransform(BaseTransform):
    """Rename ligand atoms to prevent information leakage.

    From AF3: Ligand atom names are renamed to element + count
    (e.g., C1, C2, N1) to avoid leaking structural information
    through atom naming conventions.

    Input keys:
        - atoms: List[Dict[str, AtomData]] per token
        - token_types: List[str]

    Output keys:
        - atoms: Updated with renamed atom names for ligands
        - original_atom_names: List[List[str]] original names for reference
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atoms_per_token = data.get("atoms", [])
        token_types = data.get("token_types", [])

        original_names = []
        renamed_atoms = []

        for atoms, ttype in zip(atoms_per_token, token_types):
            if not isinstance(atoms, dict):
                renamed_atoms.append(atoms)
                original_names.append([])
                continue

            if ttype not in ("ligand", "other"):
                renamed_atoms.append(atoms)
                original_names.append(list(atoms.keys()))
                continue

            # Rename ligand atoms
            element_counts = {}
            new_atoms = {}
            token_original_names = []

            for atom_name, atom_data in atoms.items():
                token_original_names.append(atom_name)

                # Get element
                element = atom_data.element if hasattr(atom_data, "element") else atom_data.get("element", "X")
                element = element.upper()

                # Generate new name
                element_counts[element] = element_counts.get(element, 0) + 1
                new_name = f"{element}{element_counts[element]}"

                new_atoms[new_name] = atom_data

            renamed_atoms.append(new_atoms)
            original_names.append(token_original_names)

        data["atoms"] = renamed_atoms
        data["original_atom_names"] = original_names
        return data


# =============================================================================
# Token Bond Transforms
# =============================================================================


@dataclass(frozen=True)
class BondConfig:
    """Configuration for token bond feature extraction.

    From AF3 Table 5: token_bonds restricted to polymer-ligand
    and ligand-ligand bonds.

    Attributes:
        include_polymer_ligand: Include polymer-ligand bonds.
        include_ligand_ligand: Include ligand-ligand bonds.
        include_disulfide: Include disulfide bonds.
        exclude_standard_polymer: Exclude standard polymer-polymer bonds.
        bond_distance_threshold: Maximum distance for bond detection.
    """

    include_polymer_ligand: bool = True
    include_ligand_ligand: bool = True
    include_disulfide: bool = True
    exclude_standard_polymer: bool = True
    bond_distance_threshold: float = 2.4


class TokenBondTransform(BaseTransform):
    """Extract token-level bond features.

    From AF3 Table 5: token_bonds is a 2D matrix indicating bonds
    between tokens, restricted to polymer-ligand and ligand-ligand bonds.

    Input keys:
        - bonds: List of (atom_i, atom_j, bond_type) tuples
        - atom_to_token: np.ndarray (Natoms,)
        - token_types: List[str]
        - num_tokens: int

    Output keys:
        - token_bonds: np.ndarray (Ntokens, Ntokens) adjacency matrix
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        bond_config: Optional[BondConfig] = None,
    ):
        super().__init__(config)
        self.bond_config = bond_config or BondConfig()

    def __call__(self, data: FeatureDict) -> FeatureDict:
        bonds = data.get("bonds", [])
        atom_to_token = data.get("atom_to_token", np.array([]))
        token_types = data.get("token_types", [])
        num_tokens = data.get("num_tokens", len(token_types))

        token_bonds = np.zeros((num_tokens, num_tokens), dtype=np.int32)

        if len(bonds) == 0 or len(atom_to_token) == 0:
            data["token_bonds"] = token_bonds
            return data

        # Classify tokens by type
        polymer_tokens = set()
        ligand_tokens = set()

        for i, ttype in enumerate(token_types):
            if ttype in ("protein", "rna", "dna"):
                polymer_tokens.add(i)
            else:
                ligand_tokens.add(i)

        # Process bonds
        for bond in bonds:
            if len(bond) >= 2:
                atom_i, atom_j = bond[0], bond[1]

                if atom_i >= len(atom_to_token) or atom_j >= len(atom_to_token):
                    continue

                token_i = atom_to_token[atom_i]
                token_j = atom_to_token[atom_j]

                if token_i == token_j:
                    continue  # Skip intra-token bonds

                # Determine bond type
                is_polymer_i = token_i in polymer_tokens
                is_polymer_j = token_j in polymer_tokens
                is_ligand_i = token_i in ligand_tokens
                is_ligand_j = token_j in ligand_tokens

                include_bond = False

                # Polymer-ligand bonds
                if self.bond_config.include_polymer_ligand:
                    if (is_polymer_i and is_ligand_j) or (is_ligand_i and is_polymer_j):
                        include_bond = True

                # Ligand-ligand bonds
                if self.bond_config.include_ligand_ligand:
                    if is_ligand_i and is_ligand_j:
                        include_bond = True

                # Exclude standard polymer-polymer bonds
                if self.bond_config.exclude_standard_polymer:
                    if is_polymer_i and is_polymer_j:
                        include_bond = False

                if include_bond:
                    token_bonds[token_i, token_j] = 1
                    token_bonds[token_j, token_i] = 1

        data["token_bonds"] = token_bonds
        return data


# =============================================================================
# Mask Feature Transforms
# =============================================================================


class RepresentativeAtomMaskTransform(BaseTransform):
    """Compute representative atom masks for various metrics.

    From AF3: Different metrics use different representative atoms:
    - PAE: Center atoms (CA for protein, C1' for nucleic, center for ligand)
    - pLDDT: All heavy atoms
    - Distogram: CB atoms (CA for glycine)

    Input keys:
        - atoms: List[Dict[str, AtomData]] per token
        - token_types: List[str]
        - residue_names: List[str]
        - atom_to_token: np.ndarray (Natoms,)

    Output keys:
        - pae_rep_atom_mask: np.ndarray (Natoms,) center atoms
        - plddt_rep_atom_mask: np.ndarray (Natoms,) all heavy atoms
        - distogram_rep_atom_mask: np.ndarray (Natoms,) CB/CA atoms
    """

    # Center atoms by molecule type
    PROTEIN_CENTER = "CA"
    NUCLEIC_CENTER = "C1'"

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atoms_per_token = data.get("atoms", [])
        token_types = data.get("token_types", [])
        residue_names = data.get("residue_names", [])
        atom_to_token = data.get("atom_to_token", np.array([]))

        num_atoms = len(atom_to_token)

        pae_rep_mask = np.zeros(num_atoms, dtype=np.int32)
        plddt_rep_mask = np.ones(num_atoms, dtype=np.int32)  # All atoms by default
        distogram_rep_mask = np.zeros(num_atoms, dtype=np.int32)

        # Build atom index mapping
        global_idx = 0
        for token_idx, (atoms, ttype, res_name) in enumerate(
            zip(atoms_per_token, token_types, residue_names)
        ):
            if not isinstance(atoms, dict):
                continue

            atom_names = list(atoms.keys())

            for atom_name in atom_names:
                # PAE representative (center atoms)
                if ttype == "protein" and atom_name == self.PROTEIN_CENTER:
                    pae_rep_mask[global_idx] = 1
                elif ttype in ("rna", "dna") and atom_name == self.NUCLEIC_CENTER:
                    pae_rep_mask[global_idx] = 1
                elif ttype in ("ligand", "other"):
                    # First atom is center for ligands
                    if atom_name == atom_names[0]:
                        pae_rep_mask[global_idx] = 1

                # Distogram representative (CB or CA for GLY)
                if ttype == "protein":
                    if res_name == "GLY" and atom_name == "CA":
                        distogram_rep_mask[global_idx] = 1
                    elif atom_name == "CB":
                        distogram_rep_mask[global_idx] = 1

                global_idx += 1

        data["pae_rep_atom_mask"] = pae_rep_mask
        data["plddt_rep_atom_mask"] = plddt_rep_mask
        data["distogram_rep_atom_mask"] = distogram_rep_mask
        return data


class ModifiedResidueMaskTransform(BaseTransform):
    """Identify modified residues.

    Modified residues are non-standard residues in polymer chains
    (e.g., phosphorylated amino acids, methylated bases).

    Input keys:
        - residue_names: List[str]
        - token_types: List[str]

    Output keys:
        - modified_res_mask: np.ndarray (Ntokens,) int32
    """

    # Standard residue names
    STANDARD_AA = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL",
    }
    STANDARD_RNA = {"A", "G", "C", "U"}
    STANDARD_DNA = {"DA", "DG", "DC", "DT"}

    def __call__(self, data: FeatureDict) -> FeatureDict:
        residue_names = data.get("residue_names", [])
        token_types = data.get("token_types", [])

        n_tokens = len(residue_names)
        modified_mask = np.zeros(n_tokens, dtype=np.int32)

        for i, (name, ttype) in enumerate(zip(residue_names, token_types)):
            is_modified = False

            if ttype == "protein" and name not in self.STANDARD_AA:
                is_modified = True
            elif ttype == "rna" and name not in self.STANDARD_RNA:
                is_modified = True
            elif ttype == "dna" and name not in self.STANDARD_DNA:
                is_modified = True

            if is_modified:
                modified_mask[i] = 1

        data["modified_res_mask"] = modified_mask
        return data


# =============================================================================
# Atom Permutation Transform
# =============================================================================


class AtomPermutationTransform(BaseTransform):
    """Compute valid atom permutations for symmetric residues.

    From AF3: Some residues have symmetric atoms that can be permuted
    without changing the structure (e.g., ARG NH1/NH2, PHE ring carbons).

    This transform computes the valid permutation indices for each residue.

    Input keys:
        - atoms: List[Dict[str, AtomData]] per token
        - residue_names: List[str]

    Output keys:
        - atom_permutations: List[np.ndarray] permutation matrices per residue
    """

    # Symmetric atom groups for amino acids
    SYMMETRIC_ATOMS = {
        "ARG": [["NH1", "NH2"]],
        "ASP": [["OD1", "OD2"]],
        "GLU": [["OE1", "OE2"]],
        "LEU": [["CD1", "CD2"]],
        "PHE": [["CD1", "CD2"], ["CE1", "CE2"]],
        "TYR": [["CD1", "CD2"], ["CE1", "CE2"]],
        "VAL": [["CG1", "CG2"]],
    }

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atoms_per_token = data.get("atoms", [])
        residue_names = data.get("residue_names", [])

        atom_permutations = []

        for atoms, res_name in zip(atoms_per_token, residue_names):
            if not isinstance(atoms, dict):
                atom_permutations.append(np.array([]))
                continue

            atom_names = list(atoms.keys())
            n_atoms = len(atom_names)

            # Identity permutation
            identity = np.arange(n_atoms, dtype=np.int32)
            permutations = [identity]

            # Check for symmetric atoms
            if res_name in self.SYMMETRIC_ATOMS:
                for sym_group in self.SYMMETRIC_ATOMS[res_name]:
                    # Check if both atoms exist
                    if all(atom in atom_names for atom in sym_group):
                        # Create swapped permutation
                        swapped = identity.copy()
                        idx1 = atom_names.index(sym_group[0])
                        idx2 = atom_names.index(sym_group[1])
                        swapped[idx1], swapped[idx2] = swapped[idx2], swapped[idx1]
                        permutations.append(swapped)

            atom_permutations.append(np.array(permutations, dtype=np.int32))

        data["atom_permutations"] = atom_permutations
        return data


# =============================================================================
# Ligand Pocket Mask Transform
# =============================================================================


@dataclass(frozen=True)
class PocketConfig:
    """Configuration for ligand pocket detection.

    Attributes:
        pocket_radius: Radius (Angstroms) for pocket definition.
        protein_backbone_only: Only include backbone atoms for protein pocket.
    """

    pocket_radius: float = 10.0
    protein_backbone_only: bool = True


class LigandPocketMaskTransform(BaseTransform):
    """Compute ligand binding pocket masks.

    From AF3 Methods: The pocket is defined as all heavy atoms within
    10 Angstroms of any heavy atom of the ligand, restricted to
    backbone atoms for proteins.

    Input keys:
        - atom_ref_pos: np.ndarray (Natoms, 3)
        - atom_to_token: np.ndarray (Natoms,)
        - token_types: List[str]
        - atoms: List[Dict[str, AtomData]] per token

    Output keys:
        - ligand_mask: np.ndarray (Natoms,) atoms belonging to ligands
        - pocket_mask: np.ndarray (Natoms,) atoms in binding pocket
    """

    PROTEIN_BACKBONE = {"N", "CA", "C", "O"}

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        pocket_config: Optional[PocketConfig] = None,
    ):
        super().__init__(config)
        self.pocket_config = pocket_config or PocketConfig()

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atom_ref_pos = data.get("atom_ref_pos", np.zeros((0, 3)))
        atom_to_token = data.get("atom_to_token", np.array([]))
        token_types = data.get("token_types", [])
        atoms_per_token = data.get("atoms", [])

        num_atoms = len(atom_to_token)

        ligand_mask = np.zeros(num_atoms, dtype=np.int32)
        pocket_mask = np.zeros(num_atoms, dtype=np.int32)

        if num_atoms == 0:
            data["ligand_mask"] = ligand_mask
            data["pocket_mask"] = pocket_mask
            return data

        # Build ligand atom mask
        ligand_token_indices = set()
        for i, ttype in enumerate(token_types):
            if ttype in ("ligand", "other"):
                ligand_token_indices.add(i)

        for atom_idx in range(num_atoms):
            token_idx = atom_to_token[atom_idx]
            if token_idx in ligand_token_indices:
                ligand_mask[atom_idx] = 1

        # Find ligand positions
        ligand_positions = atom_ref_pos[ligand_mask > 0]

        if len(ligand_positions) == 0:
            data["ligand_mask"] = ligand_mask
            data["pocket_mask"] = pocket_mask
            return data

        # Build protein backbone mask
        backbone_mask = np.zeros(num_atoms, dtype=bool)
        global_idx = 0

        for token_idx, (atoms, ttype) in enumerate(zip(atoms_per_token, token_types)):
            if not isinstance(atoms, dict):
                continue

            for atom_name in atoms.keys():
                if ttype == "protein":
                    if self.pocket_config.protein_backbone_only:
                        if atom_name in self.PROTEIN_BACKBONE:
                            backbone_mask[global_idx] = True
                    else:
                        backbone_mask[global_idx] = True
                global_idx += 1

        # Use KDTree to find atoms within pocket radius
        try:
            from scipy.spatial import cKDTree

            kdtree = cKDTree(atom_ref_pos)
            pocket_indices = set()

            for lig_pos in ligand_positions:
                indices = kdtree.query_ball_point(
                    lig_pos, self.pocket_config.pocket_radius
                )
                pocket_indices.update(indices)

            for idx in pocket_indices:
                if backbone_mask[idx]:
                    pocket_mask[idx] = 1

        except ImportError:
            # Fallback: brute force distance calculation
            for atom_idx in range(num_atoms):
                if not backbone_mask[atom_idx]:
                    continue

                atom_pos = atom_ref_pos[atom_idx]
                distances = np.linalg.norm(ligand_positions - atom_pos, axis=1)

                if np.min(distances) <= self.pocket_config.pocket_radius:
                    pocket_mask[atom_idx] = 1

        data["ligand_mask"] = ligand_mask
        data["pocket_mask"] = pocket_mask
        return data


# =============================================================================
# Per-Chain Transforms
# =============================================================================


class ChainTokenFeatureTransform(BaseTransform):
    """Aggregate token features by chain.

    Useful for chain-level operations and filtering.

    Input keys:
        - asym_id: np.ndarray (Ntokens,) chain indices
        - token_index: np.ndarray (Ntokens,)

    Output keys:
        - chain_token_indices: Dict[int, np.ndarray] token indices per chain
        - chain_lengths: Dict[int, int] number of tokens per chain
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        asym_id = data.get("asym_id", np.array([]))

        chain_token_indices = {}
        chain_lengths = {}

        if len(asym_id) > 0:
            unique_chains = np.unique(asym_id)
            for chain_idx in unique_chains:
                mask = asym_id == chain_idx
                chain_token_indices[int(chain_idx)] = np.where(mask)[0]
                chain_lengths[int(chain_idx)] = int(np.sum(mask))

        data["chain_token_indices"] = chain_token_indices
        data["chain_lengths"] = chain_lengths
        return data


class ChainAtomFeatureTransform(BaseTransform):
    """Aggregate atom features by chain.

    Input keys:
        - atom_to_token: np.ndarray (Natoms,)
        - asym_id: np.ndarray (Ntokens,)

    Output keys:
        - chain_atom_indices: Dict[int, np.ndarray] atom indices per chain
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atom_to_token = data.get("atom_to_token", np.array([]))
        asym_id = data.get("asym_id", np.array([]))

        chain_atom_indices = {}

        if len(atom_to_token) > 0 and len(asym_id) > 0:
            # Map atoms to chains via tokens
            atom_chain_ids = asym_id[atom_to_token]
            unique_chains = np.unique(atom_chain_ids)

            for chain_idx in unique_chains:
                mask = atom_chain_ids == chain_idx
                chain_atom_indices[int(chain_idx)] = np.where(mask)[0]

        data["chain_atom_indices"] = chain_atom_indices
        return data


class ChainCenterCoordsTransform(BaseTransform):
    """Compute center of mass for each chain.

    Input keys:
        - atom_ref_pos: np.ndarray (Natoms, 3)
        - atom_ref_mask: np.ndarray (Natoms,)
        - chain_atom_indices: Dict[int, np.ndarray]

    Output keys:
        - chain_centers: Dict[int, np.ndarray] center coords per chain
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        atom_ref_pos = data.get("atom_ref_pos", np.zeros((0, 3)))
        atom_ref_mask = data.get("atom_ref_mask", np.array([]))
        chain_atom_indices = data.get("chain_atom_indices", {})

        chain_centers = {}

        for chain_idx, atom_indices in chain_atom_indices.items():
            if len(atom_indices) > 0:
                mask = atom_ref_mask[atom_indices]
                if mask.sum() > 0:
                    coords = atom_ref_pos[atom_indices]
                    center = np.average(coords, weights=mask, axis=0)
                    chain_centers[chain_idx] = center
                else:
                    chain_centers[chain_idx] = np.zeros(3, dtype=np.float32)
            else:
                chain_centers[chain_idx] = np.zeros(3, dtype=np.float32)

        data["chain_centers"] = chain_centers
        return data


# =============================================================================
# Per-Sequence Transforms (MSA)
# =============================================================================


class SequenceEncodeTransform(BaseTransform):
    """Encode a single sequence to residue type indices.

    Input keys:
        - sequence: str of amino acid one-letter codes

    Output keys:
        - sequence_encoded: np.ndarray (L,) int32
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        sequence = data.get("sequence", "")

        encoded = np.zeros(len(sequence), dtype=np.int32)
        for i, char in enumerate(sequence):
            if char == "-":
                encoded[i] = GAP_IDX
            elif char in AA_1TO3:
                aa3 = AA_1TO3[char]
                encoded[i] = AA_MAP.get(aa3, AA_UNK_IDX)
            else:
                encoded[i] = AA_UNK_IDX

        data["sequence_encoded"] = encoded
        return data


class MSARowTransform(BaseTransform):
    """Encode MSA rows with deletion values.

    From AF3 Table 5: msa, msa_mask, msa_deletion_value.

    Input keys:
        - msa_sequences: List[str] aligned sequences
        - msa_deletions: Optional[List[List[int]]] deletion counts
        - num_tokens: int number of query tokens

    Output keys:
        - msa: np.ndarray (Nmsa, Ntokens) int32
        - msa_mask: np.ndarray (Nmsa, Ntokens) float32
        - msa_deletion_value: np.ndarray (Nmsa, Ntokens) float32
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        msa_config: Optional[MSAConfig] = None,
    ):
        super().__init__(config)
        self.msa_config = msa_config or MSAConfig()

    def __call__(self, data: FeatureDict) -> FeatureDict:
        msa_sequences = data.get("msa_sequences", [])
        msa_deletions = data.get("msa_deletions")
        num_tokens = data.get("num_tokens", 0)

        if not msa_sequences or num_tokens == 0:
            data["msa"] = np.full((1, num_tokens), GAP_IDX, dtype=np.int32)
            data["msa_mask"] = np.zeros((1, num_tokens), dtype=np.float32)
            data["msa_deletion_value"] = np.zeros((1, num_tokens), dtype=np.float32)
            return data

        n_seqs = min(len(msa_sequences), self.msa_config.max_sequences)

        msa = np.full((n_seqs, num_tokens), GAP_IDX, dtype=np.int32)
        msa_mask = np.zeros((n_seqs, num_tokens), dtype=np.float32)
        msa_deletion_value = np.zeros((n_seqs, num_tokens), dtype=np.float32)

        for seq_idx, seq in enumerate(msa_sequences[:n_seqs]):
            token_idx = 0
            for pos, char in enumerate(seq):
                if token_idx >= num_tokens:
                    break
                if char == "-":
                    continue

                # Encode character
                if char in AA_1TO3:
                    aa3 = AA_1TO3[char]
                    msa[seq_idx, token_idx] = AA_MAP.get(aa3, AA_UNK_IDX)
                else:
                    msa[seq_idx, token_idx] = AA_UNK_IDX

                msa_mask[seq_idx, token_idx] = 1.0

                # Deletion value (arctan transform)
                if msa_deletions and seq_idx < len(msa_deletions):
                    if pos < len(msa_deletions[seq_idx]):
                        del_count = msa_deletions[seq_idx][pos]
                        msa_deletion_value[seq_idx, token_idx] = (
                            2.0 / np.pi * np.arctan(del_count / 3.0)
                        )

                token_idx += 1

        data["msa"] = msa
        data["msa_mask"] = msa_mask
        data["msa_deletion_value"] = msa_deletion_value
        return data


# =============================================================================
# Pair Transforms
# =============================================================================


class RelativePositionTransform(BaseTransform):
    """Compute relative position encoding.

    From AF3 Table 5: relative_position encodes sequence distance,
    clipped to [-max, max] for same chain, special value for different chains.

    Input keys:
        - residue_index: np.ndarray (Ntokens,)
        - asym_id: np.ndarray (Ntokens,) chain indices

    Output keys:
        - relative_position: np.ndarray (Ntokens, Ntokens) int32
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        residue_index = data.get("residue_index", np.array([]))
        asym_id = data.get("asym_id", np.array([]))

        if len(residue_index) == 0:
            data["relative_position"] = np.zeros((0, 0), dtype=np.int32)
            return data

        # Use vectorized computation
        relative_position = compute_relative_position_vectorized(
            residue_index.astype(np.int32),
            asym_id.astype(np.int32),
            self.config.max_relative_position,
        )

        data["relative_position"] = relative_position
        return data


class SameChainMaskTransform(BaseTransform):
    """Compute same-chain mask.

    From AF3 Table 5: same_chain is 1.0 if tokens are in same chain.

    Input keys:
        - asym_id: np.ndarray (Ntokens,) chain indices

    Output keys:
        - same_chain: np.ndarray (Ntokens, Ntokens) float32
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        asym_id = data.get("asym_id", np.array([]))

        if len(asym_id) == 0:
            data["same_chain"] = np.zeros((0, 0), dtype=np.float32)
            return data

        same_chain, _ = compute_pair_masks_vectorized(asym_id.astype(np.int32))
        data["same_chain"] = same_chain
        return data


class SameEntityMaskTransform(BaseTransform):
    """Compute same-entity mask.

    From AF3 Table 5: same_entity is 1.0 if tokens are in same entity.

    Input keys:
        - entity_id: np.ndarray (Ntokens,)

    Output keys:
        - same_entity: np.ndarray (Ntokens, Ntokens) float32
    """

    def __call__(self, data: FeatureDict) -> FeatureDict:
        entity_id = data.get("entity_id", np.array([]))
        asym_id = data.get("asym_id", np.array([]))

        if len(entity_id) == 0:
            data["same_entity"] = np.zeros((0, 0), dtype=np.float32)
            return data

        _, same_entity = compute_pair_masks_vectorized(
            asym_id.astype(np.int32),
            entity_id.astype(np.int32),
        )
        data["same_entity"] = same_entity
        return data


class DistogramTransform(BaseTransform):
    """Compute distogram from positions.

    From AF3 Table 5: template_distogram is one-hot distance bins.
    39 bins from 3.25 to 50.75 Angstroms.

    Input keys:
        - pseudo_beta: np.ndarray (Ntokens, 3) or (Ntemplates, Ntokens, 3)
        - pseudo_beta_mask: np.ndarray (Ntokens,) or (Ntemplates, Ntokens)

    Output keys:
        - distogram: np.ndarray (Ntokens, Ntokens, 39) or (Nt, Ntok, Ntok, 39)
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        distogram_config: Optional[DistogramConfig] = None,
    ):
        super().__init__(config)
        self.distogram_config = distogram_config or DistogramConfig()

    def __call__(self, data: FeatureDict) -> FeatureDict:
        positions = data.get("pseudo_beta", np.zeros((0, 3)))
        mask = data.get("pseudo_beta_mask", np.array([]))

        if positions.ndim == 2:
            # Single structure
            positions = positions[np.newaxis, ...]
            mask = mask[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        batch_size, n_tokens = positions.shape[:2]
        distogram = np.zeros(
            (batch_size, n_tokens, n_tokens, self.distogram_config.num_bins),
            dtype=np.float32
        )

        for b in range(batch_size):
            distogram[b] = compute_distogram_fast(
                positions[b:b+1],
                mask[b:b+1],
                num_bins=self.distogram_config.num_bins,
                min_dist=self.distogram_config.min_distance,
                max_dist=self.distogram_config.max_distance,
            )[0]

        if squeeze:
            distogram = distogram[0]

        data["distogram"] = distogram
        return data


class PairFeatureTransform(BaseTransform):
    """Combined transform for all pair features.

    Convenience transform that combines RelativePosition, SameChain,
    and SameEntity transforms.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config)
        self._transforms = [
            RelativePositionTransform(config),
            SameChainMaskTransform(config),
            SameEntityMaskTransform(config),
        ]

    def __call__(self, data: FeatureDict) -> FeatureDict:
        for transform in self._transforms:
            data = transform(data)
        return data


# =============================================================================
# Per-Structure Transforms
# =============================================================================


class TokenFeatureTransform(BaseTransform):
    """Extract all token-level features for a structure.

    Combines single-token and pair transforms.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config)
        self._transforms = [
            SingleTokenFeatureTransform(config),
            PairFeatureTransform(config),
        ]

    def __call__(self, data: FeatureDict) -> FeatureDict:
        for transform in self._transforms:
            data = transform(data)
        return data


class AtomFeatureTransform(BaseTransform):
    """Extract all atom-level features for a structure.

    Combines residue atom features with chain aggregations.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config)
        self._transforms = [
            ResidueAtomFeatureTransform(config),
            ChainAtomFeatureTransform(config),
        ]

    def __call__(self, data: FeatureDict) -> FeatureDict:
        for transform in self._transforms:
            data = transform(data)
        return data


class TemplateFeatureTransform(BaseTransform):
    """Extract template features.

    From AF3 Table 5: template_restype, template_pseudo_beta,
    template_distogram, template_backbone_frame, template_unit_vector.

    Input keys:
        - templates: List of template data dictionaries
        - num_tokens: int

    Output keys:
        - template_restype: np.ndarray (Nt, Ntok) int32
        - template_pseudo_beta: np.ndarray (Nt, Ntok, 3) float32
        - template_pseudo_beta_mask: np.ndarray (Nt, Ntok) float32
        - template_distogram: np.ndarray (Nt, Ntok, Ntok, 39) float32
        - template_unit_vector: np.ndarray (Nt, Ntok, Ntok, 3) float32
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        template_config: Optional[TemplateConfig] = None,
    ):
        super().__init__(config)
        self.template_config = template_config or TemplateConfig()

    def __call__(self, data: FeatureDict) -> FeatureDict:
        templates = data.get("templates", [])
        num_tokens = data.get("num_tokens", 0)

        n_templates = min(len(templates), self.template_config.max_templates)

        if n_templates == 0 or num_tokens == 0:
            # Return empty arrays
            data["template_restype"] = np.zeros((0, num_tokens), dtype=np.int32)
            data["template_pseudo_beta"] = np.zeros((0, num_tokens, 3), dtype=np.float32)
            data["template_pseudo_beta_mask"] = np.zeros((0, num_tokens), dtype=np.float32)
            data["template_distogram"] = np.zeros(
                (0, num_tokens, num_tokens, self.template_config.distogram_config.num_bins),
                dtype=np.float32
            )
            data["template_unit_vector"] = np.zeros(
                (0, num_tokens, num_tokens, 3), dtype=np.float32
            )
            return data

        # Allocate arrays
        template_restype = np.zeros((n_templates, num_tokens), dtype=np.int32)
        template_pseudo_beta = np.zeros((n_templates, num_tokens, 3), dtype=np.float32)
        template_pseudo_beta_mask = np.zeros((n_templates, num_tokens), dtype=np.float32)

        # Process each template
        for t_idx, template in enumerate(templates[:n_templates]):
            if "restype" in template:
                template_restype[t_idx] = template["restype"][:num_tokens]
            if "pseudo_beta" in template:
                template_pseudo_beta[t_idx] = template["pseudo_beta"][:num_tokens]
            if "pseudo_beta_mask" in template:
                template_pseudo_beta_mask[t_idx] = template["pseudo_beta_mask"][:num_tokens]

        # Compute distogram using vectorized function
        template_distogram = compute_distogram_fast(
            template_pseudo_beta,
            template_pseudo_beta_mask,
            num_bins=self.template_config.distogram_config.num_bins,
            min_dist=self.template_config.distogram_config.min_distance,
            max_dist=self.template_config.distogram_config.max_distance,
        )

        # Compute unit vectors
        template_unit_vector = compute_unit_vectors_vectorized(
            template_pseudo_beta,
            template_pseudo_beta_mask,
        )

        data["template_restype"] = template_restype
        data["template_pseudo_beta"] = template_pseudo_beta
        data["template_pseudo_beta_mask"] = template_pseudo_beta_mask
        data["template_distogram"] = template_distogram
        data["template_unit_vector"] = template_unit_vector
        return data


class FeatureExtractionTransform(BaseTransform):
    """Complete feature extraction pipeline.

    Combines all transforms for full AF3 Table 5 feature extraction.
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        msa_config: Optional[MSAConfig] = None,
        template_config: Optional[TemplateConfig] = None,
    ):
        super().__init__(config)
        self.msa_config = msa_config or MSAConfig()
        self.template_config = template_config or TemplateConfig()

        self._transforms = [
            # Token features
            SingleTokenFeatureTransform(config),
            # Atom features
            ResidueAtomFeatureTransform(config),
            # Pseudo-beta
            PseudoBetaTransform(config),
            BackboneCompleteTransform(config),
            # Chain aggregations
            ChainTokenFeatureTransform(config),
            ChainAtomFeatureTransform(config),
            # Pair features
            PairFeatureTransform(config),
            # MSA features
            MSARowTransform(config, self.msa_config),
            # Template features
            TemplateFeatureTransform(config, self.template_config),
        ]

    def __call__(self, data: FeatureDict) -> FeatureDict:
        for transform in self._transforms:
            data = transform(data)

        # Compute MSA profile if MSA is present
        if "msa" in data:
            msa_profile = compute_msa_profile_vectorized(
                data["msa"],
                num_classes=self.config.num_residue_types,
            )
            data["msa_profile"] = msa_profile

            # Deletion features
            if "msa_deletion_value" in data:
                data["deletion_mean"] = data["msa_deletion_value"].mean(axis=0)
                data["has_deletion"] = (data["msa_deletion_value"] > 0).any(axis=0).astype(np.float32)

        return data


# =============================================================================
# Utility Functions
# =============================================================================


def compute_distogram_fast(
    positions: np.ndarray,
    mask: np.ndarray,
    num_bins: int = 39,
    min_dist: float = 3.25,
    max_dist: float = 50.75,
) -> np.ndarray:
    """Fast distogram computation wrapper.

    Re-exported from optimized module for convenience.
    """
    from novadb.processing.optimized import compute_distogram_fast as _compute
    return _compute(positions, mask, num_bins, min_dist, max_dist)


def compute_relative_positions_fast(
    residue_indices: np.ndarray,
    chain_ids: np.ndarray,
    max_relative_idx: int = 32,
) -> np.ndarray:
    """Fast relative position computation wrapper.

    Re-exported from optimized module for convenience.
    """
    return compute_relative_position_vectorized(
        residue_indices, chain_ids, max_relative_idx
    )


def extract_features_fast(
    residue_names: Sequence[str],
    token_types: Sequence[str],
    residue_indices: Sequence[int],
    chain_ids: Sequence[str],
    entity_ids: Sequence[int],
    atoms: Sequence[Dict[str, Any]],
    msa_sequences: Optional[Sequence[str]] = None,
    templates: Optional[Sequence[Dict[str, Any]]] = None,
    config: Optional[FeatureConfig] = None,
) -> FeatureDict:
    """Convenience function for fast feature extraction.

    Args:
        residue_names: Residue names per token.
        token_types: Token types ("protein", "rna", "dna", "ligand").
        residue_indices: Residue indices per token.
        chain_ids: Chain IDs per token.
        entity_ids: Entity IDs per token.
        atoms: Atom dictionaries per token.
        msa_sequences: Optional MSA sequences.
        templates: Optional template data.
        config: Optional feature configuration.

    Returns:
        Feature dictionary with all extracted features.
    """
    data: FeatureDict = {
        "residue_names": list(residue_names),
        "token_types": list(token_types),
        "residue_indices": list(residue_indices),
        "chain_ids": list(chain_ids),
        "entity_ids": list(entity_ids),
        "atoms": list(atoms),
        "num_tokens": len(residue_names),
    }

    if msa_sequences:
        data["msa_sequences"] = list(msa_sequences)

    if templates:
        data["templates"] = list(templates)

    transform = FeatureExtractionTransform(config)
    return transform(data)


# =============================================================================
# Pipeline Factory Functions
# =============================================================================


def create_token_pipeline(config: Optional[FeatureConfig] = None) -> Pipeline:
    """Create pipeline for token-only features."""
    return Pipeline(
        [
            SingleTokenFeatureTransform(config),
            PairFeatureTransform(config),
        ],
        name="TokenPipeline",
    )


def create_atom_pipeline(config: Optional[FeatureConfig] = None) -> Pipeline:
    """Create pipeline for atom-level features."""
    return Pipeline(
        [
            ResidueAtomFeatureTransform(config),
            PseudoBetaTransform(config),
            BackboneCompleteTransform(config),
            ChainAtomFeatureTransform(config),
        ],
        name="AtomPipeline",
    )


def create_msa_pipeline(
    config: Optional[FeatureConfig] = None,
    msa_config: Optional[MSAConfig] = None,
) -> Pipeline:
    """Create pipeline for MSA features."""
    return Pipeline(
        [MSARowTransform(config, msa_config)],
        name="MSAPipeline",
    )


def create_template_pipeline(
    config: Optional[FeatureConfig] = None,
    template_config: Optional[TemplateConfig] = None,
) -> Pipeline:
    """Create pipeline for template features."""
    return Pipeline(
        [TemplateFeatureTransform(config, template_config)],
        name="TemplatePipeline",
    )


def create_full_pipeline(
    config: Optional[FeatureConfig] = None,
    msa_config: Optional[MSAConfig] = None,
    template_config: Optional[TemplateConfig] = None,
) -> Pipeline:
    """Create complete feature extraction pipeline."""
    return Pipeline(
        [FeatureExtractionTransform(config, msa_config, template_config)],
        name="FullFeaturePipeline",
    )
