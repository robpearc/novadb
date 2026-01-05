"""Reference conformer and advanced feature extraction for AlphaFold3.

Implements additional features from AF3 Supplement Table 5:
- Reference conformer features (ref_pos, ref_mask, ref_element, etc.)
- MSA profile features (msa_profile, deletion_mean)
- Template frame features (template_backbone_frame)
- Bond features (token_bonds, polymer-ligand bonds)
- Atom-to-token mapping with proper handling

Performance optimizations:
- Vectorized distogram computation using np.digitize
- Vectorized unit vector computation
- Optimized pair feature computation

Reference: AF3 Supplement Section 2.8 and Table 5
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from novadb.processing.optimized import (
    compute_distogram_fast,
    compute_unit_vectors_vectorized,
    compute_relative_position_vectorized,
    compute_pair_masks_vectorized,
)

logger = logging.getLogger(__name__)

# Constants from AF3
NUM_RESIDUE_TYPES = 32
NUM_ELEMENT_TYPES = 128
NUM_ATOM_NAME_CHARS = 64
ATOM_NAME_LENGTH = 4
NUM_DISTANCE_BINS = 39
DISTANCE_BIN_MIN = 3.25
DISTANCE_BIN_MAX = 50.75


@dataclass(frozen=True)
class ReferenceConformerConfig:
    """Configuration for reference conformer feature extraction.
    
    From AF3 Table 5: Reference conformers provide idealized
    geometry for each residue type.
    
    Attributes:
        ccd_dir: Directory containing CCD conformer files.
        use_ideal_coords: Whether to use ideal (vs model) coordinates.
        include_hydrogens: Whether to include hydrogen atoms.
        max_atoms_per_token: Maximum atoms per token.
    """
    
    ccd_dir: Optional[Path] = None
    use_ideal_coords: bool = True
    include_hydrogens: bool = False
    max_atoms_per_token: int = 37


@dataclass(frozen=True)
class MSAProfileConfig:
    """Configuration for MSA profile computation.
    
    From AF3 Table 5: MSA profile is a 32-dimensional distribution
    over residue types at each position.
    
    Attributes:
        num_classes: Number of residue type classes.
        pseudocount: Pseudocount for smoothing.
        use_deletion_mean: Whether to compute deletion mean.
    """
    
    num_classes: int = NUM_RESIDUE_TYPES
    pseudocount: float = 1e-8
    use_deletion_mean: bool = True


@dataclass(frozen=True)
class BondFeatureConfig:
    """Configuration for bond feature extraction.
    
    From AF3 Table 5: Token bonds encode connectivity between
    polymer residues and with ligands.
    
    Attributes:
        include_polymer_bonds: Include bonds between polymer residues.
        include_ligand_bonds: Include bonds to/from ligands.
        include_disulfide_bonds: Include cysteine disulfide bonds.
        bond_distance_threshold: Maximum distance for bond detection.
    """
    
    include_polymer_bonds: bool = True
    include_ligand_bonds: bool = True
    include_disulfide_bonds: bool = True
    bond_distance_threshold: float = 2.5


@dataclass
class ReferenceConformerFeatures:
    """Reference conformer features for a structure.
    
    From AF3 Table 5: These features provide the idealized
    geometry from CCD for each residue.
    
    Attributes:
        ref_pos: Reference atom positions (Natoms, 3).
        ref_mask: Reference position validity mask (Natoms,).
        ref_element: One-hot element encoding (Natoms, 128).
        ref_charge: Formal atomic charge (Natoms,).
        ref_atom_name_chars: 4-char atom names (Natoms, 4, 64).
        ref_space_uid: Unique residue identifier (Natoms,).
        atom_to_token: Mapping from atoms to tokens (Natoms,).
    """
    
    ref_pos: np.ndarray
    ref_mask: np.ndarray
    ref_element: np.ndarray
    ref_charge: np.ndarray
    ref_atom_name_chars: np.ndarray
    ref_space_uid: np.ndarray
    atom_to_token: np.ndarray


@dataclass
class MSAProfileFeatures:
    """MSA profile and deletion features.
    
    From AF3 Table 5: Profile is the distribution over residue
    types in the MSA at each position.
    
    Attributes:
        msa_profile: Residue type distribution (Ntokens, 32).
        deletion_mean: Mean deletion count per position (Ntokens,).
        has_deletion: Binary deletion indicator per position (Ntokens,).
        deletion_value: Transformed deletion value (Ntokens,).
    """
    
    msa_profile: np.ndarray
    deletion_mean: np.ndarray
    has_deletion: np.ndarray
    deletion_value: np.ndarray


@dataclass
class TemplateFrameFeatures:
    """Template backbone frame features.
    
    From AF3 Table 5: Template frames define local coordinate
    systems for each residue.
    
    Attributes:
        template_backbone_frame: Backbone frames (Ntemplate, Ntokens, 4, 4).
        template_backbone_frame_mask: Frame validity (Ntemplate, Ntokens).
        template_distogram: Pairwise distance bins (Ntemplate, Ntokens, Ntokens, Nbins).
        template_unit_vector: Inter-residue vectors (Ntemplate, Ntokens, Ntokens, 3).
    """
    
    template_backbone_frame: np.ndarray
    template_backbone_frame_mask: np.ndarray
    template_distogram: np.ndarray
    template_unit_vector: Optional[np.ndarray] = None


@dataclass
class BondFeatures:
    """Bond connectivity features.
    
    From AF3 Table 5: Token bonds encode which tokens are
    covalently bonded.
    
    Attributes:
        token_bonds: Sparse bond matrix (Ntokens, Ntokens).
        bond_types: Bond type encoding (Ntokens, Ntokens).
        polymer_ligand_bonds: Bonds between polymer and ligand (Nbonds, 2).
    """
    
    token_bonds: np.ndarray
    bond_types: Optional[np.ndarray] = None
    polymer_ligand_bonds: Optional[np.ndarray] = None


# Element encoding (periodic table order, 0-indexed)
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
ELEMENT_TO_INDEX = {e: i for i, e in enumerate(ELEMENTS)}

# Atom name character encoding
ATOM_NAME_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' *+-_"
CHAR_TO_INDEX = {c: i for i, c in enumerate(ATOM_NAME_ALPHABET)}


@dataclass
class ReferenceConformerExtractor:
    """Extracts reference conformer features from structures.
    
    From AF3 Table 5: Reference conformers provide idealized
    atomic coordinates from the Chemical Component Dictionary (CCD).
    
    Attributes:
        config: Extraction configuration.
        _ccd_cache: Cached CCD conformer data.
    """
    
    config: ReferenceConformerConfig = field(default_factory=ReferenceConformerConfig)
    _ccd_cache: Dict[str, Dict] = field(default_factory=dict, repr=False)
    
    def extract(
        self,
        residue_names: Sequence[str],
        atom_coords: Sequence[np.ndarray],
        atom_elements: Sequence[Sequence[str]],
        atom_names: Sequence[Sequence[str]],
        atom_charges: Optional[Sequence[Sequence[float]]] = None,
    ) -> ReferenceConformerFeatures:
        """Extract reference conformer features.
        
        Args:
            residue_names: Residue name for each token.
            atom_coords: Atom coordinates for each token.
            atom_elements: Element symbols for each token's atoms.
            atom_names: Atom names for each token's atoms.
            atom_charges: Optional formal charges for atoms.
            
        Returns:
            ReferenceConformerFeatures with all ref_* features.
        """
        num_tokens = len(residue_names)
        
        # Count total atoms
        total_atoms = sum(len(coords) for coords in atom_coords)
        
        # Initialize arrays
        ref_pos = np.zeros((total_atoms, 3), dtype=np.float32)
        ref_mask = np.zeros(total_atoms, dtype=np.float32)
        ref_element = np.zeros((total_atoms, NUM_ELEMENT_TYPES), dtype=np.int32)
        ref_charge = np.zeros(total_atoms, dtype=np.float32)
        ref_atom_name_chars = np.zeros(
            (total_atoms, ATOM_NAME_LENGTH, NUM_ATOM_NAME_CHARS),
            dtype=np.int32,
        )
        ref_space_uid = np.zeros(total_atoms, dtype=np.int32)
        atom_to_token = np.zeros(total_atoms, dtype=np.int32)
        
        atom_idx = 0
        
        for token_idx in range(num_tokens):
            coords = atom_coords[token_idx]
            elements = atom_elements[token_idx]
            names = atom_names[token_idx]
            charges = atom_charges[token_idx] if atom_charges else None
            
            num_atoms = len(coords)
            
            for i in range(num_atoms):
                # Position
                ref_pos[atom_idx] = coords[i]
                ref_mask[atom_idx] = 1.0
                
                # Element (one-hot)
                elem = elements[i] if i < len(elements) else "C"
                elem_idx = ELEMENT_TO_INDEX.get(elem.upper(), 0)
                ref_element[atom_idx, elem_idx] = 1
                
                # Charge
                if charges is not None and i < len(charges):
                    ref_charge[atom_idx] = charges[i]
                
                # Atom name (4 chars, each one-hot encoded)
                name = names[i] if i < len(names) else "UNK"
                self._encode_atom_name(name, ref_atom_name_chars[atom_idx])
                
                # Space UID (unique per residue)
                ref_space_uid[atom_idx] = token_idx
                
                # Atom-to-token mapping
                atom_to_token[atom_idx] = token_idx
                
                atom_idx += 1
        
        return ReferenceConformerFeatures(
            ref_pos=ref_pos,
            ref_mask=ref_mask,
            ref_element=ref_element,
            ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars,
            ref_space_uid=ref_space_uid,
            atom_to_token=atom_to_token,
        )
    
    def extract_from_ccd(
        self,
        residue_name: str,
    ) -> Optional[Dict]:
        """Extract reference conformer from CCD.
        
        Args:
            residue_name: 3-letter residue code.
            
        Returns:
            Dictionary with ideal coordinates and atom info.
        """
        if residue_name in self._ccd_cache:
            return self._ccd_cache[residue_name]
        
        if self.config.ccd_dir is None:
            return None
        
        ccd_file = self.config.ccd_dir / f"{residue_name}.cif"
        if not ccd_file.exists():
            return None
        
        try:
            conformer = self._parse_ccd_file(ccd_file)
            self._ccd_cache[residue_name] = conformer
            return conformer
        except Exception as e:
            logger.warning("Failed to parse CCD file for %s: %s", residue_name, e)
            return None
    
    def _encode_atom_name(
        self,
        name: str,
        output: np.ndarray,
    ) -> None:
        """Encode atom name to 4x64 one-hot array.
        
        Args:
            name: Atom name (up to 4 characters).
            output: Output array of shape (4, 64).
        """
        padded = name.ljust(ATOM_NAME_LENGTH)[:ATOM_NAME_LENGTH].upper()
        for i, char in enumerate(padded):
            char_idx = CHAR_TO_INDEX.get(char, 0)
            output[i, char_idx] = 1
    
    def _parse_ccd_file(self, ccd_file: Path) -> Dict:
        """Parse CCD conformer file.
        
        Args:
            ccd_file: Path to CCD file.
            
        Returns:
            Dictionary with atom coordinates and properties.
        """
        atoms = []
        coords_ideal = []
        coords_model = []
        elements = []
        charges = []
        
        with open(ccd_file) as f:
            in_atom_block = False
            
            for line in f:
                if "_chem_comp_atom." in line:
                    in_atom_block = True
                    continue
                
                if in_atom_block and line.startswith("#"):
                    break
                
                if in_atom_block and not line.startswith("_"):
                    parts = line.split()
                    if len(parts) >= 10:
                        atom_name = parts[1]
                        element = parts[2]
                        charge = float(parts[3]) if parts[3] not in (".", "?") else 0.0
                        
                        # Ideal coordinates
                        x_ideal = float(parts[4]) if parts[4] not in (".", "?") else 0.0
                        y_ideal = float(parts[5]) if parts[5] not in (".", "?") else 0.0
                        z_ideal = float(parts[6]) if parts[6] not in (".", "?") else 0.0
                        
                        atoms.append(atom_name)
                        elements.append(element)
                        charges.append(charge)
                        coords_ideal.append([x_ideal, y_ideal, z_ideal])
        
        return {
            "atoms": atoms,
            "elements": elements,
            "charges": charges,
            "coords_ideal": np.array(coords_ideal, dtype=np.float32) if coords_ideal else np.zeros((0, 3)),
        }


@dataclass
class MSAProfileExtractor:
    """Extracts MSA profile and deletion features.
    
    From AF3 Table 5: MSA profile encodes the residue type
    distribution at each position across the MSA.
    
    Attributes:
        config: Extraction configuration.
    """
    
    config: MSAProfileConfig = field(default_factory=MSAProfileConfig)
    
    # Residue type mapping (same as AF3)
    AA_TO_INDEX = {
        "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
        "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
        "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
        "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
        "X": 20,  # Unknown
    }
    GAP_INDEX = 31
    
    def extract(
        self,
        msa_array: np.ndarray,
        deletion_matrix: Optional[np.ndarray] = None,
    ) -> MSAProfileFeatures:
        """Extract MSA profile and deletion features.
        
        Args:
            msa_array: MSA as integer array (Nmsa, Ntokens).
            deletion_matrix: Deletion counts (Nmsa, Ntokens).
            
        Returns:
            MSAProfileFeatures with profile and deletion info.
        """
        n_seqs, n_tokens = msa_array.shape
        n_classes = self.config.num_classes
        
        # Compute profile (residue type distribution)
        msa_profile = np.zeros((n_tokens, n_classes), dtype=np.float32)
        
        for j in range(n_tokens):
            for c in range(n_classes):
                msa_profile[j, c] = np.sum(msa_array[:, j] == c)
        
        # Normalize with pseudocount
        row_sums = msa_profile.sum(axis=1, keepdims=True) + self.config.pseudocount
        msa_profile = msa_profile / row_sums
        
        # Compute deletion features
        if deletion_matrix is not None:
            deletion_mean = deletion_matrix.mean(axis=0).astype(np.float32)
            has_deletion = (deletion_matrix > 0).any(axis=0).astype(np.float32)
            # Transform: 2/π * arctan(d/3)
            deletion_value = (2.0 / np.pi) * np.arctan(deletion_mean / 3.0)
        else:
            deletion_mean = np.zeros(n_tokens, dtype=np.float32)
            has_deletion = np.zeros(n_tokens, dtype=np.float32)
            deletion_value = np.zeros(n_tokens, dtype=np.float32)
        
        return MSAProfileFeatures(
            msa_profile=msa_profile,
            deletion_mean=deletion_mean,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
        )
    
    def extract_from_sequences(
        self,
        sequences: Sequence[str],
        deletions: Optional[Sequence[Sequence[int]]] = None,
    ) -> MSAProfileFeatures:
        """Extract profile from sequence strings.
        
        Args:
            sequences: List of aligned sequence strings.
            deletions: Optional deletion counts per sequence.
            
        Returns:
            MSAProfileFeatures.
        """
        if not sequences:
            return MSAProfileFeatures(
                msa_profile=np.zeros((0, self.config.num_classes), dtype=np.float32),
                deletion_mean=np.zeros(0, dtype=np.float32),
                has_deletion=np.zeros(0, dtype=np.float32),
                deletion_value=np.zeros(0, dtype=np.float32),
            )
        
        n_seqs = len(sequences)
        n_tokens = len(sequences[0])
        
        # Convert sequences to integer array
        msa_array = np.zeros((n_seqs, n_tokens), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                msa_array[i, j] = self._char_to_index(char)
        
        # Convert deletions
        deletion_matrix = None
        if deletions is not None:
            deletion_matrix = np.zeros((n_seqs, n_tokens), dtype=np.int32)
            for i, dels in enumerate(deletions):
                for j, d in enumerate(dels[:n_tokens]):
                    deletion_matrix[i, j] = d
        
        return self.extract(msa_array, deletion_matrix)
    
    def _char_to_index(self, char: str) -> int:
        """Convert character to residue type index."""
        char = char.upper()
        if char in self.AA_TO_INDEX:
            return self.AA_TO_INDEX[char]
        if char == "-":
            return self.GAP_INDEX
        return self.AA_TO_INDEX["X"]


@dataclass
class TemplateFrameExtractor:
    """Extracts template backbone frame features.
    
    From AF3 Table 5 and Section 4.3.2: Template frames define
    local coordinate systems at each residue.
    
    Attributes:
        num_distance_bins: Number of distance histogram bins.
        distance_min: Minimum distance for binning.
        distance_max: Maximum distance for binning.
    """
    
    num_distance_bins: int = NUM_DISTANCE_BINS
    distance_min: float = DISTANCE_BIN_MIN
    distance_max: float = DISTANCE_BIN_MAX
    
    def extract(
        self,
        template_coords: np.ndarray,
        template_mask: np.ndarray,
        num_tokens: int,
    ) -> TemplateFrameFeatures:
        """Extract template frame features.
        
        Args:
            template_coords: Template atom coordinates (Ntemplate, Ntokens, 3, 3).
                For proteins: N, CA, C positions.
                For nucleotides: C1', C3', C4' positions.
            template_mask: Template validity mask (Ntemplate, Ntokens).
            num_tokens: Number of tokens in query.
            
        Returns:
            TemplateFrameFeatures.
        """
        n_templates = template_coords.shape[0]
        
        # Compute backbone frames
        backbone_frame = np.zeros(
            (n_templates, num_tokens, 4, 4),
            dtype=np.float32,
        )
        backbone_frame_mask = np.zeros(
            (n_templates, num_tokens),
            dtype=np.float32,
        )
        
        for t in range(n_templates):
            for i in range(min(num_tokens, template_coords.shape[1])):
                if template_mask[t, i] > 0:
                    frame = self._compute_frame(template_coords[t, i])
                    if frame is not None:
                        backbone_frame[t, i] = frame
                        backbone_frame_mask[t, i] = 1.0
        
        # Compute distogram
        distogram = self._compute_distogram(
            template_coords[:, :, 1, :],  # Use CA/C1' positions
            template_mask,
            num_tokens,
        )
        
        # Compute unit vectors
        unit_vector = self._compute_unit_vectors(
            template_coords[:, :, 1, :],
            template_mask,
            num_tokens,
        )
        
        return TemplateFrameFeatures(
            template_backbone_frame=backbone_frame,
            template_backbone_frame_mask=backbone_frame_mask,
            template_distogram=distogram,
            template_unit_vector=unit_vector,
        )
    
    def _compute_frame(
        self,
        coords: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute 4x4 transformation matrix from backbone atoms.
        
        Args:
            coords: Backbone atom coordinates (3, 3) for N, CA, C.
            
        Returns:
            4x4 transformation matrix or None if invalid.
        """
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            return None
        
        # Extract N, CA, C
        n_pos = coords[0]
        ca_pos = coords[1]
        c_pos = coords[2]
        
        # Build local frame at CA
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
        
        # Z-axis: cross product
        z_axis = np.cross(x_axis, y_axis)
        
        # Build 4x4 matrix
        frame = np.eye(4, dtype=np.float32)
        frame[:3, 0] = x_axis
        frame[:3, 1] = y_axis
        frame[:3, 2] = z_axis
        frame[:3, 3] = ca_pos
        
        return frame
    
    def _compute_distogram(
        self,
        positions: np.ndarray,
        mask: np.ndarray,
        num_tokens: int,
    ) -> np.ndarray:
        """Compute pairwise distance histogram.

        Uses vectorized np.digitize for O(n²) instead of O(n² * bins).

        Args:
            positions: Representative positions (Ntemplate, Ntokens, 3).
            mask: Validity mask (Ntemplate, Ntokens).
            num_tokens: Number of tokens.

        Returns:
            Distogram (Ntemplate, Ntokens, Ntokens, Nbins).
        """
        n_templates = positions.shape[0]
        n_bins = self.num_distance_bins
        n_pos = min(num_tokens, positions.shape[1])

        # Use optimized vectorized distogram computation
        distogram = compute_distogram_fast(
            positions[:, :n_pos, :],
            mask[:, :n_pos],
            num_bins=n_bins,
            min_dist=self.distance_min,
            max_dist=self.distance_max,
        )

        # Pad to num_tokens if needed
        if n_pos < num_tokens:
            padded = np.zeros(
                (n_templates, num_tokens, num_tokens, n_bins),
                dtype=np.float32,
            )
            padded[:, :n_pos, :n_pos, :] = distogram
            distogram = padded

        return distogram
    
    def _compute_unit_vectors(
        self,
        positions: np.ndarray,
        mask: np.ndarray,
        num_tokens: int,
    ) -> np.ndarray:
        """Compute unit vectors between residue pairs.

        Uses fully vectorized computation for all templates.

        Args:
            positions: Representative positions (Ntemplate, Ntokens, 3).
            mask: Validity mask (Ntemplate, Ntokens).
            num_tokens: Number of tokens.

        Returns:
            Unit vectors (Ntemplate, Ntokens, Ntokens, 3).
        """
        n_templates = positions.shape[0]
        n_pos = min(num_tokens, positions.shape[1])

        # Use optimized vectorized unit vector computation
        unit_vectors = compute_unit_vectors_vectorized(
            positions[:, :n_pos, :],
            mask[:, :n_pos],
        )

        # Pad to num_tokens if needed
        if n_pos < num_tokens:
            padded = np.zeros(
                (n_templates, num_tokens, num_tokens, 3),
                dtype=np.float32,
            )
            padded[:, :n_pos, :n_pos, :] = unit_vectors
            unit_vectors = padded

        return unit_vectors


@dataclass
class BondFeatureExtractor:
    """Extracts bond connectivity features.
    
    From AF3 Table 5: Token bonds encode covalent connections
    between tokens (residues/atoms).
    
    Attributes:
        config: Bond feature configuration.
    """
    
    config: BondFeatureConfig = field(default_factory=BondFeatureConfig)
    
    # Bond type encoding
    BOND_TYPE_NONE = 0
    BOND_TYPE_POLYMER = 1
    BOND_TYPE_LIGAND = 2
    BOND_TYPE_DISULFIDE = 3
    
    def extract(
        self,
        chain_ids: Sequence[str],
        residue_indices: Sequence[int],
        residue_names: Sequence[str],
        atom_coords: Optional[Sequence[np.ndarray]] = None,
    ) -> BondFeatures:
        """Extract bond features.
        
        Args:
            chain_ids: Chain ID for each token.
            residue_indices: Residue index for each token.
            residue_names: Residue name for each token.
            atom_coords: Optional atom coordinates for distance-based detection.
            
        Returns:
            BondFeatures with connectivity information.
        """
        num_tokens = len(chain_ids)
        
        token_bonds = np.zeros((num_tokens, num_tokens), dtype=np.int32)
        bond_types = np.zeros((num_tokens, num_tokens), dtype=np.int32)
        
        # Detect polymer bonds (sequential residues in same chain)
        if self.config.include_polymer_bonds:
            self._add_polymer_bonds(
                token_bonds,
                bond_types,
                chain_ids,
                residue_indices,
                residue_names,
            )
        
        # Detect disulfide bonds
        if self.config.include_disulfide_bonds and atom_coords is not None:
            self._add_disulfide_bonds(
                token_bonds,
                bond_types,
                residue_names,
                atom_coords,
            )
        
        # Detect ligand bonds
        polymer_ligand_bonds = None
        if self.config.include_ligand_bonds and atom_coords is not None:
            polymer_ligand_bonds = self._detect_ligand_bonds(
                residue_names,
                atom_coords,
            )
        
        return BondFeatures(
            token_bonds=token_bonds,
            bond_types=bond_types,
            polymer_ligand_bonds=polymer_ligand_bonds,
        )
    
    def _add_polymer_bonds(
        self,
        token_bonds: np.ndarray,
        bond_types: np.ndarray,
        chain_ids: Sequence[str],
        residue_indices: Sequence[int],
        residue_names: Sequence[str],
    ) -> None:
        """Add polymer backbone bonds.
        
        Args:
            token_bonds: Bond matrix to update.
            bond_types: Bond type matrix to update.
            chain_ids: Chain ID for each token.
            residue_indices: Residue index for each token.
            residue_names: Residue name for each token.
        """
        num_tokens = len(chain_ids)
        
        # Standard amino acids and nucleotides form polymer bonds
        polymer_residues = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
            "A", "G", "C", "U", "DA", "DG", "DC", "DT",
        }
        
        for i in range(num_tokens - 1):
            j = i + 1
            
            # Check if same chain and sequential
            if chain_ids[i] != chain_ids[j]:
                continue
            
            if residue_indices[j] - residue_indices[i] != 1:
                continue
            
            # Check if both are polymer residues
            if residue_names[i] not in polymer_residues:
                continue
            if residue_names[j] not in polymer_residues:
                continue
            
            # Add bond
            token_bonds[i, j] = 1
            token_bonds[j, i] = 1
            bond_types[i, j] = self.BOND_TYPE_POLYMER
            bond_types[j, i] = self.BOND_TYPE_POLYMER
    
    def _add_disulfide_bonds(
        self,
        token_bonds: np.ndarray,
        bond_types: np.ndarray,
        residue_names: Sequence[str],
        atom_coords: Sequence[np.ndarray],
    ) -> None:
        """Add cysteine disulfide bonds.
        
        Args:
            token_bonds: Bond matrix to update.
            bond_types: Bond type matrix to update.
            residue_names: Residue name for each token.
            atom_coords: Atom coordinates for each token.
        """
        # Find cysteine residues
        cys_indices = [
            i for i, name in enumerate(residue_names)
            if name == "CYS"
        ]
        
        # Check SG-SG distances
        for i, idx_i in enumerate(cys_indices):
            for idx_j in cys_indices[i + 1:]:
                sg_i = self._get_sg_position(atom_coords[idx_i])
                sg_j = self._get_sg_position(atom_coords[idx_j])
                
                if sg_i is None or sg_j is None:
                    continue
                
                distance = np.linalg.norm(sg_i - sg_j)
                
                # Typical disulfide S-S distance is ~2.05 Å
                if distance < 2.5:
                    token_bonds[idx_i, idx_j] = 1
                    token_bonds[idx_j, idx_i] = 1
                    bond_types[idx_i, idx_j] = self.BOND_TYPE_DISULFIDE
                    bond_types[idx_j, idx_i] = self.BOND_TYPE_DISULFIDE
    
    def _get_sg_position(
        self,
        coords: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Get SG atom position from cysteine coordinates.
        
        Args:
            coords: Atom coordinates for residue.
            
        Returns:
            SG position or None.
        """
        # Simplified: assume SG is at index 5 (after N, CA, C, O, CB)
        if len(coords) > 5:
            return coords[5]
        return None
    
    def _detect_ligand_bonds(
        self,
        residue_names: Sequence[str],
        atom_coords: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Detect bonds between polymer and ligand atoms.
        
        Args:
            residue_names: Residue name for each token.
            atom_coords: Atom coordinates for each token.
            
        Returns:
            Array of (polymer_token, ligand_token) bond pairs.
        """
        polymer_residues = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
            "A", "G", "C", "U", "DA", "DG", "DC", "DT",
        }
        
        # Identify polymer and ligand tokens
        polymer_tokens = [
            i for i, name in enumerate(residue_names)
            if name in polymer_residues
        ]
        ligand_tokens = [
            i for i, name in enumerate(residue_names)
            if name not in polymer_residues
        ]
        
        bonds = []
        threshold = self.config.bond_distance_threshold
        
        for p_idx in polymer_tokens:
            for l_idx in ligand_tokens:
                # Check minimum distance between any atoms
                min_dist = float('inf')
                
                for p_coord in atom_coords[p_idx]:
                    for l_coord in atom_coords[l_idx]:
                        dist = np.linalg.norm(p_coord - l_coord)
                        min_dist = min(min_dist, dist)
                
                if min_dist < threshold:
                    bonds.append([p_idx, l_idx])
        
        return np.array(bonds, dtype=np.int32) if bonds else np.zeros((0, 2), dtype=np.int32)


@dataclass(frozen=True)
class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline.
    
    Combines all feature extractors for AF3 Table 5 features.
    
    Attributes:
        ref_conformer_extractor: Reference conformer extractor.
        msa_profile_extractor: MSA profile extractor.
        template_frame_extractor: Template frame extractor.
        bond_extractor: Bond feature extractor.
    """
    
    ref_conformer_extractor: ReferenceConformerExtractor = field(
        default_factory=ReferenceConformerExtractor
    )
    msa_profile_extractor: MSAProfileExtractor = field(
        default_factory=MSAProfileExtractor
    )
    template_frame_extractor: TemplateFrameExtractor = field(
        default_factory=TemplateFrameExtractor
    )
    bond_extractor: BondFeatureExtractor = field(
        default_factory=BondFeatureExtractor
    )
    
    def extract_all_features(
        self,
        residue_names: Sequence[str],
        chain_ids: Sequence[str],
        residue_indices: Sequence[int],
        atom_coords: Sequence[np.ndarray],
        atom_elements: Sequence[Sequence[str]],
        atom_names: Sequence[Sequence[str]],
        msa_sequences: Optional[Sequence[str]] = None,
        msa_deletions: Optional[Sequence[Sequence[int]]] = None,
        template_coords: Optional[np.ndarray] = None,
        template_mask: Optional[np.ndarray] = None,
        resolution: Optional[float] = None,
        is_distillation: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Extract all AF3 Table 5 features.

        Args:
            residue_names: Residue name for each token.
            chain_ids: Chain ID for each token.
            residue_indices: Residue index for each token.
            atom_coords: Atom coordinates for each token.
            atom_elements: Element symbols for each token's atoms.
            atom_names: Atom names for each token's atoms.
            msa_sequences: Optional MSA sequences.
            msa_deletions: Optional MSA deletion counts.
            template_coords: Optional template backbone coords.
            template_mask: Optional template validity mask.
            resolution: Optional experimental resolution in Angstroms.
            is_distillation: Whether data comes from distillation.

        Returns:
            Dictionary of all extracted features.
        """
        features = {}
        num_tokens = len(residue_names)

        # Reference conformer features
        ref_features = self.ref_conformer_extractor.extract(
            residue_names,
            atom_coords,
            atom_elements,
            atom_names,
        )
        features["ref_pos"] = ref_features.ref_pos
        features["ref_mask"] = ref_features.ref_mask
        features["ref_element"] = ref_features.ref_element
        features["ref_charge"] = ref_features.ref_charge
        features["ref_atom_name_chars"] = ref_features.ref_atom_name_chars
        features["ref_space_uid"] = ref_features.ref_space_uid
        features["atom_to_token"] = ref_features.atom_to_token

        # Extract atom existence and ambiguity features
        atom_exists_features = self._extract_atom_exists_features(
            residue_names, atom_coords, atom_names
        )
        features["atom_exists"] = atom_exists_features["atom_exists"]
        features["atom_is_ambiguous"] = atom_exists_features["atom_is_ambiguous"]

        # Extract frame atom mask
        features["frame_atom_mask"] = self._extract_frame_atom_mask(
            residue_names, atom_names
        )

        # Extract target features (one-hot residue types)
        features["target_feat"] = self._extract_target_feat(residue_names)

        # Extract pseudo-beta features
        pseudo_beta_features = self._extract_pseudo_beta_features(
            residue_names, atom_coords, atom_names
        )
        features["pseudo_beta"] = pseudo_beta_features["pseudo_beta"]
        features["pseudo_beta_mask"] = pseudo_beta_features["pseudo_beta_mask"]

        # Extract backbone rigid features
        rigid_features = self._extract_backbone_rigid_features(
            residue_names, atom_coords, atom_names
        )
        features["backbone_rigid_tensor"] = rigid_features["backbone_rigid_tensor"]
        features["backbone_rigid_mask"] = rigid_features["backbone_rigid_mask"]

        # Extract pair features
        pair_features = self._extract_pair_features(
            chain_ids, residue_indices, num_tokens
        )
        features["relative_position"] = pair_features["relative_position"]
        features["same_chain"] = pair_features["same_chain"]
        features["same_entity"] = pair_features["same_entity"]

        # MSA profile features
        if msa_sequences is not None:
            msa_features = self.msa_profile_extractor.extract_from_sequences(
                msa_sequences,
                msa_deletions,
            )
            features["msa_profile"] = msa_features.msa_profile
            features["deletion_mean"] = msa_features.deletion_mean
            features["has_deletion"] = msa_features.has_deletion
            features["deletion_value"] = msa_features.deletion_value

        # Template frame features
        if template_coords is not None and template_mask is not None:
            template_features = self.template_frame_extractor.extract(
                template_coords,
                template_mask,
                num_tokens,
            )
            features["template_backbone_frame"] = template_features.template_backbone_frame
            features["template_backbone_frame_mask"] = template_features.template_backbone_frame_mask
            features["template_distogram"] = template_features.template_distogram
            if template_features.template_unit_vector is not None:
                features["template_unit_vector"] = template_features.template_unit_vector

        # Bond features
        bond_features = self.bond_extractor.extract(
            chain_ids,
            residue_indices,
            residue_names,
            atom_coords,
        )
        features["token_bonds"] = bond_features.token_bonds
        if bond_features.bond_types is not None:
            features["bond_types"] = bond_features.bond_types
        if bond_features.polymer_ligand_bonds is not None:
            features["polymer_ligand_bonds"] = bond_features.polymer_ligand_bonds

        # Metadata features
        if resolution is not None:
            features["resolution"] = np.array([resolution], dtype=np.float32)
        features["is_distillation"] = np.array([is_distillation], dtype=np.bool_)

        return features

    def _extract_atom_exists_features(
        self,
        residue_names: Sequence[str],
        atom_coords: Sequence[np.ndarray],
        atom_names: Sequence[Sequence[str]],
    ) -> Dict[str, np.ndarray]:
        """Extract atom existence and ambiguity features."""
        total_atoms = sum(len(coords) for coords in atom_coords)

        atom_exists = np.zeros(total_atoms, dtype=np.float32)
        atom_is_ambiguous = np.zeros(total_atoms, dtype=np.float32)

        # Ambiguous atom sets for symmetric sidechains
        AMBIGUOUS_ATOMS = {
            "ARG": {"NH1", "NH2"},
            "ASP": {"OD1", "OD2"},
            "GLU": {"OE1", "OE2"},
            "LEU": {"CD1", "CD2"},
            "PHE": {"CD1", "CD2", "CE1", "CE2"},
            "TYR": {"CD1", "CD2", "CE1", "CE2"},
            "VAL": {"CG1", "CG2"},
        }

        atom_idx = 0
        for token_idx, coords in enumerate(atom_coords):
            res_name = residue_names[token_idx]
            names = atom_names[token_idx] if token_idx < len(atom_names) else []
            ambiguous_set = AMBIGUOUS_ATOMS.get(res_name, set())

            for i, coord in enumerate(coords):
                if not np.any(np.isnan(coord)):
                    atom_exists[atom_idx] = 1.0

                if i < len(names) and names[i] in ambiguous_set:
                    atom_is_ambiguous[atom_idx] = 1.0

                atom_idx += 1

        return {
            "atom_exists": atom_exists,
            "atom_is_ambiguous": atom_is_ambiguous,
        }

    def _extract_frame_atom_mask(
        self,
        residue_names: Sequence[str],
        atom_names: Sequence[Sequence[str]],
    ) -> np.ndarray:
        """Extract frame atom mask."""
        total_atoms = sum(len(names) for names in atom_names)
        frame_atom_mask = np.zeros(total_atoms, dtype=np.float32)

        PROTEIN_FRAME_ATOMS = {"N", "CA", "C"}
        NUCLEIC_FRAME_ATOMS = {"C1'", "C3'", "C4'"}
        PROTEIN_RESIDUES = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        }
        NUCLEIC_RESIDUES = {"A", "G", "C", "U", "DA", "DG", "DC", "DT"}

        atom_idx = 0
        for token_idx, names in enumerate(atom_names):
            res_name = residue_names[token_idx] if token_idx < len(residue_names) else ""

            if res_name in PROTEIN_RESIDUES:
                frame_atoms = PROTEIN_FRAME_ATOMS
            elif res_name in NUCLEIC_RESIDUES:
                frame_atoms = NUCLEIC_FRAME_ATOMS
            else:
                frame_atoms = set(list(names)[:3]) if names else set()

            for name in names:
                if name in frame_atoms:
                    frame_atom_mask[atom_idx] = 1.0
                atom_idx += 1

        return frame_atom_mask

    def _extract_target_feat(
        self,
        residue_names: Sequence[str],
    ) -> np.ndarray:
        """Extract target features (one-hot residue types)."""
        n_tokens = len(residue_names)
        target_feat = np.zeros((n_tokens, NUM_RESIDUE_TYPES), dtype=np.float32)

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

        for i, res_name in enumerate(residue_names):
            if res_name in AA_MAP:
                idx = AA_MAP[res_name]
            elif res_name in RNA_MAP:
                idx = RNA_MAP[res_name]
            elif res_name in DNA_MAP:
                idx = DNA_MAP[res_name]
            else:
                idx = AA_UNK_IDX
            target_feat[i, idx] = 1.0

        return target_feat

    def _extract_pseudo_beta_features(
        self,
        residue_names: Sequence[str],
        atom_coords: Sequence[np.ndarray],
        atom_names: Sequence[Sequence[str]],
    ) -> Dict[str, np.ndarray]:
        """Extract pseudo-beta features (CB or CA for GLY)."""
        n_tokens = len(residue_names)
        pseudo_beta = np.zeros((n_tokens, 3), dtype=np.float32)
        pseudo_beta_mask = np.zeros(n_tokens, dtype=np.float32)

        for i, (res_name, coords, names) in enumerate(
            zip(residue_names, atom_coords, atom_names)
        ):
            names_list = list(names) if names else []
            coords_arr = np.array(coords) if len(coords) > 0 else np.zeros((0, 3))

            # Find CB or CA
            cb_idx = None
            ca_idx = None
            for j, name in enumerate(names_list):
                if name == "CB":
                    cb_idx = j
                elif name == "CA":
                    ca_idx = j

            # Use CB for non-GLY, CA for GLY
            if res_name == "GLY":
                target_idx = ca_idx
            else:
                target_idx = cb_idx if cb_idx is not None else ca_idx

            if target_idx is not None and target_idx < len(coords_arr):
                pseudo_beta[i] = coords_arr[target_idx]
                pseudo_beta_mask[i] = 1.0

        return {
            "pseudo_beta": pseudo_beta,
            "pseudo_beta_mask": pseudo_beta_mask,
        }

    def _extract_backbone_rigid_features(
        self,
        residue_names: Sequence[str],
        atom_coords: Sequence[np.ndarray],
        atom_names: Sequence[Sequence[str]],
    ) -> Dict[str, np.ndarray]:
        """Extract backbone rigid body representation."""
        n_tokens = len(residue_names)
        backbone_rigid_tensor = np.zeros((n_tokens, 4, 4), dtype=np.float32)
        backbone_rigid_mask = np.zeros(n_tokens, dtype=np.float32)

        PROTEIN_RESIDUES = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        }
        NUCLEIC_RESIDUES = {"A", "G", "C", "U", "DA", "DG", "DC", "DT"}

        for i, (res_name, coords, names) in enumerate(
            zip(residue_names, atom_coords, atom_names)
        ):
            names_list = list(names) if names else []
            coords_arr = np.array(coords) if len(coords) > 0 else np.zeros((0, 3))

            frame = None

            if res_name in PROTEIN_RESIDUES:
                # Find N, CA, C atoms
                n_idx = ca_idx = c_idx = None
                for j, name in enumerate(names_list):
                    if name == "N":
                        n_idx = j
                    elif name == "CA":
                        ca_idx = j
                    elif name == "C":
                        c_idx = j

                if all(idx is not None and idx < len(coords_arr)
                       for idx in [n_idx, ca_idx, c_idx]):
                    frame = self._compute_frame(
                        coords_arr[n_idx],
                        coords_arr[ca_idx],
                        coords_arr[c_idx],
                    )

            elif res_name in NUCLEIC_RESIDUES:
                # Find C1', C3', C4' atoms
                c1_idx = c3_idx = c4_idx = None
                for j, name in enumerate(names_list):
                    if name == "C1'":
                        c1_idx = j
                    elif name == "C3'":
                        c3_idx = j
                    elif name == "C4'":
                        c4_idx = j

                if all(idx is not None and idx < len(coords_arr)
                       for idx in [c1_idx, c3_idx, c4_idx]):
                    frame = self._compute_frame(
                        coords_arr[c3_idx],
                        coords_arr[c4_idx],
                        coords_arr[c1_idx],
                    )

            else:
                # For ligands, use first 3 atoms if available
                if len(coords_arr) >= 3:
                    frame = self._compute_frame(
                        coords_arr[0],
                        coords_arr[1],
                        coords_arr[2],
                    )

            if frame is not None:
                backbone_rigid_tensor[i] = frame
                backbone_rigid_mask[i] = 1.0
            else:
                backbone_rigid_tensor[i] = np.eye(4, dtype=np.float32)

        return {
            "backbone_rigid_tensor": backbone_rigid_tensor,
            "backbone_rigid_mask": backbone_rigid_mask,
        }

    def _compute_frame(
        self,
        n_pos: np.ndarray,
        ca_pos: np.ndarray,
        c_pos: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute 4x4 transformation matrix from three atoms."""
        if np.any(np.isnan(n_pos)) or np.any(np.isnan(ca_pos)) or np.any(np.isnan(c_pos)):
            return None

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

        # Z-axis: cross product
        z_axis = np.cross(x_axis, y_axis)

        # Build 4x4 matrix
        frame = np.eye(4, dtype=np.float32)
        frame[:3, 0] = x_axis
        frame[:3, 1] = y_axis
        frame[:3, 2] = z_axis
        frame[:3, 3] = ca_pos

        return frame

    def _extract_pair_features(
        self,
        chain_ids: Sequence[str],
        residue_indices: Sequence[int],
        num_tokens: int,
        max_relative_idx: int = 32,
    ) -> Dict[str, np.ndarray]:
        """Extract pairwise token features."""
        relative_position = np.zeros((num_tokens, num_tokens), dtype=np.int32)
        same_chain = np.zeros((num_tokens, num_tokens), dtype=np.float32)
        same_entity = np.zeros((num_tokens, num_tokens), dtype=np.float32)

        for i in range(num_tokens):
            for j in range(num_tokens):
                if chain_ids[i] == chain_ids[j]:
                    same_chain[i, j] = 1.0
                    diff = residue_indices[j] - residue_indices[i]
                    diff = max(-max_relative_idx, min(max_relative_idx, diff))
                    relative_position[i, j] = diff + max_relative_idx
                else:
                    relative_position[i, j] = 2 * max_relative_idx + 1

                # same_entity would need entity_id mapping
                # For now, assume same chain = same entity
                if chain_ids[i] == chain_ids[j]:
                    same_entity[i, j] = 1.0

        return {
            "relative_position": relative_position,
            "same_chain": same_chain,
            "same_entity": same_entity,
        }


def create_feature_pipeline(
    *,
    ccd_dir: Optional[Path] = None,
    include_hydrogens: bool = False,
    bond_distance_threshold: float = 2.5,
) -> FeatureEngineeringPipeline:
    """Factory function to create a configured feature pipeline.
    
    Args:
        ccd_dir: Directory containing CCD conformer files.
        include_hydrogens: Whether to include hydrogen atoms.
        bond_distance_threshold: Distance threshold for bond detection.
        
    Returns:
        Configured FeatureEngineeringPipeline instance.
    """
    ref_config = ReferenceConformerConfig(
        ccd_dir=ccd_dir,
        include_hydrogens=include_hydrogens,
    )
    
    bond_config = BondFeatureConfig(
        bond_distance_threshold=bond_distance_threshold,
    )
    
    return FeatureEngineeringPipeline(
        ref_conformer_extractor=ReferenceConformerExtractor(config=ref_config),
        msa_profile_extractor=MSAProfileExtractor(),
        template_frame_extractor=TemplateFrameExtractor(),
        bond_extractor=BondFeatureExtractor(config=bond_config),
    )
