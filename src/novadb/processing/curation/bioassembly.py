"""Bioassembly handling with symmetry operations.

This module provides comprehensive support for biological assembly generation
from asymmetric unit structures, as described in the AlphaFold3 supplement.

From AF3 Supplement Section 2.1:
- Bioassemblies expand the asymmetric unit using symmetry operations
- Multiple assemblies may exist; typically assembly "1" is used
- Operators include rotation matrices and translation vectors
- Operator expressions can be combined (e.g., "(1-60)" for icosahedral symmetry)

Key concepts:
- Asymmetric unit: The smallest portion of a crystal that can generate
  the complete unit cell using symmetry operations
- Bioassembly: The biologically relevant oligomeric form of a structure
- Symmetry operation: A rotation + translation that generates copies
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from novadb.data.parsers.structure import (
    Atom,
    Bond,
    Chain,
    Residue,
    Structure,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Common symmetry types found in PDB structures
class SymmetryType(Enum):
    """Common symmetry types for bioassemblies."""
    IDENTITY = auto()        # No transformation (1x)
    C2 = auto()              # Two-fold rotational (2x)
    C3 = auto()              # Three-fold rotational (3x)
    C4 = auto()              # Four-fold rotational (4x)
    C5 = auto()              # Five-fold rotational (5x)
    C6 = auto()              # Six-fold rotational (6x)
    D2 = auto()              # Dihedral 2 (4x)
    D3 = auto()              # Dihedral 3 (6x)
    D4 = auto()              # Dihedral 4 (8x)
    D5 = auto()              # Dihedral 5 (10x)
    D6 = auto()              # Dihedral 6 (12x)
    T = auto()               # Tetrahedral (12x)
    O = auto()               # Octahedral (24x)
    I = auto()               # Icosahedral (60x)
    HELICAL = auto()         # Helical symmetry
    CRYSTAL = auto()         # Crystal contacts
    OTHER = auto()           # Other/unknown


# Tolerance for floating point comparisons
ROTATION_TOLERANCE = 1e-6
TRANSLATION_TOLERANCE = 1e-4


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SymmetryOperation:
    """A single symmetry operation (rotation + translation).
    
    The transformation is applied as: x' = R @ x + t
    where R is the 3x3 rotation matrix and t is the translation vector.
    
    Attributes:
        operator_id: Unique identifier for this operation
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
        name: Optional human-readable name (e.g., "2-fold rotation")
    """
    operator_id: str
    rotation: np.ndarray  # Shape (3, 3)
    translation: np.ndarray  # Shape (3,)
    name: Optional[str] = None
    
    def __post_init__(self):
        """Validate and convert arrays."""
        if not isinstance(self.rotation, np.ndarray):
            self.rotation = np.array(self.rotation, dtype=np.float64)
        if not isinstance(self.translation, np.ndarray):
            self.translation = np.array(self.translation, dtype=np.float64)
        
        assert self.rotation.shape == (3, 3), \
            f"Rotation must be (3, 3), got {self.rotation.shape}"
        assert self.translation.shape == (3,), \
            f"Translation must be (3,), got {self.translation.shape}"
    
    @property
    def is_identity(self) -> bool:
        """Check if this is an identity operation."""
        return (
            np.allclose(self.rotation, np.eye(3), atol=ROTATION_TOLERANCE) and
            np.allclose(self.translation, np.zeros(3), atol=TRANSLATION_TOLERANCE)
        )
    
    @property
    def is_pure_rotation(self) -> bool:
        """Check if this is a pure rotation (no translation)."""
        return np.allclose(self.translation, np.zeros(3), atol=TRANSLATION_TOLERANCE)
    
    @property
    def is_pure_translation(self) -> bool:
        """Check if this is a pure translation (no rotation)."""
        return np.allclose(self.rotation, np.eye(3), atol=ROTATION_TOLERANCE)
    
    @property
    def rotation_angle(self) -> float:
        """Calculate rotation angle in degrees."""
        trace = np.trace(self.rotation)
        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = (trace - 1) / 2
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))
    
    @property
    def rotation_axis(self) -> Optional[np.ndarray]:
        """Calculate rotation axis (eigenvector with eigenvalue 1)."""
        if self.is_identity:
            return None
        
        # Find eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.rotation)
        
        for i, ev in enumerate(eigenvalues):
            if np.isclose(ev.real, 1.0, atol=ROTATION_TOLERANCE) and np.isclose(ev.imag, 0.0):
                axis = eigenvectors[:, i].real
                return axis / np.linalg.norm(axis)
        
        return None
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Apply transformation to a single 3D point."""
        return self.rotation @ point + self.translation
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply transformation to multiple 3D points.
        
        Args:
            points: Array of shape (N, 3)
            
        Returns:
            Transformed points of shape (N, 3)
        """
        return (self.rotation @ points.T).T + self.translation
    
    def compose(self, other: "SymmetryOperation") -> "SymmetryOperation":
        """Compose this operation with another (self after other).
        
        The result R' = R_self @ R_other, t' = R_self @ t_other + t_self
        """
        new_rotation = self.rotation @ other.rotation
        new_translation = self.rotation @ other.translation + self.translation
        return SymmetryOperation(
            operator_id=f"{self.operator_id}_{other.operator_id}",
            rotation=new_rotation,
            translation=new_translation,
        )
    
    def inverse(self) -> "SymmetryOperation":
        """Compute the inverse transformation."""
        inv_rotation = self.rotation.T  # Rotation matrices are orthogonal
        inv_translation = -inv_rotation @ self.translation
        return SymmetryOperation(
            operator_id=f"{self.operator_id}_inv",
            rotation=inv_rotation,
            translation=inv_translation,
        )
    
    def to_matrix_4x4(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix
    
    @classmethod
    def from_matrix_4x4(cls, matrix: np.ndarray, operator_id: str = "1") -> "SymmetryOperation":
        """Create from 4x4 homogeneous transformation matrix."""
        return cls(
            operator_id=operator_id,
            rotation=matrix[:3, :3].copy(),
            translation=matrix[:3, 3].copy(),
        )
    
    @classmethod
    def identity(cls) -> "SymmetryOperation":
        """Create identity operation."""
        return cls(
            operator_id="1",
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            name="identity",
        )
    
    @classmethod
    def rotation_around_axis(
        cls,
        axis: np.ndarray,
        angle_degrees: float,
        operator_id: str = "1",
    ) -> "SymmetryOperation":
        """Create rotation around arbitrary axis through origin.
        
        Args:
            axis: Rotation axis (will be normalized)
            angle_degrees: Rotation angle in degrees
            operator_id: Identifier for operation
        """
        axis = np.array(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        
        angle = np.radians(angle_degrees)
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        
        x, y, z = axis
        rotation = np.array([
            [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
        ], dtype=np.float64)
        
        return cls(
            operator_id=operator_id,
            rotation=rotation,
            translation=np.zeros(3, dtype=np.float64),
            name=f"{angle_degrees}° rotation around {axis}",
        )


@dataclass
class AssemblyDefinition:
    """Definition of a biological assembly.
    
    A bioassembly specifies which chains participate and which
    symmetry operations to apply.
    
    Attributes:
        assembly_id: Unique identifier (e.g., "1", "2")
        operations: List of symmetry operations
        chain_ids: Chains to include in this assembly
        details: Description from PDB (e.g., "author_defined_assembly")
        method: How assembly was determined (author, software)
        oligomeric_count: Number of copies in assembly
    """
    assembly_id: str
    operations: List[SymmetryOperation] = field(default_factory=list)
    chain_ids: Set[str] = field(default_factory=set)
    details: Optional[str] = None
    method: Optional[str] = None
    oligomeric_count: Optional[int] = None
    
    @property
    def num_operations(self) -> int:
        """Number of symmetry operations."""
        return len(self.operations)
    
    @property
    def expected_chain_count(self) -> int:
        """Expected number of chains after expansion."""
        return len(self.chain_ids) * self.num_operations
    
    def get_symmetry_type(self) -> SymmetryType:
        """Infer symmetry type from operations."""
        n_ops = self.num_operations
        
        if n_ops == 1:
            return SymmetryType.IDENTITY
        
        # Check for common symmetry groups
        if n_ops == 2:
            return SymmetryType.C2
        elif n_ops == 3:
            return SymmetryType.C3
        elif n_ops == 4:
            # Could be C4 or D2
            return SymmetryType.C4
        elif n_ops == 5:
            return SymmetryType.C5
        elif n_ops == 6:
            # Could be C6 or D3
            return SymmetryType.C6
        elif n_ops == 12:
            # Could be D6 or T
            return SymmetryType.T
        elif n_ops == 24:
            return SymmetryType.O
        elif n_ops == 60:
            return SymmetryType.I
        
        return SymmetryType.OTHER


@dataclass
class ChainMapping:
    """Mapping of original chain to transformed chain.
    
    Tracks the relationship between asymmetric unit chains
    and their symmetry-expanded copies.
    
    Attributes:
        original_chain_id: Chain ID in asymmetric unit
        new_chain_id: Chain ID after transformation
        operator: Symmetry operation applied
        is_original: Whether this is the identity copy
    """
    original_chain_id: str
    new_chain_id: str
    operator: SymmetryOperation
    is_original: bool = False


# =============================================================================
# Bioassembly Expander
# =============================================================================

@dataclass
class BioassemblyExpanderConfig:
    """Configuration for bioassembly expansion.
    
    Attributes:
        assembly_id: Which assembly to expand ("1" is typically the primary)
        max_operations: Maximum number of operations (for large symmetry groups)
        max_chains: Maximum chains in expanded assembly
        max_atoms: Maximum atoms in expanded assembly
        preserve_original_ids: Keep original chain IDs for identity operation
        chain_id_scheme: How to generate new chain IDs ("suffix", "sequential")
        validate_symmetry: Check that operations are valid transformations
    """
    assembly_id: str = "1"
    max_operations: int = 100
    max_chains: int = 1000
    max_atoms: int = 500000
    preserve_original_ids: bool = True
    chain_id_scheme: str = "suffix"  # "suffix" or "sequential"
    validate_symmetry: bool = True


class BioassemblyExpander:
    """Expands asymmetric unit to biological assembly using symmetry operations.
    
    This class implements bioassembly expansion as described in the AF3 supplement.
    It takes a structure containing the asymmetric unit and applies the
    symmetry operations defined in the mmCIF file to generate the complete
    biological assembly.
    
    Example usage:
        >>> expander = BioassemblyExpander()
        >>> expanded = expander.expand(structure, assembly)
    """
    
    def __init__(self, config: Optional[BioassemblyExpanderConfig] = None):
        """Initialize expander with configuration.
        
        Args:
            config: Expansion configuration (uses defaults if None)
        """
        self.config = config or BioassemblyExpanderConfig()
        self._chain_mapping: List[ChainMapping] = []
    
    def expand(
        self,
        structure: Structure,
        assembly: AssemblyDefinition,
    ) -> Tuple[Structure, List[ChainMapping]]:
        """Expand structure to bioassembly.
        
        Args:
            structure: Input structure (asymmetric unit)
            assembly: Assembly definition with symmetry operations
            
        Returns:
            Tuple of (expanded structure, chain mappings)
        """
        self._chain_mapping = []
        
        # Validate inputs
        if not assembly.operations:
            logger.warning(f"No operations for assembly {assembly.assembly_id}")
            return structure, []
        
        # Check size limits
        if len(assembly.operations) > self.config.max_operations:
            logger.warning(
                f"Assembly has {len(assembly.operations)} operations, "
                f"exceeding limit of {self.config.max_operations}"
            )
            return structure, []
        
        # Check if only identity operation
        if len(assembly.operations) == 1 and assembly.operations[0].is_identity:
            logger.debug("Assembly has only identity operation, returning original")
            # Still apply chain filtering
            return self._filter_chains(structure, assembly.chain_ids), []
        
        # Validate operations if requested
        if self.config.validate_symmetry:
            self._validate_operations(assembly.operations)
        
        # Generate expanded chains
        expanded_chains: Dict[str, Chain] = {}
        expanded_bonds: List[Bond] = []
        
        # Track chain ID generation
        chain_id_counter = 0
        used_chain_ids: Set[str] = set()
        
        for op in assembly.operations:
            is_identity = op.is_identity
            
            for chain_id in assembly.chain_ids:
                if chain_id not in structure.chains:
                    logger.warning(f"Chain {chain_id} not found in structure")
                    continue
                
                original_chain = structure.chains[chain_id]
                
                # Generate new chain ID
                new_chain_id = self._generate_chain_id(
                    chain_id, op, is_identity, used_chain_ids, chain_id_counter
                )
                used_chain_ids.add(new_chain_id)
                chain_id_counter += 1
                
                # Transform chain
                if is_identity:
                    # Use original coordinates
                    new_chain = self._copy_chain(original_chain, new_chain_id)
                else:
                    new_chain = self._transform_chain(original_chain, new_chain_id, op)
                
                expanded_chains[new_chain_id] = new_chain
                
                # Track mapping
                self._chain_mapping.append(ChainMapping(
                    original_chain_id=chain_id,
                    new_chain_id=new_chain_id,
                    operator=op,
                    is_original=is_identity,
                ))
        
        # Check final size
        total_atoms = sum(c.num_atoms for c in expanded_chains.values())
        if total_atoms > self.config.max_atoms:
            logger.warning(
                f"Expanded assembly has {total_atoms} atoms, "
                f"exceeding limit of {self.config.max_atoms}"
            )
        
        # Transform bonds
        expanded_bonds = self._transform_bonds(structure.bonds, assembly.operations, assembly.chain_ids)
        
        # Create expanded structure
        expanded = Structure(
            pdb_id=structure.pdb_id,
            chains=expanded_chains,
            resolution=structure.resolution,
            method=structure.method,
            release_date=structure.release_date,
            bonds=expanded_bonds,
            title=structure.title,
            authors=structure.authors,
        )
        
        return expanded, self._chain_mapping
    
    def _filter_chains(
        self,
        structure: Structure,
        chain_ids: Set[str],
    ) -> Structure:
        """Filter structure to only include specified chains."""
        filtered_chains = {
            cid: chain for cid, chain in structure.chains.items()
            if cid in chain_ids
        }
        
        return Structure(
            pdb_id=structure.pdb_id,
            chains=filtered_chains,
            resolution=structure.resolution,
            method=structure.method,
            release_date=structure.release_date,
            bonds=[b for b in structure.bonds if b.chain1_id in chain_ids and b.chain2_id in chain_ids],
            title=structure.title,
            authors=structure.authors,
        )
    
    def _generate_chain_id(
        self,
        original_id: str,
        operation: SymmetryOperation,
        is_identity: bool,
        used_ids: Set[str],
        counter: int,
    ) -> str:
        """Generate a unique chain ID for transformed chain."""
        if is_identity and self.config.preserve_original_ids:
            if original_id not in used_ids:
                return original_id
        
        if self.config.chain_id_scheme == "suffix":
            # Append operator ID as suffix
            suffix = f"_{operation.operator_id}" if not is_identity else ""
            new_id = f"{original_id}{suffix}"
            
            # Handle collisions
            collision_counter = 1
            while new_id in used_ids:
                new_id = f"{original_id}_{operation.operator_id}_{collision_counter}"
                collision_counter += 1
            
            return new_id
        
        else:  # sequential
            # Use sequential single characters/numbers
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            
            if counter < len(chars):
                new_id = chars[counter]
            else:
                # Extended: AA, AB, ...
                first = counter // len(chars)
                second = counter % len(chars)
                new_id = f"{chars[first - 1]}{chars[second]}"
            
            return new_id
    
    def _copy_chain(self, chain: Chain, new_chain_id: str) -> Chain:
        """Create a copy of a chain with new ID."""
        new_residues = []
        
        for residue in chain.residues:
            new_atoms = {}
            for atom_name, atom in residue.atoms.items():
                new_atoms[atom_name] = Atom(
                    name=atom.name,
                    element=atom.element,
                    coords=atom.coords.copy(),
                    occupancy=atom.occupancy,
                    b_factor=atom.b_factor,
                    charge=atom.charge,
                    is_hetero=atom.is_hetero,
                    alt_loc=atom.alt_loc,
                    serial=atom.serial,
                )
            
            new_residues.append(Residue(
                name=residue.name,
                seq_id=residue.seq_id,
                atoms=new_atoms,
                insertion_code=residue.insertion_code,
                is_standard=residue.is_standard,
            ))
        
        return Chain(
            chain_id=new_chain_id,
            residues=new_residues,
            entity_id=chain.entity_id,
            chain_type=chain.chain_type,
        )
    
    def _transform_chain(
        self,
        chain: Chain,
        new_chain_id: str,
        operation: SymmetryOperation,
    ) -> Chain:
        """Transform chain coordinates using symmetry operation."""
        new_residues = []
        
        for residue in chain.residues:
            new_atoms = {}
            for atom_name, atom in residue.atoms.items():
                # Apply transformation: x' = R @ x + t
                new_coords = operation.transform_point(atom.coords)
                
                new_atoms[atom_name] = Atom(
                    name=atom.name,
                    element=atom.element,
                    coords=new_coords.astype(np.float32),
                    occupancy=atom.occupancy,
                    b_factor=atom.b_factor,
                    charge=atom.charge,
                    is_hetero=atom.is_hetero,
                    alt_loc=atom.alt_loc,
                    serial=atom.serial,
                )
            
            new_residues.append(Residue(
                name=residue.name,
                seq_id=residue.seq_id,
                atoms=new_atoms,
                insertion_code=residue.insertion_code,
                is_standard=residue.is_standard,
            ))
        
        return Chain(
            chain_id=new_chain_id,
            residues=new_residues,
            entity_id=chain.entity_id,
            chain_type=chain.chain_type,
        )
    
    def _transform_bonds(
        self,
        bonds: List[Bond],
        operations: List[SymmetryOperation],
        chain_ids: Set[str],
    ) -> List[Bond]:
        """Transform inter-chain bonds for expanded assembly."""
        expanded_bonds = []
        
        for bond in bonds:
            # Skip bonds not involving assembly chains
            if bond.chain1_id not in chain_ids or bond.chain2_id not in chain_ids:
                continue
            
            for op in operations:
                suffix = f"_{op.operator_id}" if not op.is_identity else ""
                
                expanded_bonds.append(Bond(
                    chain1_id=f"{bond.chain1_id}{suffix}",
                    res1_seq_id=bond.res1_seq_id,
                    atom1_name=bond.atom1_name,
                    chain2_id=f"{bond.chain2_id}{suffix}",
                    res2_seq_id=bond.res2_seq_id,
                    atom2_name=bond.atom2_name,
                    bond_order=bond.bond_order,
                ))
        
        return expanded_bonds
    
    def _validate_operations(self, operations: List[SymmetryOperation]) -> None:
        """Validate that operations are valid rotation matrices."""
        for op in operations:
            # Check rotation matrix is orthogonal
            RtR = op.rotation.T @ op.rotation
            if not np.allclose(RtR, np.eye(3), atol=ROTATION_TOLERANCE):
                logger.warning(
                    f"Operation {op.operator_id} has non-orthogonal rotation matrix"
                )
            
            # Check determinant is +1 (proper rotation)
            det = np.linalg.det(op.rotation)
            if not np.isclose(det, 1.0, atol=ROTATION_TOLERANCE):
                logger.warning(
                    f"Operation {op.operator_id} has determinant {det:.4f} "
                    f"(expected 1.0 for proper rotation)"
                )
    
    @property
    def chain_mapping(self) -> List[ChainMapping]:
        """Get chain mapping from last expansion."""
        return self._chain_mapping


# =============================================================================
# Symmetry Analysis
# =============================================================================

def analyze_symmetry(operations: List[SymmetryOperation]) -> Dict[str, Any]:
    """Analyze symmetry operations to determine symmetry group.
    
    Args:
        operations: List of symmetry operations
        
    Returns:
        Dictionary with symmetry analysis results
    """
    n_ops = len(operations)
    
    # Count identity operations
    n_identity = sum(1 for op in operations if op.is_identity)
    n_pure_rotation = sum(1 for op in operations if op.is_pure_rotation and not op.is_identity)
    n_translation = sum(1 for op in operations if not op.is_pure_rotation)
    
    # Collect rotation angles
    rotation_angles = []
    for op in operations:
        if not op.is_identity:
            rotation_angles.append(op.rotation_angle)
    
    # Analyze rotation axes
    rotation_axes = []
    for op in operations:
        if not op.is_identity and op.is_pure_rotation:
            axis = op.rotation_axis
            if axis is not None:
                rotation_axes.append(axis)
    
    return {
        "num_operations": n_ops,
        "num_identity": n_identity,
        "num_pure_rotation": n_pure_rotation,
        "num_with_translation": n_translation,
        "rotation_angles": rotation_angles,
        "unique_angles": list(set(round(a, 1) for a in rotation_angles)),
        "symmetry_type": _infer_symmetry_type(n_ops, rotation_angles),
    }


def _infer_symmetry_type(n_ops: int, rotation_angles: List[float]) -> str:
    """Infer symmetry type from operation count and rotation angles."""
    if n_ops == 1:
        return "C1 (identity)"
    
    # Round angles for comparison
    rounded = [round(a) for a in rotation_angles]
    
    if n_ops == 2:
        if 180 in rounded:
            return "C2 (2-fold rotational)"
    elif n_ops == 3:
        if 120 in rounded or 240 in rounded:
            return "C3 (3-fold rotational)"
    elif n_ops == 4:
        if 90 in rounded:
            return "C4 (4-fold rotational)"
        elif 180 in rounded:
            return "D2 (dihedral)"
    elif n_ops == 6:
        if 60 in rounded:
            return "C6 (6-fold rotational)"
        elif 120 in rounded:
            return "D3 (dihedral)"
    elif n_ops == 12:
        return "T (tetrahedral) or D6"
    elif n_ops == 24:
        return "O (octahedral)"
    elif n_ops == 60:
        return "I (icosahedral)"
    
    return f"Unknown ({n_ops} operations)"


def compute_assembly_center(structure: Structure) -> np.ndarray:
    """Compute geometric center of assembly.
    
    Args:
        structure: Structure to analyze
        
    Returns:
        Center of mass coordinates (3,)
    """
    all_coords = []
    for chain in structure.chains.values():
        for residue in chain.residues:
            for atom in residue.atoms.values():
                all_coords.append(atom.coords)
    
    if all_coords:
        return np.mean(np.stack(all_coords), axis=0)
    return np.zeros(3, dtype=np.float32)


def compute_assembly_radius(structure: Structure, center: Optional[np.ndarray] = None) -> float:
    """Compute maximum radius of assembly from center.
    
    Args:
        structure: Structure to analyze
        center: Center point (computed if not provided)
        
    Returns:
        Maximum distance from center to any atom
    """
    if center is None:
        center = compute_assembly_center(structure)
    
    max_radius = 0.0
    for chain in structure.chains.values():
        for residue in chain.residues:
            for atom in residue.atoms.values():
                dist = np.linalg.norm(atom.coords - center)
                max_radius = max(max_radius, dist)
    
    return max_radius


# =============================================================================
# Utility Functions
# =============================================================================

def parse_operator_expression(expression: str) -> List[str]:
    """Parse PDB operator expression like '1,2,3' or '(1-5)'.
    
    PDB uses expressions like:
    - "1" - single operator
    - "1,2,3" - list of operators
    - "(1-5)" - range of operators
    - "(1-3)(4-6)" - Cartesian product (combined operators)
    
    Args:
        expression: Operator expression string
        
    Returns:
        List of operator IDs
    """
    expression = expression.strip()
    
    # Handle Cartesian product notation: (1-3)(4-6)
    if ")(" in expression:
        parts = expression.split(")(")
        parts[0] = parts[0].lstrip("(")
        parts[-1] = parts[-1].rstrip(")")
        
        # Parse each part
        part_ids = [_parse_single_expression(p) for p in parts]
        
        # Generate Cartesian product
        result = [""]
        for ids in part_ids:
            new_result = []
            for prefix in result:
                for op_id in ids:
                    sep = "_" if prefix else ""
                    new_result.append(f"{prefix}{sep}{op_id}")
            result = new_result
        
        return result
    
    return _parse_single_expression(expression)


def _parse_single_expression(expression: str) -> List[str]:
    """Parse a single operator expression (no Cartesian product)."""
    expression = expression.strip("()")
    operator_ids = []
    
    for part in expression.split(","):
        part = part.strip()
        if not part:
            continue
        
        if "-" in part and not part.startswith("-"):
            # Range like "1-5"
            try:
                start, end = part.split("-", 1)
                for i in range(int(start), int(end) + 1):
                    operator_ids.append(str(i))
            except ValueError:
                operator_ids.append(part)
        else:
            operator_ids.append(part)
    
    return operator_ids


def create_icosahedral_operations() -> List[SymmetryOperation]:
    """Generate the 60 operations of icosahedral symmetry.
    
    This is useful for viral capsid structures.
    
    Returns:
        List of 60 symmetry operations
    """
    operations = []
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedral 5-fold axes
    five_fold_axes = np.array([
        [0, 1, phi],
        [0, 1, -phi],
        [0, -1, phi],
        [0, -1, -phi],
        [1, phi, 0],
        [1, -phi, 0],
        [-1, phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1],
    ]) / np.sqrt(1 + phi**2)
    
    op_id = 1
    
    # Identity
    operations.append(SymmetryOperation.identity())
    
    # 5-fold rotations (72° and 144°)
    for axis in five_fold_axes[:6]:  # 6 unique axes
        for angle in [72, 144, 216, 288]:
            operations.append(
                SymmetryOperation.rotation_around_axis(axis, angle, str(op_id))
            )
            op_id += 1
    
    # 3-fold rotations (120° and 240°) - 10 axes
    # ... would need full implementation
    
    # 2-fold rotations (180°) - 15 axes
    # ... would need full implementation
    
    return operations


def get_assembly_statistics(
    original: Structure,
    expanded: Structure,
    mappings: List[ChainMapping],
) -> Dict[str, Any]:
    """Get statistics about assembly expansion.
    
    Args:
        original: Original asymmetric unit
        expanded: Expanded assembly
        mappings: Chain mappings from expansion
        
    Returns:
        Dictionary of statistics
    """
    return {
        "original_chains": original.num_chains,
        "expanded_chains": expanded.num_chains,
        "expansion_factor": expanded.num_chains / max(1, original.num_chains),
        "original_atoms": original.num_atoms,
        "expanded_atoms": expanded.num_atoms,
        "num_operations": len(set(m.operator.operator_id for m in mappings)),
        "symmetry_copies": len(mappings) // max(1, original.num_chains),
    }
