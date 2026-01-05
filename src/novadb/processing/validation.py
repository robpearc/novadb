"""Structure validation for quality control.

Provides validation utilities for ensuring structure quality
before processing through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

# Backbone atoms for validation
PROTEIN_BACKBONE_ATOMS = {"N", "CA", "C", "O"}
RNA_BACKBONE_ATOMS = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'", "C2'", "O2'", "C1'", "O4'"}
DNA_BACKBONE_ATOMS = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'", "C2'", "C1'", "O4'"}

# Covalent radii for bond detection (Å)
COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
    "P": 1.07, "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39,
    "MG": 1.41, "CA": 1.76, "ZN": 1.22, "FE": 1.32, "MN": 1.39,
}


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    message: str
    chain_id: Optional[str] = None
    residue_id: Optional[int] = None
    atom_name: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of structure validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the result."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.is_valid = False


@dataclass
class StructureValidationConfig:
    """Configuration for structure validation."""
    check_backbone: bool = True
    check_atom_coords: bool = True
    check_occupancy: bool = True
    min_occupancy: float = 0.5
    max_missing_backbone_frac: float = 0.2


class StructureValidator:
    """Validates structures for quality issues."""

    def __init__(self, config: Optional[StructureValidationConfig] = None):
        self.config = config or StructureValidationConfig()

    def validate(self, structure) -> ValidationResult:
        """Validate a structure."""
        result = ValidationResult(is_valid=True)

        # Check each chain
        for chain_id, chain in structure.chains.items():
            # Check backbone completeness
            if self.config.check_backbone:
                self._check_backbone(chain, chain_id, result)

            # Check atom coordinates
            if self.config.check_atom_coords:
                self._check_coordinates(chain, chain_id, result)

        return result

    def _check_backbone(self, chain, chain_id: str, result: ValidationResult) -> None:
        """Check backbone completeness."""
        from novadb.data.parsers.structure import ChainType

        if chain.chain_type == ChainType.PROTEIN:
            required = PROTEIN_BACKBONE_ATOMS
        elif chain.chain_type in (ChainType.RNA,):
            required = RNA_BACKBONE_ATOMS
        elif chain.chain_type in (ChainType.DNA,):
            required = DNA_BACKBONE_ATOMS
        else:
            return

        missing_count = 0
        total_residues = len(chain.residues)

        for residue in chain.residues:
            missing = required - set(residue.atoms.keys())
            if missing:
                missing_count += 1

        if total_residues > 0:
            missing_frac = missing_count / total_residues
            if missing_frac > self.config.max_missing_backbone_frac:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Chain {chain_id} has {missing_frac:.1%} missing backbone atoms",
                    chain_id=chain_id,
                ))

    def _check_coordinates(self, chain, chain_id: str, result: ValidationResult) -> None:
        """Check for invalid coordinates."""
        import numpy as np

        for residue in chain.residues:
            for atom_name, atom in residue.atoms.items():
                if np.any(np.isnan(atom.coords)):
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"NaN coordinates in {chain_id}:{residue.seq_id}:{atom_name}",
                        chain_id=chain_id,
                        residue_id=residue.seq_id,
                        atom_name=atom_name,
                    ))


def validate_structure(structure, config: Optional[StructureValidationConfig] = None) -> ValidationResult:
    """Convenience function to validate a structure."""
    validator = StructureValidator(config)
    return validator.validate(structure)


def is_structure_valid(structure, config: Optional[StructureValidationConfig] = None) -> bool:
    """Check if a structure passes validation."""
    return validate_structure(structure, config).is_valid


def create_strict_config() -> StructureValidationConfig:
    """Create a strict validation config."""
    return StructureValidationConfig(
        check_backbone=True,
        check_atom_coords=True,
        check_occupancy=True,
        min_occupancy=0.7,
        max_missing_backbone_frac=0.1,
    )


def create_lenient_config() -> StructureValidationConfig:
    """Create a lenient validation config."""
    return StructureValidationConfig(
        check_backbone=False,
        check_atom_coords=True,
        check_occupancy=False,
        min_occupancy=0.0,
        max_missing_backbone_frac=0.5,
    )


__all__ = [
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "StructureValidationConfig",
    "StructureValidator",
    "validate_structure",
    "is_structure_valid",
    "create_strict_config",
    "create_lenient_config",
    "PROTEIN_BACKBONE_ATOMS",
    "RNA_BACKBONE_ATOMS",
    "DNA_BACKBONE_ATOMS",
    "COVALENT_RADII",
]
