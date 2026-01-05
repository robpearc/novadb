"""Reference conformer generation for AlphaFold3.

Implements reference conformer generation from AF3 Supplement Section 2.8:
- RDKit-based 3D conformer generation for ligands
- CCD ideal coordinate extraction for standard residues
- Conformer alignment to experimental structures
- Chirality preservation and validation

From AF3 Section 2.8:
"For each token, we provide reference conformer coordinates that define
the idealized geometry. For standard residues, these come from the CCD.
For ligands, we generate conformers using RDKit."

Reference: Abramson et al. 2024, Nature.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

# Suppress RDKit warnings during import
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms
        RDKIT_AVAILABLE = True
    except ImportError:
        RDKIT_AVAILABLE = False
        Chem = None
        AllChem = None
        rdMolAlign = None
        rdMolTransforms = None
        logger.warning("RDKit not available. Conformer generation will be limited.")



# Constants
MAX_CONFORMER_ATTEMPTS = 50
DEFAULT_RANDOM_SEED = 42
RMSD_THRESHOLD = 0.5
ENERGY_WINDOW = 50.0  # kcal/mol


@dataclass(frozen=True)
class ConformerGenerationConfig:
    """Configuration for conformer generation.
    
    From AF3 Section 2.8: Conformers are generated using RDKit
    with ETKDG method and energy minimization.
    
    Attributes:
        num_conformers: Number of conformers to generate.
        max_attempts: Maximum embedding attempts.
        random_seed: Random seed for reproducibility.
        use_random_coords: Whether to use random initial coordinates.
        force_field: Force field for minimization ('MMFF94' or 'UFF').
        max_iterations: Maximum minimization iterations.
        prune_rms_threshold: RMSD threshold for pruning similar conformers.
        energy_window: Energy window for filtering (kcal/mol).
        use_basic_knowledge: Use basic chemical knowledge in ETKDG.
        use_exp_torsion_angles: Use experimental torsion angle preferences.
        enforce_chirality: Preserve stereochemistry.
    """
    
    num_conformers: int = 10
    max_attempts: int = MAX_CONFORMER_ATTEMPTS
    random_seed: int = DEFAULT_RANDOM_SEED
    use_random_coords: bool = False
    force_field: str = "MMFF94"
    max_iterations: int = 500
    prune_rms_threshold: float = RMSD_THRESHOLD
    energy_window: float = ENERGY_WINDOW
    use_basic_knowledge: bool = True
    use_exp_torsion_angles: bool = True
    enforce_chirality: bool = True


@dataclass(frozen=True)
class CCDConfig:
    """Configuration for CCD coordinate extraction.
    
    Attributes:
        ccd_dir: Directory containing CCD component files.
        use_ideal_coords: Use ideal (True) or model (False) coordinates.
        include_hydrogens: Include hydrogen atoms.
        fallback_to_model: Fall back to model coords if ideal unavailable.
    """
    
    ccd_dir: Optional[Path] = None
    use_ideal_coords: bool = True
    include_hydrogens: bool = False
    fallback_to_model: bool = True


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration for conformer alignment.
    
    Attributes:
        max_iterations: Maximum ICP iterations.
        rmsd_threshold: RMSD convergence threshold.
        use_kabsch: Use Kabsch algorithm for optimal rotation.
        weight_by_mass: Weight atoms by mass during alignment.
        align_heavy_only: Only align heavy (non-hydrogen) atoms.
    """
    
    max_iterations: int = 100
    rmsd_threshold: float = 0.01
    use_kabsch: bool = True
    weight_by_mass: bool = False
    align_heavy_only: bool = True


@dataclass
class Conformer:
    """A single molecular conformer.
    
    Attributes:
        coords: Atom coordinates (Natoms, 3).
        atom_names: Atom names.
        elements: Element symbols.
        charges: Formal charges.
        energy: Conformer energy (kcal/mol).
        rmsd_to_reference: RMSD to reference structure.
        is_valid: Whether conformer passed validation.
    """
    
    coords: np.ndarray
    atom_names: List[str]
    elements: List[str]
    charges: List[float]
    energy: Optional[float] = None
    rmsd_to_reference: Optional[float] = None
    is_valid: bool = True


@dataclass
class ConformerEnsemble:
    """Collection of conformers for a molecule.
    
    Attributes:
        conformers: List of conformer objects.
        smiles: SMILES string.
        residue_name: CCD residue name.
        best_conformer_idx: Index of lowest energy conformer.
    """
    
    conformers: List[Conformer]
    smiles: Optional[str] = None
    residue_name: Optional[str] = None
    best_conformer_idx: int = 0
    
    @property
    def num_conformers(self) -> int:
        return len(self.conformers)
    
    @property
    def best_conformer(self) -> Optional[Conformer]:
        if self.conformers:
            return self.conformers[self.best_conformer_idx]
        return None
    
    def get_coords_array(self) -> np.ndarray:
        """Get coordinates as (Nconf, Natoms, 3) array."""
        if not self.conformers:
            return np.zeros((0, 0, 3), dtype=np.float32)
        return np.stack([c.coords for c in self.conformers])


@dataclass
class RDKitConformerGenerator:
    """Generates 3D conformers using RDKit.
    
    From AF3 Section 2.8: Uses ETKDG (Experimental-Torsion Knowledge
    Distance Geometry) with energy minimization.
    
    Attributes:
        config: Generation configuration.
    """
    
    config: ConformerGenerationConfig = field(default_factory=ConformerGenerationConfig)
    
    def generate_from_smiles(
        self,
        smiles: str,
        reference_coords: Optional[np.ndarray] = None,
    ) -> ConformerEnsemble:
        """Generate conformers from SMILES string.
        
        Args:
            smiles: SMILES string.
            reference_coords: Optional reference coordinates for alignment.
            
        Returns:
            ConformerEnsemble with generated conformers.
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, returning empty ensemble")
            return ConformerEnsemble(conformers=[], smiles=smiles)
        
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Failed to parse SMILES: %s", smiles)
            return ConformerEnsemble(conformers=[], smiles=smiles)
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate conformers
        conformers = self._generate_conformers(mol)
        
        # Minimize and filter
        conformers = self._minimize_conformers(mol, conformers)
        conformers = self._filter_conformers(mol, conformers)
        
        # Convert to Conformer objects
        result_conformers = []
        for conf_id in conformers:
            conf = self._extract_conformer(mol, conf_id)
            result_conformers.append(conf)
        
        # Align to reference if provided
        if reference_coords is not None and result_conformers:
            result_conformers = self._align_to_reference(
                result_conformers,
                reference_coords,
            )
        
        # Find best conformer
        best_idx = 0
        if result_conformers:
            energies = [c.energy for c in result_conformers if c.energy is not None]
            if energies:
                best_idx = energies.index(min(energies))
        
        return ConformerEnsemble(
            conformers=result_conformers,
            smiles=smiles,
            best_conformer_idx=best_idx,
        )
    
    def generate_from_mol(
        self,
        mol: "Mol",
        reference_coords: Optional[np.ndarray] = None,
    ) -> ConformerEnsemble:
        """Generate conformers from RDKit Mol object.
        
        Args:
            mol: RDKit Mol object.
            reference_coords: Optional reference coordinates.
            
        Returns:
            ConformerEnsemble with generated conformers.
        """
        if not RDKIT_AVAILABLE:
            return ConformerEnsemble(conformers=[])
        
        # Ensure hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate
        conformers = self._generate_conformers(mol)
        conformers = self._minimize_conformers(mol, conformers)
        conformers = self._filter_conformers(mol, conformers)
        
        result_conformers = []
        for conf_id in conformers:
            conf = self._extract_conformer(mol, conf_id)
            result_conformers.append(conf)
        
        if reference_coords is not None and result_conformers:
            result_conformers = self._align_to_reference(
                result_conformers,
                reference_coords,
            )
        
        best_idx = 0
        if result_conformers:
            energies = [c.energy for c in result_conformers if c.energy is not None]
            if energies:
                best_idx = energies.index(min(energies))
        
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol)) if mol else None
        
        return ConformerEnsemble(
            conformers=result_conformers,
            smiles=smiles,
            best_conformer_idx=best_idx,
        )
    
    def _generate_conformers(self, mol: "Mol") -> List[int]:
        """Generate initial conformers using ETKDG.
        
        Args:
            mol: RDKit Mol object with hydrogens.
            
        Returns:
            List of conformer IDs.
        """
        params = AllChem.ETKDGv3()
        params.randomSeed = self.config.random_seed
        params.useBasicKnowledge = self.config.use_basic_knowledge
        params.useExpTorsionAnglePrefs = self.config.use_exp_torsion_angles
        params.enforceChirality = self.config.enforce_chirality
        params.useRandomCoords = self.config.use_random_coords
        params.maxAttempts = self.config.max_attempts
        params.pruneRmsThresh = self.config.prune_rms_threshold
        
        # Generate conformers
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=self.config.num_conformers,
            params=params,
        )
        
        if not conf_ids:
            # Try with random coordinates as fallback
            params.useRandomCoords = True
            conf_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=self.config.num_conformers,
                params=params,
            )
        
        return list(conf_ids)
    
    def _minimize_conformers(
        self,
        mol: "Mol",
        conf_ids: List[int],
    ) -> List[int]:
        """Minimize conformers with force field.
        
        Args:
            mol: RDKit Mol object.
            conf_ids: Conformer IDs to minimize.
            
        Returns:
            List of successfully minimized conformer IDs.
        """
        if not conf_ids:
            return []
        
        minimized = []
        
        for conf_id in conf_ids:
            try:
                if self.config.force_field == "MMFF94":
                    # Get MMFF properties
                    props = AllChem.MMFFGetMoleculeProperties(mol)
                    if props is None:
                        # Fall back to UFF
                        result = AllChem.UFFOptimizeMolecule(
                            mol,
                            confId=conf_id,
                            maxIters=self.config.max_iterations,
                        )
                    else:
                        ff = AllChem.MMFFGetMoleculeForceField(
                            mol,
                            props,
                            confId=conf_id,
                        )
                        if ff is not None:
                            ff.Minimize(maxIts=self.config.max_iterations)
                        result = 0
                else:
                    result = AllChem.UFFOptimizeMolecule(
                        mol,
                        confId=conf_id,
                        maxIters=self.config.max_iterations,
                    )
                
                if result == 0:  # Converged
                    minimized.append(conf_id)
                else:
                    minimized.append(conf_id)  # Keep anyway
                    
            except Exception as e:
                logger.debug("Minimization failed for conf %d: %s", conf_id, e)
                minimized.append(conf_id)  # Keep unminimized
        
        return minimized
    
    def _filter_conformers(
        self,
        mol: "Mol",
        conf_ids: List[int],
    ) -> List[int]:
        """Filter conformers by energy and RMSD.
        
        Args:
            mol: RDKit Mol object.
            conf_ids: Conformer IDs to filter.
            
        Returns:
            Filtered list of conformer IDs.
        """
        if len(conf_ids) <= 1:
            return conf_ids
        
        # Calculate energies
        energies = {}
        for conf_id in conf_ids:
            try:
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                    if ff is not None:
                        energies[conf_id] = ff.CalcEnergy()
                        continue
                
                # Fall back to UFF
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                if ff is not None:
                    energies[conf_id] = ff.CalcEnergy()
                else:
                    energies[conf_id] = 0.0
            except Exception:
                energies[conf_id] = 0.0
        
        if not energies:
            return conf_ids
        
        # Filter by energy window
        min_energy = min(energies.values())
        filtered = [
            conf_id for conf_id in conf_ids
            if energies.get(conf_id, 0) - min_energy < self.config.energy_window
        ]
        
        # Sort by energy
        filtered.sort(key=lambda x: energies.get(x, 0))
        
        return filtered
    
    def _extract_conformer(
        self,
        mol: "Mol",
        conf_id: int,
    ) -> Conformer:
        """Extract conformer data from RDKit molecule.
        
        Args:
            mol: RDKit Mol object.
            conf_id: Conformer ID.
            
        Returns:
            Conformer object.
        """
        conf = mol.GetConformer(conf_id)
        
        # Get coordinates
        coords = np.array(conf.GetPositions(), dtype=np.float32)
        
        # Get atom info
        atom_names = []
        elements = []
        charges = []
        
        for atom in mol.GetAtoms():
            atom_names.append(atom.GetSymbol() + str(atom.GetIdx()))
            elements.append(atom.GetSymbol())
            charges.append(float(atom.GetFormalCharge()))
        
        # Calculate energy
        energy = None
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                if ff is not None:
                    energy = ff.CalcEnergy()
        except Exception:
            pass
        
        return Conformer(
            coords=coords,
            atom_names=atom_names,
            elements=elements,
            charges=charges,
            energy=energy,
            is_valid=True,
        )
    
    def _align_to_reference(
        self,
        conformers: List[Conformer],
        reference_coords: np.ndarray,
    ) -> List[Conformer]:
        """Align conformers to reference coordinates.
        
        Args:
            conformers: List of conformers.
            reference_coords: Reference coordinates (Natoms, 3).
            
        Returns:
            Aligned conformers.
        """
        aligned = []
        
        for conf in conformers:
            if len(conf.coords) != len(reference_coords):
                logger.warning(
                    "Atom count mismatch: %d vs %d",
                    len(conf.coords),
                    len(reference_coords),
                )
                aligned.append(conf)
                continue
            
            # Compute optimal alignment using Kabsch
            aligned_coords, rmsd = kabsch_align(conf.coords, reference_coords)
            
            aligned.append(Conformer(
                coords=aligned_coords,
                atom_names=conf.atom_names,
                elements=conf.elements,
                charges=conf.charges,
                energy=conf.energy,
                rmsd_to_reference=rmsd,
                is_valid=conf.is_valid,
            ))
        
        return aligned


@dataclass
class CCDConformerExtractor:
    """Extracts ideal coordinates from CCD component files.
    
    From AF3 Section 2.8: Standard residues use idealized coordinates
    from the Chemical Component Dictionary.
    
    Attributes:
        config: CCD configuration.
        _cache: Cached CCD data.
    """
    
    config: CCDConfig = field(default_factory=CCDConfig)
    _cache: Dict[str, Dict] = field(default_factory=dict, repr=False)
    
    def get_ideal_coords(
        self,
        residue_name: str,
    ) -> Optional[Conformer]:
        """Get ideal coordinates for a standard residue.
        
        Args:
            residue_name: 3-letter residue code.
            
        Returns:
            Conformer with ideal coordinates or None.
        """
        ccd_data = self._load_ccd_component(residue_name)
        
        if ccd_data is None:
            return None
        
        if self.config.use_ideal_coords and ccd_data.get("coords_ideal") is not None:
            coords = ccd_data["coords_ideal"]
        elif ccd_data.get("coords_model") is not None:
            coords = ccd_data["coords_model"]
        else:
            return None
        
        # Filter hydrogens if needed
        if not self.config.include_hydrogens:
            mask = np.array([e != "H" for e in ccd_data["elements"]])
            coords = coords[mask]
            atom_names = [n for n, m in zip(ccd_data["atoms"], mask) if m]
            elements = [e for e, m in zip(ccd_data["elements"], mask) if m]
            charges = [c for c, m in zip(ccd_data["charges"], mask) if m]
        else:
            atom_names = ccd_data["atoms"]
            elements = ccd_data["elements"]
            charges = ccd_data["charges"]
        
        return Conformer(
            coords=coords,
            atom_names=atom_names,
            elements=elements,
            charges=charges,
            energy=None,
            is_valid=True,
        )
    
    def get_smiles(self, residue_name: str) -> Optional[str]:
        """Get SMILES string for a residue.
        
        Args:
            residue_name: 3-letter residue code.
            
        Returns:
            SMILES string or None.
        """
        ccd_data = self._load_ccd_component(residue_name)
        if ccd_data is None:
            return None
        return ccd_data.get("smiles")
    
    def get_bond_info(
        self,
        residue_name: str,
    ) -> Optional[List[Tuple[str, str, str]]]:
        """Get bond information for a residue.
        
        Args:
            residue_name: 3-letter residue code.
            
        Returns:
            List of (atom1, atom2, bond_type) tuples or None.
        """
        ccd_data = self._load_ccd_component(residue_name)
        if ccd_data is None:
            return None
        return ccd_data.get("bonds")
    
    def _load_ccd_component(self, residue_name: str) -> Optional[Dict]:
        """Load CCD component data.
        
        Args:
            residue_name: 3-letter residue code.
            
        Returns:
            Dictionary with CCD data or None.
        """
        if residue_name in self._cache:
            return self._cache[residue_name]
        
        if self.config.ccd_dir is None:
            return None
        
        # Try different file naming conventions
        ccd_file = self.config.ccd_dir / f"{residue_name}.cif"
        if not ccd_file.exists():
            ccd_file = self.config.ccd_dir / f"{residue_name.upper()}.cif"
        if not ccd_file.exists():
            ccd_file = self.config.ccd_dir / f"{residue_name.lower()}.cif"
        if not ccd_file.exists():
            return None
        
        try:
            ccd_data = self._parse_ccd_file(ccd_file)
            self._cache[residue_name] = ccd_data
            return ccd_data
        except Exception as e:
            logger.warning("Failed to parse CCD for %s: %s", residue_name, e)
            return None
    
    def _parse_ccd_file(self, ccd_file: Path) -> Dict:
        """Parse a CCD component file.
        
        Args:
            ccd_file: Path to CCD file.
            
        Returns:
            Dictionary with atom/bond information.
        """
        atoms = []
        elements = []
        charges = []
        coords_ideal = []
        coords_model = []
        bonds = []
        smiles = None
        
        with open(ccd_file) as f:
            lines = f.readlines()
        
        in_atom_block = False
        in_bond_block = False
        
        for line in lines:
            line = line.strip()
            
            # Parse SMILES
            if "_pdbx_chem_comp_descriptor.descriptor" in line and "SMILES" in line:
                # Try to extract SMILES from next lines
                pass
            if line.startswith("_pdbx_chem_comp_descriptor.descriptor"):
                parts = line.split()
                if len(parts) > 1:
                    smiles = parts[-1].strip('"')
            
            # Atom block
            if "_chem_comp_atom." in line:
                in_atom_block = True
                in_bond_block = False
                continue
            
            # Bond block
            if "_chem_comp_bond." in line:
                in_atom_block = False
                in_bond_block = True
                continue
            
            # End of block
            if line.startswith("#") or line.startswith("loop_"):
                if in_atom_block and atoms:
                    in_atom_block = False
                if in_bond_block and bonds:
                    in_bond_block = False
                continue
            
            # Parse atom line
            if in_atom_block and not line.startswith("_"):
                parts = line.split()
                if len(parts) >= 12:
                    try:
                        atom_name = parts[1]
                        element = parts[2]
                        charge = float(parts[3]) if parts[3] not in (".", "?") else 0.0
                        
                        # Ideal coordinates (columns 5-7 typically)
                        x_ideal = float(parts[4]) if parts[4] not in (".", "?") else 0.0
                        y_ideal = float(parts[5]) if parts[5] not in (".", "?") else 0.0
                        z_ideal = float(parts[6]) if parts[6] not in (".", "?") else 0.0
                        
                        # Model coordinates (columns 8-10 typically)
                        x_model = float(parts[7]) if len(parts) > 7 and parts[7] not in (".", "?") else x_ideal
                        y_model = float(parts[8]) if len(parts) > 8 and parts[8] not in (".", "?") else y_ideal
                        z_model = float(parts[9]) if len(parts) > 9 and parts[9] not in (".", "?") else z_ideal
                        
                        atoms.append(atom_name)
                        elements.append(element)
                        charges.append(charge)
                        coords_ideal.append([x_ideal, y_ideal, z_ideal])
                        coords_model.append([x_model, y_model, z_model])
                    except (ValueError, IndexError):
                        continue
            
            # Parse bond line
            if in_bond_block and not line.startswith("_"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        atom1 = parts[1]
                        atom2 = parts[2]
                        bond_type = parts[3]
                        bonds.append((atom1, atom2, bond_type))
                    except (ValueError, IndexError):
                        continue
        
        return {
            "atoms": atoms,
            "elements": elements,
            "charges": charges,
            "coords_ideal": np.array(coords_ideal, dtype=np.float32) if coords_ideal else None,
            "coords_model": np.array(coords_model, dtype=np.float32) if coords_model else None,
            "bonds": bonds,
            "smiles": smiles,
        }


@dataclass
class ConformerAligner:
    """Aligns conformers to reference structures.
    
    From AF3 Section 2.8: Conformers are aligned to experimental
    coordinates when available.
    
    Attributes:
        config: Alignment configuration.
    """
    
    config: AlignmentConfig = field(default_factory=AlignmentConfig)
    
    def align(
        self,
        mobile_coords: np.ndarray,
        target_coords: np.ndarray,
        mobile_mask: Optional[np.ndarray] = None,
        target_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Align mobile coordinates to target.
        
        Args:
            mobile_coords: Coordinates to align (N, 3).
            target_coords: Target coordinates (N, 3).
            mobile_mask: Optional mask for mobile atoms.
            target_mask: Optional mask for target atoms.
            
        Returns:
            Tuple of (aligned_coords, rmsd).
        """
        # Apply masks
        if mobile_mask is not None and target_mask is not None:
            mask = (mobile_mask > 0) & (target_mask > 0)
            mobile_masked = mobile_coords[mask]
            target_masked = target_coords[mask]
        else:
            mobile_masked = mobile_coords
            target_masked = target_coords
        
        if len(mobile_masked) < 3:
            return mobile_coords.copy(), float('inf')
        
        if self.config.use_kabsch:
            aligned, rmsd = kabsch_align(mobile_masked, target_masked)
            
            # Apply same transformation to all atoms
            if mobile_mask is not None:
                # Compute transformation and apply to full coords
                R, t = compute_kabsch_rotation(mobile_masked, target_masked)
                full_aligned = (mobile_coords - mobile_coords.mean(axis=0)) @ R.T + target_coords.mean(axis=0)
                return full_aligned, rmsd
            else:
                return aligned, rmsd
        else:
            # Simple centroid alignment
            mobile_centered = mobile_coords - mobile_coords.mean(axis=0)
            target_centered = target_coords - target_coords.mean(axis=0)
            aligned = mobile_centered + target_coords.mean(axis=0)
            rmsd = np.sqrt(np.mean(np.sum((aligned - target_coords) ** 2, axis=1)))
            return aligned, rmsd
    
    def align_by_atom_names(
        self,
        mobile_coords: np.ndarray,
        mobile_names: Sequence[str],
        target_coords: np.ndarray,
        target_names: Sequence[str],
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Align by matching atom names.
        
        Args:
            mobile_coords: Mobile coordinates.
            mobile_names: Mobile atom names.
            target_coords: Target coordinates.
            target_names: Target atom names.
            
        Returns:
            Tuple of (aligned_coords, rmsd, mapping).
        """
        # Build name to index mapping
        target_name_to_idx = {name: i for i, name in enumerate(target_names)}
        
        # Find matching atoms
        mobile_indices = []
        target_indices = []
        
        for i, name in enumerate(mobile_names):
            if name in target_name_to_idx:
                mobile_indices.append(i)
                target_indices.append(target_name_to_idx[name])
        
        if len(mobile_indices) < 3:
            logger.warning("Too few matching atoms for alignment: %d", len(mobile_indices))
            return mobile_coords.copy(), float('inf'), np.array([])
        
        # Extract matching coordinates
        mobile_matched = mobile_coords[mobile_indices]
        target_matched = target_coords[target_indices]
        
        # Compute alignment on matched atoms
        R, t = compute_kabsch_rotation(mobile_matched, target_matched)
        
        # Apply to all mobile coordinates
        aligned = (mobile_coords - mobile_coords.mean(axis=0)) @ R.T + target_coords[target_indices].mean(axis=0)
        
        # Compute RMSD on matched atoms
        aligned_matched = aligned[mobile_indices]
        rmsd = np.sqrt(np.mean(np.sum((aligned_matched - target_matched) ** 2, axis=1)))
        
        # Create mapping array
        mapping = np.array(list(zip(mobile_indices, target_indices)), dtype=np.int32)
        
        return aligned, rmsd, mapping


@dataclass(frozen=True)
class ReferenceConformerPipeline:
    """Complete pipeline for reference conformer generation.
    
    Implements the full AF3 Section 2.8 workflow:
    1. Use CCD ideal coordinates for standard residues
    2. Generate RDKit conformers for ligands
    3. Align conformers to experimental structure
    
    Attributes:
        rdkit_generator: RDKit conformer generator.
        ccd_extractor: CCD coordinate extractor.
        aligner: Conformer aligner.
    """
    
    rdkit_generator: RDKitConformerGenerator = field(
        default_factory=RDKitConformerGenerator
    )
    ccd_extractor: CCDConformerExtractor = field(
        default_factory=CCDConformerExtractor
    )
    aligner: ConformerAligner = field(default_factory=ConformerAligner)
    
    # Standard residues that use CCD coordinates
    STANDARD_RESIDUES: FrozenSet[str] = frozenset({
        # Amino acids
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        # RNA
        "A", "G", "C", "U",
        # DNA
        "DA", "DG", "DC", "DT",
    })
    
    def generate_reference_conformer(
        self,
        residue_name: str,
        experimental_coords: Optional[np.ndarray] = None,
        experimental_atom_names: Optional[Sequence[str]] = None,
        smiles: Optional[str] = None,
    ) -> Optional[Conformer]:
        """Generate reference conformer for a residue.
        
        Args:
            residue_name: 3-letter residue code.
            experimental_coords: Optional experimental coordinates.
            experimental_atom_names: Optional experimental atom names.
            smiles: Optional SMILES for ligand generation.
            
        Returns:
            Reference conformer or None.
        """
        # Try CCD first for standard residues
        if residue_name.upper() in self.STANDARD_RESIDUES:
            conformer = self.ccd_extractor.get_ideal_coords(residue_name)
            if conformer is not None:
                # Align to experimental if available
                if experimental_coords is not None and experimental_atom_names is not None:
                    aligned_coords, rmsd, _ = self.aligner.align_by_atom_names(
                        conformer.coords,
                        conformer.atom_names,
                        experimental_coords,
                        list(experimental_atom_names),
                    )
                    return Conformer(
                        coords=aligned_coords,
                        atom_names=conformer.atom_names,
                        elements=conformer.elements,
                        charges=conformer.charges,
                        rmsd_to_reference=rmsd,
                        is_valid=True,
                    )
                return conformer
        
        # Try CCD for non-standard residues
        conformer = self.ccd_extractor.get_ideal_coords(residue_name)
        if conformer is not None:
            if experimental_coords is not None and experimental_atom_names is not None:
                aligned_coords, rmsd, _ = self.aligner.align_by_atom_names(
                    conformer.coords,
                    conformer.atom_names,
                    experimental_coords,
                    list(experimental_atom_names),
                )
                return Conformer(
                    coords=aligned_coords,
                    atom_names=conformer.atom_names,
                    elements=conformer.elements,
                    charges=conformer.charges,
                    rmsd_to_reference=rmsd,
                    is_valid=True,
                )
            return conformer
        
        # Fall back to RDKit generation
        if smiles is None:
            smiles = self.ccd_extractor.get_smiles(residue_name)
        
        if smiles is not None and RDKIT_AVAILABLE:
            ensemble = self.rdkit_generator.generate_from_smiles(
                smiles,
                reference_coords=experimental_coords,
            )
            if ensemble.best_conformer is not None:
                return ensemble.best_conformer
        
        # Last resort: use experimental coordinates directly
        if experimental_coords is not None:
            return Conformer(
                coords=experimental_coords.astype(np.float32),
                atom_names=list(experimental_atom_names) if experimental_atom_names else [],
                elements=[],
                charges=[],
                is_valid=True,
            )
        
        return None
    
    def generate_for_structure(
        self,
        residue_names: Sequence[str],
        atom_coords_per_residue: Sequence[np.ndarray],
        atom_names_per_residue: Sequence[Sequence[str]],
    ) -> List[Optional[Conformer]]:
        """Generate reference conformers for entire structure.
        
        Args:
            residue_names: Residue name for each token.
            atom_coords_per_residue: Experimental coordinates per residue.
            atom_names_per_residue: Atom names per residue.
            
        Returns:
            List of conformers (None for failed generation).
        """
        conformers = []
        
        for i, res_name in enumerate(residue_names):
            exp_coords = atom_coords_per_residue[i] if i < len(atom_coords_per_residue) else None
            exp_names = atom_names_per_residue[i] if i < len(atom_names_per_residue) else None
            
            conformer = self.generate_reference_conformer(
                res_name,
                experimental_coords=exp_coords,
                experimental_atom_names=exp_names,
            )
            conformers.append(conformer)
        
        return conformers


def kabsch_align(
    mobile: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Align mobile coordinates to target using Kabsch algorithm.
    
    Args:
        mobile: Mobile coordinates (N, 3).
        target: Target coordinates (N, 3).
        
    Returns:
        Tuple of (aligned_coords, rmsd).
    """
    R, t = compute_kabsch_rotation(mobile, target)
    
    # Apply transformation
    mobile_centered = mobile - mobile.mean(axis=0)
    aligned = mobile_centered @ R.T + target.mean(axis=0)
    
    # Compute RMSD
    rmsd = np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1)))
    
    return aligned.astype(np.float32), float(rmsd)


def compute_kabsch_rotation(
    mobile: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal rotation matrix using Kabsch algorithm.
    
    Args:
        mobile: Mobile coordinates (N, 3).
        target: Target coordinates (N, 3).
        
    Returns:
        Tuple of (rotation_matrix, translation).
    """
    # Center coordinates
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    
    mobile_centered = mobile - mobile_center
    target_centered = target - target_center
    
    # Compute covariance matrix
    H = mobile_centered.T @ target_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Translation
    t = target_center - mobile_center @ R.T
    
    return R.astype(np.float32), t.astype(np.float32)


def compute_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute RMSD between two coordinate sets.
    
    Args:
        coords1: First coordinates (N, 3).
        coords2: Second coordinates (N, 3).
        mask: Optional validity mask.
        
    Returns:
        RMSD value.
    """
    if mask is not None:
        coords1 = coords1[mask > 0]
        coords2 = coords2[mask > 0]
    
    if len(coords1) == 0:
        return float('inf')
    
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def create_conformer_pipeline(
    *,
    ccd_dir: Optional[Path] = None,
    num_conformers: int = 10,
    use_ideal_coords: bool = True,
    include_hydrogens: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> ReferenceConformerPipeline:
    """Factory function to create a configured conformer pipeline.
    
    Args:
        ccd_dir: Directory containing CCD component files.
        num_conformers: Number of RDKit conformers to generate.
        use_ideal_coords: Use ideal CCD coordinates.
        include_hydrogens: Include hydrogen atoms.
        random_seed: Random seed for conformer generation.
        
    Returns:
        Configured ReferenceConformerPipeline instance.
    """
    gen_config = ConformerGenerationConfig(
        num_conformers=num_conformers,
        random_seed=random_seed,
    )
    
    ccd_config = CCDConfig(
        ccd_dir=ccd_dir,
        use_ideal_coords=use_ideal_coords,
        include_hydrogens=include_hydrogens,
    )
    
    return ReferenceConformerPipeline(
        rdkit_generator=RDKitConformerGenerator(config=gen_config),
        ccd_extractor=CCDConformerExtractor(config=ccd_config),
        aligner=ConformerAligner(),
    )
