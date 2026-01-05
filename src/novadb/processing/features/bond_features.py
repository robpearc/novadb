"""Bond feature extraction for AlphaFold3.

Implements bond feature extraction from AF3 Supplement Section 2.8:
- Token-level bond matrices (token_bonds feature)
- Polymer backbone connectivity
- Ligand-polymer covalent bonds
- Intra-ligand bonds
- Glycosidic bonds

From AF3 Table 5:
"token_bonds: A 2D matrix indicating if there is a bond between any atom
in token i and token j, restricted to just polymer-ligand and ligand-ligand
bonds and bonds less than 2.4 Å during training."

Performance optimizations:
- Uses KD-tree for O(n log n) cross-chain bond detection instead of O(n⁴)
- Vectorized intra-residue bond detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import numpy as np

from novadb.data.parsers.structure import Structure, Chain, Residue, Atom, ChainType
from novadb.processing.optimized import (
    detect_cross_chain_bonds_fast,
    find_all_pairs_within_distance_symmetric,
    BondCandidate,
)

logger = logging.getLogger(__name__)


# Bond length thresholds (Å)
COVALENT_BOND_THRESHOLD = 2.4  # AF3 training cutoff
STANDARD_BOND_THRESHOLD = 1.9  # Typical covalent bond
LONG_BOND_THRESHOLD = 2.8  # Extended bonds (disulfides, etc.)

# Standard backbone atoms for connectivity
PROTEIN_BACKBONE = {"N", "CA", "C", "O"}
PROTEIN_PEPTIDE_BOND = ("C", "N")  # C of residue i to N of residue i+1

RNA_BACKBONE = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'", "C2'", "O2'", "C1'"}
DNA_BACKBONE = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'", "C2'", "C1'"}
NUCLEIC_PHOSPHODIESTER = ("O3'", "P")  # O3' of residue i to P of residue i+1


@dataclass(frozen=True)
class Bond:
    """Represents a covalent bond between two atoms."""
    atom1_chain: str
    atom1_residue_idx: int
    atom1_name: str
    atom2_chain: str
    atom2_residue_idx: int
    atom2_name: str
    bond_order: int = 1
    is_polymer_bond: bool = False
    is_ligand_bond: bool = False
    distance: float = 0.0
    
    def __hash__(self) -> int:
        # Order-independent hash
        atoms = tuple(sorted([
            (self.atom1_chain, self.atom1_residue_idx, self.atom1_name),
            (self.atom2_chain, self.atom2_residue_idx, self.atom2_name),
        ]))
        return hash(atoms)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bond):
            return False
        self_atoms = {
            (self.atom1_chain, self.atom1_residue_idx, self.atom1_name),
            (self.atom2_chain, self.atom2_residue_idx, self.atom2_name),
        }
        other_atoms = {
            (other.atom1_chain, other.atom1_residue_idx, other.atom1_name),
            (other.atom2_chain, other.atom2_residue_idx, other.atom2_name),
        }
        return self_atoms == other_atoms


@dataclass
class TokenBond:
    """Bond at token level for AF3 feature."""
    token_idx_1: int
    token_idx_2: int
    bond_type: str = "covalent"  # covalent, ionic, etc.
    
    def __hash__(self) -> int:
        return hash((min(self.token_idx_1, self.token_idx_2), 
                     max(self.token_idx_1, self.token_idx_2)))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenBond):
            return False
        return {self.token_idx_1, self.token_idx_2} == {other.token_idx_1, other.token_idx_2}


@dataclass
class BondFeatures:
    """Extracted bond features for a structure."""
    token_bonds: np.ndarray  # (Ntoken, Ntoken) binary matrix
    atom_bonds: List[Bond]
    num_polymer_bonds: int = 0
    num_ligand_bonds: int = 0
    num_cross_bonds: int = 0  # Polymer-ligand bonds
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for feature storage."""
        return {
            "token_bonds": self.token_bonds.astype(np.float32),
        }


class BondDetector:
    """Detect covalent bonds in structures.
    
    From AF3 Section 2.8: Bond information is used to construct
    the token_bonds feature matrix.
    """
    
    def __init__(
        self,
        covalent_threshold: float = COVALENT_BOND_THRESHOLD,
        include_polymer_backbone: bool = False,  # AF3 excludes backbone
    ):
        """Initialize bond detector.
        
        Args:
            covalent_threshold: Maximum distance for covalent bond (Å)
            include_polymer_backbone: Include polymer backbone bonds
        """
        self.covalent_threshold = covalent_threshold
        self.include_polymer_backbone = include_polymer_backbone
    
    def detect_bonds(self, structure: Structure) -> List[Bond]:
        """Detect all covalent bonds in structure.
        
        Args:
            structure: Structure to analyze
            
        Returns:
            List of detected bonds
        """
        bonds = []
        
        # Detect intra-chain bonds
        for chain_id, chain in structure.chains.items():
            chain_bonds = self._detect_chain_bonds(chain_id, chain)
            bonds.extend(chain_bonds)
        
        # Detect inter-chain bonds (polymer-ligand, ligand-ligand)
        chain_ids = list(structure.chains.keys())
        for i, chain_id_1 in enumerate(chain_ids):
            for chain_id_2 in chain_ids[i + 1:]:
                chain1 = structure.chains[chain_id_1]
                chain2 = structure.chains[chain_id_2]
                
                cross_bonds = self._detect_cross_chain_bonds(
                    chain_id_1, chain1,
                    chain_id_2, chain2,
                )
                bonds.extend(cross_bonds)
        
        return bonds
    
    def _detect_chain_bonds(
        self,
        chain_id: str,
        chain: Chain,
    ) -> List[Bond]:
        """Detect bonds within a chain."""
        bonds = []
        is_polymer = chain.chain_type in {ChainType.PROTEIN, ChainType.RNA, ChainType.DNA}
        
        for i, residue in enumerate(chain.residues):
            # Intra-residue bonds (for ligands)
            if not is_polymer or chain.chain_type == ChainType.LIGAND:
                intra_bonds = self._detect_intra_residue_bonds(chain_id, i, residue)
                for bond in intra_bonds:
                    bond_with_flag = Bond(
                        atom1_chain=bond.atom1_chain,
                        atom1_residue_idx=bond.atom1_residue_idx,
                        atom1_name=bond.atom1_name,
                        atom2_chain=bond.atom2_chain,
                        atom2_residue_idx=bond.atom2_residue_idx,
                        atom2_name=bond.atom2_name,
                        is_ligand_bond=True,
                        distance=bond.distance,
                    )
                    bonds.append(bond_with_flag)
            
            # Inter-residue bonds (polymer backbone or sequential ligand)
            if i < len(chain.residues) - 1:
                next_residue = chain.residues[i + 1]
                
                if is_polymer and self.include_polymer_backbone:
                    backbone_bond = self._detect_backbone_bond(
                        chain_id, i, residue,
                        chain_id, i + 1, next_residue,
                        chain.chain_type,
                    )
                    if backbone_bond:
                        bonds.append(backbone_bond)
        
        return bonds
    
    def _detect_intra_residue_bonds(
        self,
        chain_id: str,
        residue_idx: int,
        residue: Residue,
    ) -> List[Bond]:
        """Detect bonds within a residue (for ligands).

        Uses vectorized distance computation for better performance.
        """
        bonds = []
        atoms = list(residue.atoms.values())

        if len(atoms) < 2:
            return bonds

        # Collect coordinates and names
        coords = np.array([atom.coords for atom in atoms], dtype=np.float32)
        names = [atom.name for atom in atoms]

        # Use optimized symmetric pair finding
        pairs = find_all_pairs_within_distance_symmetric(coords, self.covalent_threshold)

        for i, j, dist in pairs:
            bonds.append(Bond(
                atom1_chain=chain_id,
                atom1_residue_idx=residue_idx,
                atom1_name=names[i],
                atom2_chain=chain_id,
                atom2_residue_idx=residue_idx,
                atom2_name=names[j],
                distance=dist,
            ))

        return bonds
    
    def _detect_backbone_bond(
        self,
        chain_id_1: str,
        res_idx_1: int,
        residue1: Residue,
        chain_id_2: str,
        res_idx_2: int,
        residue2: Residue,
        chain_type: Optional[ChainType],
    ) -> Optional[Bond]:
        """Detect backbone bond between consecutive residues."""
        if chain_type == ChainType.PROTEIN:
            atom1_name, atom2_name = PROTEIN_PEPTIDE_BOND
        elif chain_type in {ChainType.RNA, ChainType.DNA}:
            atom1_name, atom2_name = NUCLEIC_PHOSPHODIESTER
        else:
            return None
        
        atom1 = residue1.atoms.get(atom1_name)
        atom2 = residue2.atoms.get(atom2_name)
        
        if atom1 is None or atom2 is None:
            return None
        
        dist = np.linalg.norm(atom1.coords - atom2.coords)
        
        if dist <= self.covalent_threshold:
            return Bond(
                atom1_chain=chain_id_1,
                atom1_residue_idx=res_idx_1,
                atom1_name=atom1_name,
                atom2_chain=chain_id_2,
                atom2_residue_idx=res_idx_2,
                atom2_name=atom2_name,
                is_polymer_bond=True,
                distance=float(dist),
            )
        
        return None
    
    def _detect_cross_chain_bonds(
        self,
        chain_id_1: str,
        chain1: Chain,
        chain_id_2: str,
        chain2: Chain,
    ) -> List[Bond]:
        """Detect bonds between two chains.

        Uses KD-tree for O(n log m) complexity instead of O(n*m).
        """
        bonds = []

        is_polymer_1 = chain1.chain_type in {ChainType.PROTEIN, ChainType.RNA, ChainType.DNA}
        is_polymer_2 = chain2.chain_type in {ChainType.PROTEIN, ChainType.RNA, ChainType.DNA}

        # AF3 focuses on polymer-ligand and ligand-ligand bonds
        if is_polymer_1 and is_polymer_2:
            # Skip polymer-polymer (disulfides handled separately if needed)
            return bonds

        # Collect all atom coordinates and info for each chain
        coords1 = []
        info1 = []  # (residue_idx, atom_name)
        for i, res in enumerate(chain1.residues):
            for atom in res.atoms.values():
                coords1.append(atom.coords)
                info1.append((i, atom.name))

        coords2 = []
        info2 = []
        for j, res in enumerate(chain2.residues):
            for atom in res.atoms.values():
                coords2.append(atom.coords)
                info2.append((j, atom.name))

        if not coords1 or not coords2:
            return bonds

        coords1 = np.array(coords1, dtype=np.float32)
        coords2 = np.array(coords2, dtype=np.float32)

        # Use optimized KD-tree based detection
        bond_candidates = detect_cross_chain_bonds_fast(
            coords1, coords2, info1, info2,
            chain_id_1, chain_id_2,
            threshold=self.covalent_threshold,
        )

        is_ligand = not is_polymer_1 or not is_polymer_2

        for bc in bond_candidates:
            bonds.append(Bond(
                atom1_chain=chain_id_1,
                atom1_residue_idx=bc.residue1_idx,
                atom1_name=bc.atom1_name,
                atom2_chain=chain_id_2,
                atom2_residue_idx=bc.residue2_idx,
                atom2_name=bc.atom2_name,
                is_ligand_bond=is_ligand,
                distance=bc.distance,
            ))

        return bonds


class BondFeatureExtractor:
    """Extract bond features for AF3 model input.
    
    Generates the token_bonds feature matrix from detected bonds.
    
    From AF3 Table 5:
    "token_bonds [Ntoken, Ntoken]: A 2D matrix indicating if there is
    a bond between any atom in token i and token j"
    """
    
    def __init__(
        self,
        bond_threshold: float = COVALENT_BOND_THRESHOLD,
        include_polymer_backbone: bool = False,
    ):
        """Initialize feature extractor.
        
        Args:
            bond_threshold: Maximum distance for covalent bond
            include_polymer_backbone: Include backbone connectivity
        """
        self.detector = BondDetector(
            covalent_threshold=bond_threshold,
            include_polymer_backbone=include_polymer_backbone,
        )
    
    def extract(
        self,
        structure: Structure,
        token_to_residue: Dict[int, Tuple[str, int]],  # token_idx -> (chain_id, res_idx)
        num_tokens: int,
    ) -> BondFeatures:
        """Extract bond features.
        
        Args:
            structure: Structure to process
            token_to_residue: Mapping from token index to (chain_id, residue_idx)
            num_tokens: Total number of tokens
            
        Returns:
            BondFeatures object
        """
        # Detect atomic bonds
        atom_bonds = self.detector.detect_bonds(structure)
        
        # Build reverse mapping: (chain_id, res_idx) -> [token_indices]
        residue_to_tokens: Dict[Tuple[str, int], List[int]] = {}
        for token_idx, (chain_id, res_idx) in token_to_residue.items():
            key = (chain_id, res_idx)
            if key not in residue_to_tokens:
                residue_to_tokens[key] = []
            residue_to_tokens[key].append(token_idx)
        
        # Build token bond matrix
        token_bonds = np.zeros((num_tokens, num_tokens), dtype=np.float32)
        
        num_polymer = 0
        num_ligand = 0
        num_cross = 0
        
        for bond in atom_bonds:
            # Get token indices for both atoms
            key1 = (bond.atom1_chain, bond.atom1_residue_idx)
            key2 = (bond.atom2_chain, bond.atom2_residue_idx)
            
            tokens1 = residue_to_tokens.get(key1, [])
            tokens2 = residue_to_tokens.get(key2, [])
            
            # Mark all token pairs as bonded
            for t1 in tokens1:
                for t2 in tokens2:
                    token_bonds[t1, t2] = 1.0
                    token_bonds[t2, t1] = 1.0
            
            # Count bond types
            if bond.is_polymer_bond:
                num_polymer += 1
            elif bond.is_ligand_bond:
                num_ligand += 1
            else:
                num_cross += 1
        
        return BondFeatures(
            token_bonds=token_bonds,
            atom_bonds=atom_bonds,
            num_polymer_bonds=num_polymer,
            num_ligand_bonds=num_ligand,
            num_cross_bonds=num_cross,
        )
    
    def extract_from_tokenized(
        self,
        structure: Structure,
        tokenized,  # TokenizedStructure
    ) -> BondFeatures:
        """Extract bond features from tokenized structure.
        
        Args:
            structure: Original structure
            tokenized: Tokenized structure with token mapping
            
        Returns:
            BondFeatures object
        """
        # Build token to residue mapping from tokenized structure
        token_to_residue = {}
        for i, token in enumerate(tokenized.tokens):
            token_to_residue[i] = (token.chain_id, token.residue_index)
        
        return self.extract(
            structure,
            token_to_residue,
            len(tokenized.tokens),
        )


class DisulfideBondDetector:
    """Detect disulfide bonds between cysteine residues.
    
    Specialized detector for S-S bonds which have longer
    typical distances (~2.05 Å).
    """
    
    DISULFIDE_MIN = 1.8  # Å
    DISULFIDE_MAX = 2.3  # Å
    
    def detect(self, structure: Structure) -> List[Bond]:
        """Detect disulfide bonds.
        
        Args:
            structure: Structure to analyze
            
        Returns:
            List of disulfide bonds
        """
        bonds = []
        
        # Collect all cysteine SG atoms
        cys_atoms: List[Tuple[str, int, Atom]] = []
        
        for chain_id, chain in structure.chains.items():
            if chain.chain_type != ChainType.PROTEIN:
                continue
            
            for i, residue in enumerate(chain.residues):
                if residue.name in {"CYS", "CYX"}:  # CYX = disulfide cysteine
                    sg = residue.atoms.get("SG")
                    if sg is not None:
                        cys_atoms.append((chain_id, i, sg))
        
        # Check all pairs
        for i, (chain1, res1, sg1) in enumerate(cys_atoms):
            for chain2, res2, sg2 in cys_atoms[i + 1:]:
                dist = np.linalg.norm(sg1.coords - sg2.coords)
                
                if self.DISULFIDE_MIN <= dist <= self.DISULFIDE_MAX:
                    bonds.append(Bond(
                        atom1_chain=chain1,
                        atom1_residue_idx=res1,
                        atom1_name="SG",
                        atom2_chain=chain2,
                        atom2_residue_idx=res2,
                        atom2_name="SG",
                        bond_order=1,
                        is_polymer_bond=True,
                        distance=float(dist),
                    ))
        
        return bonds


class GlycosidicBondDetector:
    """Detect glycosidic bonds for carbohydrate chains.
    
    From AF3 Section 2.5: Glycans are bonded to proteins
    through glycosidic linkages.
    """
    
    # Common glycosidic bond atoms
    GLYCOSIDIC_ATOMS = {"C1", "O1", "C2", "O2", "C3", "O3", "C4", "O4", "C6", "O6"}
    
    # Protein atoms that can be glycosylated
    PROTEIN_GLYCO_ATOMS = {
        "ND2",  # Asparagine (N-linked)
        "OG",   # Serine (O-linked)
        "OG1",  # Threonine (O-linked)
    }
    
    GLYCO_BOND_THRESHOLD = 1.6  # Å, tighter for C-O bonds
    
    def detect(self, structure: Structure) -> List[Bond]:
        """Detect glycosidic bonds.
        
        Args:
            structure: Structure to analyze
            
        Returns:
            List of glycosidic bonds
        """
        bonds = []
        
        # Find glycan chains (ligand chains with known glycan residues)
        glycan_atoms: List[Tuple[str, int, Atom]] = []
        protein_atoms: List[Tuple[str, int, Atom]] = []
        
        for chain_id, chain in structure.chains.items():
            if chain.chain_type == ChainType.LIGAND:
                for i, residue in enumerate(chain.residues):
                    for atom in residue.atoms.values():
                        if atom.name in self.GLYCOSIDIC_ATOMS:
                            glycan_atoms.append((chain_id, i, atom))
            
            elif chain.chain_type == ChainType.PROTEIN:
                for i, residue in enumerate(chain.residues):
                    for atom in residue.atoms.values():
                        if atom.name in self.PROTEIN_GLYCO_ATOMS:
                            protein_atoms.append((chain_id, i, atom))
        
        # Check glycan-protein bonds
        for chain1, res1, atom1 in glycan_atoms:
            for chain2, res2, atom2 in protein_atoms:
                dist = np.linalg.norm(atom1.coords - atom2.coords)
                
                if dist <= self.GLYCO_BOND_THRESHOLD:
                    bonds.append(Bond(
                        atom1_chain=chain1,
                        atom1_residue_idx=res1,
                        atom1_name=atom1.name,
                        atom2_chain=chain2,
                        atom2_residue_idx=res2,
                        atom2_name=atom2.name,
                        is_ligand_bond=True,
                        distance=float(dist),
                    ))
        
        # Check glycan-glycan bonds
        for i, (chain1, res1, atom1) in enumerate(glycan_atoms):
            for chain2, res2, atom2 in glycan_atoms[i + 1:]:
                if chain1 == chain2 and abs(res1 - res2) <= 1:
                    # Same or adjacent residue in same chain
                    dist = np.linalg.norm(atom1.coords - atom2.coords)
                    
                    if dist <= self.GLYCO_BOND_THRESHOLD:
                        bonds.append(Bond(
                            atom1_chain=chain1,
                            atom1_residue_idx=res1,
                            atom1_name=atom1.name,
                            atom2_chain=chain2,
                            atom2_residue_idx=res2,
                            atom2_name=atom2.name,
                            is_ligand_bond=True,
                            distance=float(dist),
                        ))
        
        return bonds


def extract_bond_features(
    structure: Structure,
    tokenized,  # TokenizedStructure
    include_disulfides: bool = True,
    include_glycosidic: bool = True,
) -> BondFeatures:
    """Convenience function to extract all bond features.
    
    Args:
        structure: Structure to process
        tokenized: Tokenized structure
        include_disulfides: Include disulfide bonds
        include_glycosidic: Include glycosidic bonds
        
    Returns:
        Complete BondFeatures
    """
    extractor = BondFeatureExtractor()
    features = extractor.extract_from_tokenized(structure, tokenized)
    
    # Add specialized bonds
    additional_bonds = []
    
    if include_disulfides:
        disulfide_detector = DisulfideBondDetector()
        additional_bonds.extend(disulfide_detector.detect(structure))
    
    if include_glycosidic:
        glyco_detector = GlycosidicBondDetector()
        additional_bonds.extend(glyco_detector.detect(structure))
    
    # Update features with additional bonds
    if additional_bonds:
        features.atom_bonds.extend(additional_bonds)
        
        # Update token bond matrix
        token_to_residue = {}
        for i, token in enumerate(tokenized.tokens):
            token_to_residue[(token.chain_id, token.residue_idx)] = i
        
        for bond in additional_bonds:
            key1 = (bond.atom1_chain, bond.atom1_residue_idx)
            key2 = (bond.atom2_chain, bond.atom2_residue_idx)
            
            t1 = token_to_residue.get(key1)
            t2 = token_to_residue.get(key2)
            
            if t1 is not None and t2 is not None:
                features.token_bonds[t1, t2] = 1.0
                features.token_bonds[t2, t1] = 1.0
    
    return features
