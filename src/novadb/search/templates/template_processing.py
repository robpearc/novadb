"""Template processing for AlphaFold3-style structure prediction.

Implements template processing from AF3 Supplement Section 2.4:
- Template HMM building from UniRef90 MSAs
- Template date filtering for training/inference separation
- Template deduplication by sequence and structure
- Template feature extraction (coordinates, pseudo-beta, masks)

Reference: Abramson et al. 2024, Nature.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Constants from AF3 Section 2.4
MAX_TEMPLATES = 4
DEFAULT_MIN_IDENTITY = 0.1
DEFAULT_MAX_IDENTITY = 0.95
DEFAULT_MIN_COVERAGE = 0.1


@dataclass(frozen=True)
class TemplateHMMConfig:
    """Configuration for template HMM building.
    
    From AF3 Section 2.4: Build HMM from UniRef90 MSA for template search.
    
    Attributes:
        hmmbuild_binary: Path to hmmbuild executable.
        use_msa_weights: Whether to use sequence weights in HMM.
        effective_sequence_number: Effective number of sequences.
        amino_acid_alphabet: Whether to use amino acid alphabet.
    """
    
    hmmbuild_binary: Path = field(default_factory=lambda: Path("hmmbuild"))
    use_msa_weights: bool = True
    effective_sequence_number: Optional[float] = None
    amino_acid_alphabet: bool = True


@dataclass(frozen=True)
class DateFilterConfig:
    """Configuration for template date filtering.
    
    From AF3 Section 2.4: Filter templates by release date to prevent
    data leakage during training.
    
    Attributes:
        max_release_date: Maximum allowed release date.
        date_source: Source of release date ('mmcif', 'obsolete_file', 'api').
        allow_missing_dates: Whether to allow templates with unknown dates.
        obsolete_file_path: Path to obsolete entries file.
    """
    
    max_release_date: Optional[date] = None
    date_source: str = "mmcif"
    allow_missing_dates: bool = False
    obsolete_file_path: Optional[Path] = None


@dataclass(frozen=True)
class DeduplicationConfig:
    """Configuration for template deduplication.
    
    Attributes:
        sequence_identity_threshold: Maximum identity for sequence dedup.
        structural_rmsd_threshold: Maximum RMSD for structural dedup.
        use_structural_dedup: Whether to use structural deduplication.
        prefer_higher_resolution: When deduplicating, prefer higher resolution.
    """
    
    sequence_identity_threshold: float = 0.95
    structural_rmsd_threshold: float = 1.0
    use_structural_dedup: bool = False
    prefer_higher_resolution: bool = True


@dataclass(frozen=True)
class TemplateFeatureConfig:
    """Configuration for template feature extraction.
    
    From AF3 Table 5: Features extracted from templates.
    
    Attributes:
        include_coordinates: Extract atom coordinates.
        include_pseudo_beta: Extract pseudo-Cβ positions.
        include_backbone_mask: Extract backbone atom mask.
        include_distances: Extract pairwise distances.
        distance_bins: Number of distance bins for discretization.
        max_distance: Maximum distance for binning.
    """
    
    include_coordinates: bool = True
    include_pseudo_beta: bool = True
    include_backbone_mask: bool = True
    include_distances: bool = True
    distance_bins: int = 64
    max_distance: float = 22.0


@dataclass
class TemplateHit:
    """A single template hit from search.
    
    Attributes:
        template_id: Unique identifier (e.g., '1abc_A').
        pdb_id: PDB ID (e.g., '1abc').
        chain_id: Chain identifier (e.g., 'A').
        sequence: Template sequence.
        aligned_query: Query sequence aligned to template.
        aligned_template: Template sequence aligned to query.
        query_start: Start position in query (0-indexed).
        query_end: End position in query (exclusive).
        template_start: Start position in template (0-indexed).
        template_end: End position in template (exclusive).
        e_value: E-value from search.
        score: Alignment score.
        identity: Sequence identity (0-1).
        coverage: Query coverage (0-1).
        release_date: Structure release date.
        resolution: Structure resolution in Angstroms.
    """
    
    template_id: str
    pdb_id: str
    chain_id: str
    sequence: str
    aligned_query: str
    aligned_template: str
    query_start: int
    query_end: int
    template_start: int
    template_end: int
    e_value: float
    score: float
    identity: float
    coverage: float = 0.0
    release_date: Optional[date] = None
    resolution: Optional[float] = None


@dataclass
class TemplateFeatures:
    """Extracted features from a template.
    
    From AF3 Table 5: Template features for structure prediction.
    
    Attributes:
        template_aatype: Residue types (Nres,).
        template_all_atom_positions: Atom positions (Nres, 37, 3).
        template_all_atom_mask: Atom mask (Nres, 37).
        template_pseudo_beta: Pseudo-Cβ positions (Nres, 3).
        template_pseudo_beta_mask: Pseudo-Cβ mask (Nres,).
        template_backbone_frame_mask: Backbone frame validity (Nres,).
        template_distogram: Pairwise distances (Nres, Nres, Nbins).
        template_unit_vector: Unit vectors between residues (Nres, Nres, 3).
        query_to_template_mapping: Mapping from query to template positions.
    """
    
    template_aatype: np.ndarray
    template_all_atom_positions: np.ndarray
    template_all_atom_mask: np.ndarray
    template_pseudo_beta: np.ndarray
    template_pseudo_beta_mask: np.ndarray
    template_backbone_frame_mask: np.ndarray
    template_distogram: Optional[np.ndarray] = None
    template_unit_vector: Optional[np.ndarray] = None
    query_to_template_mapping: Optional[np.ndarray] = None


@dataclass
class TemplateHMMBuilder:
    """Builds HMMs from MSAs for template search.
    
    From AF3 Section 2.4: Build profile HMM from UniRef90 MSA
    for searching against PDB sequences.
    
    Attributes:
        config: HMM building configuration.
    """
    
    config: TemplateHMMConfig = field(default_factory=TemplateHMMConfig)
    
    def build_hmm(
        self,
        msa_path: Path,
        output_hmm_path: Path,
        name: str = "query",
    ) -> Path:
        """Build HMM from MSA file.
        
        Args:
            msa_path: Path to MSA in A3M or Stockholm format.
            output_hmm_path: Path for output HMM file.
            name: Name for the HMM.
            
        Returns:
            Path to the created HMM file.
            
        Raises:
            RuntimeError: If hmmbuild fails.
        """
        cmd = [str(self.config.hmmbuild_binary)]
        
        if self.config.amino_acid_alphabet:
            cmd.append("--amino")
        
        if self.config.use_msa_weights:
            cmd.append("--wpb")  # Position-based sequence weighting
        
        if self.config.effective_sequence_number is not None:
            cmd.extend(["--eset", str(self.config.effective_sequence_number)])
        
        cmd.extend(["-n", name])
        cmd.append(str(output_hmm_path))
        cmd.append(str(msa_path))
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"hmmbuild failed: {result.stderr}")
        
        logger.debug("Built HMM: %s", output_hmm_path)
        return output_hmm_path
    
    def build_from_sequences(
        self,
        sequences: Sequence[Tuple[str, str]],
        output_hmm_path: Path,
        name: str = "query",
    ) -> Path:
        """Build HMM from a list of sequences.
        
        Args:
            sequences: List of (description, sequence) tuples.
            output_hmm_path: Path for output HMM file.
            name: Name for the HMM.
            
        Returns:
            Path to the created HMM file.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".a3m",
            delete=False,
        ) as f:
            for desc, seq in sequences:
                f.write(f">{desc}\n{seq}\n")
            msa_path = Path(f.name)
        
        try:
            return self.build_hmm(msa_path, output_hmm_path, name)
        finally:
            msa_path.unlink(missing_ok=True)
    
    def build_from_uniref90_msa(
        self,
        uniref90_msa_path: Path,
        output_hmm_path: Path,
    ) -> Path:
        """Build HMM from UniRef90 MSA.
        
        From AF3 Section 2.4: The template search uses an HMM built
        from the UniRef90 MSA.
        
        Args:
            uniref90_msa_path: Path to UniRef90 MSA file.
            output_hmm_path: Path for output HMM.
            
        Returns:
            Path to the created HMM file.
        """
        return self.build_hmm(
            uniref90_msa_path,
            output_hmm_path,
            name="uniref90_profile",
        )


@dataclass
class TemplateDateFilter:
    """Filters templates by release date.
    
    From AF3 Section 2.4: Filter templates to prevent data leakage
    during training by excluding structures released after a cutoff date.
    
    Attributes:
        config: Date filtering configuration.
        _release_date_cache: Cached release dates.
        _obsolete_entries: Set of obsolete PDB entries.
    """
    
    config: DateFilterConfig = field(default_factory=DateFilterConfig)
    _release_date_cache: Dict[str, date] = field(default_factory=dict, repr=False)
    _obsolete_entries: Optional[FrozenSet[str]] = field(default=None, repr=False)
    
    def filter(
        self,
        hits: Sequence[TemplateHit],
        max_date: Optional[date] = None,
    ) -> List[TemplateHit]:
        """Filter templates by release date.
        
        Args:
            hits: Template hits to filter.
            max_date: Maximum release date (overrides config).
            
        Returns:
            List of hits passing the date filter.
        """
        cutoff_date = max_date or self.config.max_release_date
        
        if cutoff_date is None:
            return list(hits)
        
        filtered: list[TemplateHit] = []
        
        for hit in hits:
            release = hit.release_date or self._get_release_date(hit.pdb_id)
            
            if release is None:
                if self.config.allow_missing_dates:
                    filtered.append(hit)
                continue
            
            if release <= cutoff_date:
                # Update hit with release date
                if hit.release_date is None:
                    hit = TemplateHit(
                        template_id=hit.template_id,
                        pdb_id=hit.pdb_id,
                        chain_id=hit.chain_id,
                        sequence=hit.sequence,
                        aligned_query=hit.aligned_query,
                        aligned_template=hit.aligned_template,
                        query_start=hit.query_start,
                        query_end=hit.query_end,
                        template_start=hit.template_start,
                        template_end=hit.template_end,
                        e_value=hit.e_value,
                        score=hit.score,
                        identity=hit.identity,
                        coverage=hit.coverage,
                        release_date=release,
                        resolution=hit.resolution,
                    )
                filtered.append(hit)
        
        logger.debug(
            "Date filter: %d -> %d hits (cutoff: %s)",
            len(hits),
            len(filtered),
            cutoff_date,
        )
        return filtered
    
    def filter_obsolete(
        self,
        hits: Sequence[TemplateHit],
    ) -> List[TemplateHit]:
        """Remove obsolete PDB entries.
        
        Args:
            hits: Template hits to filter.
            
        Returns:
            List of hits with obsolete entries removed.
        """
        obsolete = self._load_obsolete_entries()
        
        if not obsolete:
            return list(hits)
        
        filtered = [
            hit for hit in hits
            if hit.pdb_id.upper() not in obsolete
        ]
        
        logger.debug(
            "Obsolete filter: %d -> %d hits",
            len(hits),
            len(filtered),
        )
        return filtered
    
    def _get_release_date(self, pdb_id: str) -> Optional[date]:
        """Get release date for a PDB entry.
        
        Args:
            pdb_id: 4-character PDB ID.
            
        Returns:
            Release date or None if unknown.
        """
        pdb_id = pdb_id.lower()
        
        if pdb_id in self._release_date_cache:
            return self._release_date_cache[pdb_id]
        
        # Would fetch from mmCIF or API in production
        return None
    
    def _load_obsolete_entries(self) -> FrozenSet[str]:
        """Load list of obsolete PDB entries.
        
        Returns:
            Set of obsolete PDB IDs.
        """
        if self._obsolete_entries is not None:
            return self._obsolete_entries
        
        if (
            self.config.obsolete_file_path is None
            or not self.config.obsolete_file_path.exists()
        ):
            self._obsolete_entries = frozenset()
            return self._obsolete_entries
        
        obsolete: set[str] = set()
        
        with open(self.config.obsolete_file_path) as f:
            for line in f:
                if line.startswith("OBSLTE"):
                    parts = line.split()
                    if len(parts) >= 3:
                        obsolete.add(parts[2].upper())
        
        self._obsolete_entries = frozenset(obsolete)
        logger.info("Loaded %d obsolete PDB entries", len(self._obsolete_entries))
        return self._obsolete_entries
    
    def add_release_dates(
        self,
        hits: Sequence[TemplateHit],
        mmcif_dir: Path,
    ) -> List[TemplateHit]:
        """Add release dates to hits by parsing mmCIF files.
        
        Args:
            hits: Template hits.
            mmcif_dir: Directory containing mmCIF files.
            
        Returns:
            Hits with release dates populated.
        """
        updated_hits: list[TemplateHit] = []
        
        for hit in hits:
            if hit.release_date is not None:
                updated_hits.append(hit)
                continue
            
            release = self._parse_release_date_from_mmcif(
                hit.pdb_id,
                mmcif_dir,
            )
            
            updated_hit = TemplateHit(
                template_id=hit.template_id,
                pdb_id=hit.pdb_id,
                chain_id=hit.chain_id,
                sequence=hit.sequence,
                aligned_query=hit.aligned_query,
                aligned_template=hit.aligned_template,
                query_start=hit.query_start,
                query_end=hit.query_end,
                template_start=hit.template_start,
                template_end=hit.template_end,
                e_value=hit.e_value,
                score=hit.score,
                identity=hit.identity,
                coverage=hit.coverage,
                release_date=release,
                resolution=hit.resolution,
            )
            updated_hits.append(updated_hit)
        
        return updated_hits
    
    def _parse_release_date_from_mmcif(
        self,
        pdb_id: str,
        mmcif_dir: Path,
    ) -> Optional[date]:
        """Parse release date from mmCIF file.
        
        Args:
            pdb_id: PDB identifier.
            mmcif_dir: Directory containing mmCIF files.
            
        Returns:
            Release date or None.
        """
        pdb_id = pdb_id.lower()
        
        # Check cache first
        if pdb_id in self._release_date_cache:
            return self._release_date_cache[pdb_id]
        
        mmcif_path = mmcif_dir / f"{pdb_id}.cif"
        if not mmcif_path.exists():
            mmcif_path = mmcif_dir / f"{pdb_id.upper()}.cif"
        
        if not mmcif_path.exists():
            return None
        
        try:
            with open(mmcif_path) as f:
                for line in f:
                    # Try initial deposition date
                    if "_pdbx_database_status.recvd_initial_deposition_date" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            date_str = parts[-1].strip()
                            if date_str not in (".", "?"):
                                release = date.fromisoformat(date_str)
                                self._release_date_cache[pdb_id] = release
                                return release
                    
                    # Try revision date
                    if "_database_PDB_rev.date_original" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            date_str = parts[-1].strip()
                            if date_str not in (".", "?"):
                                release = date.fromisoformat(date_str)
                                self._release_date_cache[pdb_id] = release
                                return release
        except (OSError, ValueError) as e:
            logger.warning("Failed to parse release date for %s: %s", pdb_id, e)
        
        return None


@dataclass
class TemplateDeduplicator:
    """Removes duplicate template chains.
    
    From AF3 Section 2.4: Deduplicate templates to ensure diversity
    and prevent redundant computations.
    
    Attributes:
        config: Deduplication configuration.
    """
    
    config: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    
    def deduplicate(
        self,
        hits: Sequence[TemplateHit],
    ) -> List[TemplateHit]:
        """Remove duplicate templates by sequence identity.
        
        Args:
            hits: Template hits to deduplicate.
            
        Returns:
            Deduplicated list of hits.
        """
        if not hits:
            return []
        
        unique_hits: list[TemplateHit] = []
        seen_sequences: set[str] = set()
        
        # Sort by score descending to keep best hits
        sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
        
        for hit in sorted_hits:
            seq_hash = self._get_sequence_hash(hit.sequence)
            
            # Check if we've seen a very similar sequence
            is_duplicate = False
            for seen_hash in seen_sequences:
                if self._are_similar_hashes(seq_hash, seen_hash, hit.sequence, seen_sequences):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_sequences.add(seq_hash)
                unique_hits.append(hit)
        
        logger.debug(
            "Deduplication: %d -> %d hits",
            len(hits),
            len(unique_hits),
        )
        return unique_hits
    
    def deduplicate_by_pdb(
        self,
        hits: Sequence[TemplateHit],
    ) -> List[TemplateHit]:
        """Keep only one template per PDB entry.
        
        Args:
            hits: Template hits to deduplicate.
            
        Returns:
            List with one hit per PDB ID.
        """
        seen_pdbs: set[str] = set()
        unique_hits: list[TemplateHit] = []
        
        # Sort by score descending
        sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
        
        for hit in sorted_hits:
            pdb_id = hit.pdb_id.lower()
            if pdb_id not in seen_pdbs:
                seen_pdbs.add(pdb_id)
                unique_hits.append(hit)
        
        return unique_hits
    
    def deduplicate_by_sequence_cluster(
        self,
        hits: Sequence[TemplateHit],
        cluster_mapping: Mapping[str, str],
    ) -> List[TemplateHit]:
        """Keep one template per sequence cluster.
        
        Args:
            hits: Template hits.
            cluster_mapping: Mapping from template ID to cluster ID.
            
        Returns:
            List with one hit per cluster.
        """
        seen_clusters: set[str] = set()
        unique_hits: list[TemplateHit] = []
        
        # Sort by resolution if available, then by score
        def sort_key(h: TemplateHit) -> Tuple[float, float]:
            res = h.resolution if h.resolution is not None else float('inf')
            return (res, -h.score)
        
        sorted_hits = sorted(hits, key=sort_key)
        
        for hit in sorted_hits:
            cluster_id = cluster_mapping.get(hit.template_id, hit.template_id)
            
            if cluster_id not in seen_clusters:
                seen_clusters.add(cluster_id)
                unique_hits.append(hit)
        
        return unique_hits
    
    def _get_sequence_hash(self, sequence: str) -> str:
        """Get hash of sequence for comparison."""
        return hashlib.md5(sequence.upper().encode()).hexdigest()
    
    def _are_similar_hashes(
        self,
        hash1: str,
        hash2: str,
        seq1: str,
        seen_sequences: set[str],
    ) -> bool:
        """Check if two sequences are similar enough to be duplicates.
        
        For now, just check hash equality. In production, would compute
        actual sequence identity.
        """
        return hash1 == hash2


@dataclass
class TemplateFeatureExtractor:
    """Extracts features from template structures.
    
    From AF3 Table 5: Extract template features for structure prediction.
    
    Attributes:
        config: Feature extraction configuration.
    """
    
    config: TemplateFeatureConfig = field(default_factory=TemplateFeatureConfig)
    
    # Atom indices for 37-atom representation
    ATOM_ORDER = [
        "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1",
        "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
        "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
        "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
    ]
    
    RESIDUE_TYPES = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "UNK",
    ]
    
    def extract_features(
        self,
        hit: TemplateHit,
        query_length: int,
        structure_coords: np.ndarray,
        structure_mask: np.ndarray,
        structure_aatype: np.ndarray,
    ) -> TemplateFeatures:
        """Extract features from a template hit.
        
        Args:
            hit: Template hit with alignment information.
            query_length: Length of the query sequence.
            structure_coords: Template atom coordinates (Ntemplate, 37, 3).
            structure_mask: Template atom mask (Ntemplate, 37).
            structure_aatype: Template residue types (Ntemplate,).
            
        Returns:
            Extracted template features.
        """
        # Initialize arrays with query length
        template_aatype = np.zeros(query_length, dtype=np.int32)
        template_all_atom_positions = np.zeros(
            (query_length, 37, 3),
            dtype=np.float32,
        )
        template_all_atom_mask = np.zeros(
            (query_length, 37),
            dtype=np.float32,
        )
        
        # Map aligned positions
        query_to_template = self._build_alignment_mapping(hit)
        
        for query_idx, template_idx in enumerate(query_to_template):
            if template_idx >= 0 and template_idx < len(structure_coords):
                template_aatype[query_idx] = structure_aatype[template_idx]
                template_all_atom_positions[query_idx] = structure_coords[template_idx]
                template_all_atom_mask[query_idx] = structure_mask[template_idx]
        
        # Extract pseudo-beta positions
        pseudo_beta, pseudo_beta_mask = self._compute_pseudo_beta(
            template_all_atom_positions,
            template_all_atom_mask,
            template_aatype,
        )
        
        # Compute backbone frame mask
        backbone_frame_mask = self._compute_backbone_frame_mask(
            template_all_atom_positions,
            template_all_atom_mask,
        )
        
        # Compute optional features
        distogram = None
        unit_vector = None
        
        if self.config.include_distances:
            distogram = self._compute_distogram(pseudo_beta, pseudo_beta_mask)
        
        return TemplateFeatures(
            template_aatype=template_aatype,
            template_all_atom_positions=template_all_atom_positions,
            template_all_atom_mask=template_all_atom_mask,
            template_pseudo_beta=pseudo_beta,
            template_pseudo_beta_mask=pseudo_beta_mask,
            template_backbone_frame_mask=backbone_frame_mask,
            template_distogram=distogram,
            template_unit_vector=unit_vector,
            query_to_template_mapping=query_to_template,
        )
    
    def extract_from_mmcif(
        self,
        hit: TemplateHit,
        query_length: int,
        mmcif_path: Path,
    ) -> Optional[TemplateFeatures]:
        """Extract features from mmCIF file.
        
        Args:
            hit: Template hit.
            query_length: Query sequence length.
            mmcif_path: Path to mmCIF file.
            
        Returns:
            Template features or None if parsing fails.
        """
        try:
            coords, mask, aatype = self._parse_mmcif_coordinates(
                mmcif_path,
                hit.chain_id,
            )
            return self.extract_features(
                hit,
                query_length,
                coords,
                mask,
                aatype,
            )
        except Exception as e:
            logger.warning(
                "Failed to extract features from %s: %s",
                mmcif_path,
                e,
            )
            return None
    
    def _build_alignment_mapping(
        self,
        hit: TemplateHit,
    ) -> np.ndarray:
        """Build mapping from query positions to template positions.
        
        Args:
            hit: Template hit with alignment.
            
        Returns:
            Array mapping query index to template index (-1 for gaps).
        """
        aligned_query = hit.aligned_query
        aligned_template = hit.aligned_template
        
        if len(aligned_query) != len(aligned_template):
            raise ValueError("Aligned sequences must have same length")
        
        mapping: list[int] = []
        query_idx = hit.query_start
        template_idx = hit.template_start
        
        for q_char, t_char in zip(aligned_query, aligned_template):
            if q_char != "-":
                if t_char != "-":
                    mapping.append(template_idx)
                else:
                    mapping.append(-1)
                query_idx += 1
            
            if t_char != "-":
                template_idx += 1
        
        return np.array(mapping, dtype=np.int32)
    
    def _compute_pseudo_beta(
        self,
        positions: np.ndarray,
        mask: np.ndarray,
        aatype: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pseudo-Cβ positions.
        
        From AF3: Use Cβ for most residues, Cα for glycine.
        
        Args:
            positions: Atom positions (N, 37, 3).
            mask: Atom mask (N, 37).
            aatype: Residue types (N,).
            
        Returns:
            Tuple of (pseudo_beta positions, pseudo_beta mask).
        """
        n_residues = positions.shape[0]
        pseudo_beta = np.zeros((n_residues, 3), dtype=np.float32)
        pseudo_beta_mask = np.zeros(n_residues, dtype=np.float32)
        
        ca_idx = self.ATOM_ORDER.index("CA")
        cb_idx = self.ATOM_ORDER.index("CB")
        gly_idx = self.RESIDUE_TYPES.index("GLY") if "GLY" in self.RESIDUE_TYPES else -1
        
        for i in range(n_residues):
            # Use CB for non-glycine, CA for glycine
            if aatype[i] == gly_idx:
                atom_idx = ca_idx
            else:
                atom_idx = cb_idx
                # Fall back to CA if CB not present
                if mask[i, cb_idx] == 0 and mask[i, ca_idx] > 0:
                    atom_idx = ca_idx
            
            if mask[i, atom_idx] > 0:
                pseudo_beta[i] = positions[i, atom_idx]
                pseudo_beta_mask[i] = 1.0
        
        return pseudo_beta, pseudo_beta_mask
    
    def _compute_backbone_frame_mask(
        self,
        positions: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Compute backbone frame validity mask.
        
        From AF3: A residue has a valid backbone frame if N, CA, C are all present.
        
        Args:
            positions: Atom positions (N, 37, 3).
            mask: Atom mask (N, 37).
            
        Returns:
            Backbone frame mask (N,).
        """
        n_idx = self.ATOM_ORDER.index("N")
        ca_idx = self.ATOM_ORDER.index("CA")
        c_idx = self.ATOM_ORDER.index("C")
        
        frame_mask = (
            (mask[:, n_idx] > 0)
            & (mask[:, ca_idx] > 0)
            & (mask[:, c_idx] > 0)
        ).astype(np.float32)
        
        return frame_mask
    
    def _compute_distogram(
        self,
        pseudo_beta: np.ndarray,
        pseudo_beta_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise distance matrix (discretized).
        
        Args:
            pseudo_beta: Pseudo-Cβ positions (N, 3).
            pseudo_beta_mask: Validity mask (N,).
            
        Returns:
            Distance histogram (N, N, num_bins).
        """
        n_residues = pseudo_beta.shape[0]
        n_bins = self.config.distance_bins
        max_dist = self.config.max_distance
        
        # Compute pairwise distances
        diff = pseudo_beta[:, None, :] - pseudo_beta[None, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Create mask for valid pairs
        pair_mask = pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
        
        # Discretize distances
        bin_edges = np.linspace(0, max_dist, n_bins + 1)
        distogram = np.zeros((n_residues, n_residues, n_bins), dtype=np.float32)
        
        for i in range(n_bins):
            in_bin = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            distogram[:, :, i] = in_bin.astype(np.float32) * pair_mask
        
        # Last bin includes everything beyond max_dist
        distogram[:, :, -1] += (distances >= max_dist).astype(np.float32) * pair_mask
        
        return distogram
    
    def _parse_mmcif_coordinates(
        self,
        mmcif_path: Path,
        chain_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse coordinates from mmCIF file.
        
        Args:
            mmcif_path: Path to mmCIF file.
            chain_id: Chain to extract.
            
        Returns:
            Tuple of (coordinates, mask, aatype).
        """
        # Simplified parsing - would use proper mmCIF parser in production
        residues: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)
        residue_names: Dict[int, str] = {}
        
        with open(mmcif_path) as f:
            in_atom_site = False
            header_line = ""
            
            for line in f:
                if line.startswith("_atom_site."):
                    in_atom_site = True
                    header_line += line
                    continue
                
                if in_atom_site and line.startswith("ATOM") or line.startswith("HETATM"):
                    # Parse atom line (simplified)
                    parts = line.split()
                    if len(parts) < 15:
                        continue
                    
                    auth_chain = parts[6] if len(parts) > 6 else ""
                    if auth_chain != chain_id:
                        continue
                    
                    atom_name = parts[3]
                    res_name = parts[5]
                    res_seq = int(parts[8]) if parts[8].isdigit() else 0
                    x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                    
                    if atom_name in self.ATOM_ORDER:
                        residues[res_seq][atom_name] = np.array([x, y, z], dtype=np.float32)
                        residue_names[res_seq] = res_name
                
                if in_atom_site and line.strip() == "#":
                    break
        
        if not residues:
            return (
                np.zeros((0, 37, 3), dtype=np.float32),
                np.zeros((0, 37), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
            )
        
        # Convert to arrays
        seq_ids = sorted(residues.keys())
        n_residues = len(seq_ids)
        
        coords = np.zeros((n_residues, 37, 3), dtype=np.float32)
        mask = np.zeros((n_residues, 37), dtype=np.float32)
        aatype = np.zeros(n_residues, dtype=np.int32)
        
        for i, seq_id in enumerate(seq_ids):
            res_atoms = residues[seq_id]
            res_name = residue_names.get(seq_id, "UNK")
            
            # Get residue type
            if res_name in self.RESIDUE_TYPES:
                aatype[i] = self.RESIDUE_TYPES.index(res_name)
            else:
                aatype[i] = self.RESIDUE_TYPES.index("UNK")
            
            # Fill coordinates
            for atom_name, atom_coords in res_atoms.items():
                atom_idx = self.ATOM_ORDER.index(atom_name)
                coords[i, atom_idx] = atom_coords
                mask[i, atom_idx] = 1.0
        
        return coords, mask, aatype


@dataclass(frozen=True)
class TemplateProcessingPipeline:
    """Complete template processing pipeline.
    
    Implements the full AF3 template processing workflow from Section 2.4.
    
    Attributes:
        hmm_builder: HMM builder for template search.
        date_filter: Date filtering for training separation.
        deduplicator: Template deduplication.
        feature_extractor: Template feature extraction.
        mmcif_dir: Directory containing mmCIF files.
        max_templates: Maximum templates to return.
    """
    
    hmm_builder: TemplateHMMBuilder = field(default_factory=TemplateHMMBuilder)
    date_filter: TemplateDateFilter = field(default_factory=TemplateDateFilter)
    deduplicator: TemplateDeduplicator = field(default_factory=TemplateDeduplicator)
    feature_extractor: TemplateFeatureExtractor = field(default_factory=TemplateFeatureExtractor)
    mmcif_dir: Optional[Path] = None
    max_templates: int = MAX_TEMPLATES
    
    def process_hits(
        self,
        hits: Sequence[TemplateHit],
        max_date: Optional[date] = None,
    ) -> List[TemplateHit]:
        """Process template hits through filtering pipeline.
        
        Args:
            hits: Raw template hits from search.
            max_date: Maximum release date for templates.
            
        Returns:
            Processed and filtered hits.
        """
        # Step 1: Add release dates if mmCIF directory available
        if self.mmcif_dir is not None:
            hits = self.date_filter.add_release_dates(hits, self.mmcif_dir)
        
        # Step 2: Filter by date
        filtered = self.date_filter.filter(hits, max_date)
        
        # Step 3: Remove obsolete entries
        filtered = self.date_filter.filter_obsolete(filtered)
        
        # Step 4: Deduplicate
        filtered = self.deduplicator.deduplicate(filtered)
        
        # Step 5: Take top templates
        filtered = filtered[:self.max_templates]
        
        return filtered
    
    def extract_all_features(
        self,
        hits: Sequence[TemplateHit],
        query_length: int,
    ) -> List[TemplateFeatures]:
        """Extract features from all template hits.
        
        Args:
            hits: Template hits.
            query_length: Query sequence length.
            
        Returns:
            List of template features.
        """
        if self.mmcif_dir is None:
            logger.warning("mmCIF directory not set, cannot extract features")
            return []
        
        features_list: list[TemplateFeatures] = []
        
        for hit in hits:
            mmcif_path = self.mmcif_dir / f"{hit.pdb_id.lower()}.cif"
            
            if not mmcif_path.exists():
                logger.warning("mmCIF not found: %s", mmcif_path)
                continue
            
            features = self.feature_extractor.extract_from_mmcif(
                hit,
                query_length,
                mmcif_path,
            )
            
            if features is not None:
                features_list.append(features)
        
        return features_list
    
    def stack_template_features(
        self,
        features_list: Sequence[TemplateFeatures],
        query_length: int,
        num_templates: int = MAX_TEMPLATES,
    ) -> Dict[str, np.ndarray]:
        """Stack template features into batched arrays.
        
        Args:
            features_list: List of template features.
            query_length: Query sequence length.
            num_templates: Number of template slots (pad if fewer).
            
        Returns:
            Dictionary of stacked feature arrays.
        """
        # Initialize arrays
        stacked = {
            "template_aatype": np.zeros(
                (num_templates, query_length),
                dtype=np.int32,
            ),
            "template_all_atom_positions": np.zeros(
                (num_templates, query_length, 37, 3),
                dtype=np.float32,
            ),
            "template_all_atom_mask": np.zeros(
                (num_templates, query_length, 37),
                dtype=np.float32,
            ),
            "template_pseudo_beta": np.zeros(
                (num_templates, query_length, 3),
                dtype=np.float32,
            ),
            "template_pseudo_beta_mask": np.zeros(
                (num_templates, query_length),
                dtype=np.float32,
            ),
            "template_backbone_frame_mask": np.zeros(
                (num_templates, query_length),
                dtype=np.float32,
            ),
        }
        
        # Fill with actual features
        for i, features in enumerate(features_list[:num_templates]):
            stacked["template_aatype"][i] = features.template_aatype
            stacked["template_all_atom_positions"][i] = features.template_all_atom_positions
            stacked["template_all_atom_mask"][i] = features.template_all_atom_mask
            stacked["template_pseudo_beta"][i] = features.template_pseudo_beta
            stacked["template_pseudo_beta_mask"][i] = features.template_pseudo_beta_mask
            stacked["template_backbone_frame_mask"][i] = features.template_backbone_frame_mask
        
        return stacked


def create_template_pipeline(
    *,
    mmcif_dir: Optional[Path] = None,
    max_release_date: Optional[date] = None,
    max_templates: int = MAX_TEMPLATES,
    obsolete_file_path: Optional[Path] = None,
) -> TemplateProcessingPipeline:
    """Factory function to create a configured template pipeline.
    
    Args:
        mmcif_dir: Directory containing mmCIF files.
        max_release_date: Maximum release date for templates.
        max_templates: Maximum templates to return.
        obsolete_file_path: Path to obsolete PDB entries file.
        
    Returns:
        Configured TemplateProcessingPipeline instance.
    """
    date_config = DateFilterConfig(
        max_release_date=max_release_date,
        obsolete_file_path=obsolete_file_path,
        allow_missing_dates=False,
    )
    
    return TemplateProcessingPipeline(
        hmm_builder=TemplateHMMBuilder(),
        date_filter=TemplateDateFilter(config=date_config),
        deduplicator=TemplateDeduplicator(),
        feature_extractor=TemplateFeatureExtractor(),
        mmcif_dir=mmcif_dir,
        max_templates=max_templates,
    )
