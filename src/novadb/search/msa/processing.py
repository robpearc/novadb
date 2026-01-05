"""MSA processing utilities for AlphaFold3-style MSA construction.

Implements MSA processing from AF3 Supplement Sections 2.2-2.3:
- Deduplication of sequences
- Block deletion removal
- Species-based pairing for multi-chain complexes
- BFD search integration
- UniProt-to-UniRef90 mapping
- RNA MSA realignment

Reference: Abramson et al. 2024, Nature.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from novadb.search.msa.msa import MSA, MSASequence

logger = logging.getLogger(__name__)

# Constants from AF3 Section 2.2
MAX_MSA_ROWS = 16384
MAX_PAIRED_ROWS = 8191
DEFAULT_BLOCK_DELETION_THRESHOLD = 0.5
DEFAULT_GAP_THRESHOLD = 0.9


@dataclass(frozen=True)
class DeduplicationConfig:
    """Configuration for MSA deduplication.
    
    Attributes:
        preserve_query: Whether to always preserve the query sequence (row 0).
        use_hash: Whether to use sequence hashing for faster deduplication.
        case_sensitive: Whether sequence comparison is case-sensitive.
    """
    
    preserve_query: bool = True
    use_hash: bool = True
    case_sensitive: bool = False


@dataclass(frozen=True)
class BlockDeletionConfig:
    """Configuration for block deletion removal.
    
    From AF3 Section 2.2: Remove sequences with large insertions relative
    to the query that create block deletions in the MSA.
    
    Attributes:
        threshold: Maximum fraction of positions with deletions allowed.
        min_deletion_length: Minimum consecutive deletion length to consider.
        max_total_deletions: Maximum total deletion count per sequence.
    """
    
    threshold: float = DEFAULT_BLOCK_DELETION_THRESHOLD
    min_deletion_length: int = 3
    max_total_deletions: int = 1000


@dataclass(frozen=True)
class SpeciesPairingConfig:
    """Configuration for species-based MSA pairing.
    
    From AF3 Section 2.3: Pair sequences across chains by species.
    
    Attributes:
        max_paired_rows: Maximum number of paired rows to generate.
        require_all_chains: Whether all chains must have a hit for a species.
        species_patterns: Regex patterns for extracting species from descriptions.
    """
    
    max_paired_rows: int = MAX_PAIRED_ROWS
    require_all_chains: bool = True
    species_patterns: Tuple[str, ...] = (
        r'OS=([^=]+?)(?:\s+OX=|\s+GN=|\s+PE=|\s*$)',  # UniProt format
        r'\[([^\]]+)\]',  # Bracket format
        r'_([A-Z]{3,})(?:\s|$)',  # UniProt ID suffix
    )


@dataclass(frozen=True)
class UniRef90MappingConfig:
    """Configuration for UniProt-to-UniRef90 mapping.
    
    From AF3 Section 2.2: Map UniProt hits to UniRef90 cluster representatives.
    
    Attributes:
        mapping_file: Path to UniProt-to-UniRef90 mapping file.
        use_cluster_representatives: Whether to use cluster representatives.
        fallback_to_original: Whether to keep unmapped sequences.
    """
    
    mapping_file: Optional[Path] = None
    use_cluster_representatives: bool = True
    fallback_to_original: bool = True


@dataclass(frozen=True)
class RNARealignmentConfig:
    """Configuration for RNA MSA realignment.
    
    From AF3 Section 2.2: Use hmmalign to realign RNA MSAs to query.
    
    Attributes:
        hmmalign_binary: Path to hmmalign executable.
        hmmbuild_binary: Path to hmmbuild executable.
        cpu_count: Number of CPUs for hmmalign.
        trim_ends: Whether to trim poorly aligned ends.
    """
    
    hmmalign_binary: Path = field(default_factory=lambda: Path("hmmalign"))
    hmmbuild_binary: Path = field(default_factory=lambda: Path("hmmbuild"))
    cpu_count: int = 4
    trim_ends: bool = True


class MSAProcessor(Protocol):
    """Protocol for MSA processing operations."""
    
    def process(self, msa: MSA) -> MSA:
        """Process an MSA and return the result."""
        ...


@dataclass
class MSADeduplicator:
    """Removes duplicate sequences from MSAs.
    
    From AF3 Section 2.2: Deduplicate sequences to avoid redundancy
    and reduce computational cost.
    
    Attributes:
        config: Deduplication configuration.
    """
    
    config: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    
    def process(self, msa: MSA) -> MSA:
        """Remove duplicate sequences from MSA.
        
        Args:
            msa: Input MSA with potential duplicates.
            
        Returns:
            MSA with unique sequences only.
        """
        if msa.depth == 0:
            return msa
        
        seen_sequences: set[str] = set()
        unique_sequences: list[MSASequence] = []
        
        for i, seq in enumerate(msa.sequences):
            sequence_key = self._get_sequence_key(seq.sequence)
            
            # Always keep query if configured
            if i == 0 and self.config.preserve_query:
                seen_sequences.add(sequence_key)
                unique_sequences.append(seq)
                continue
            
            if sequence_key not in seen_sequences:
                seen_sequences.add(sequence_key)
                unique_sequences.append(seq)
        
        logger.debug(
            "Deduplication: %d -> %d sequences",
            msa.depth,
            len(unique_sequences),
        )
        
        return MSA(
            sequences=unique_sequences,
            query_sequence=msa.query_sequence,
            database=msa.database,
        )
    
    def _get_sequence_key(self, sequence: str) -> str:
        """Get a key for sequence comparison.
        
        Args:
            sequence: Raw sequence string.
            
        Returns:
            Normalized key for comparison.
        """
        normalized = sequence if self.config.case_sensitive else sequence.upper()
        
        if self.config.use_hash:
            return hashlib.md5(normalized.encode()).hexdigest()
        return normalized
    
    def deduplicate_by_cluster(
        self,
        msa: MSA,
        cluster_mapping: Mapping[str, str],
    ) -> MSA:
        """Deduplicate sequences by cluster representative.
        
        Args:
            msa: Input MSA.
            cluster_mapping: Mapping from sequence ID to cluster representative.
            
        Returns:
            MSA with one sequence per cluster.
        """
        seen_clusters: set[str] = set()
        unique_sequences: list[MSASequence] = []
        
        for i, seq in enumerate(msa.sequences):
            # Always keep query
            if i == 0 and self.config.preserve_query:
                unique_sequences.append(seq)
                continue
            
            # Get cluster ID
            accession = seq.accession or self._extract_accession(seq.description)
            cluster_id = cluster_mapping.get(accession, accession)
            
            if cluster_id not in seen_clusters:
                seen_clusters.add(cluster_id)
                unique_sequences.append(seq)
        
        return MSA(
            sequences=unique_sequences,
            query_sequence=msa.query_sequence,
            database=msa.database,
        )
    
    @staticmethod
    def _extract_accession(description: str) -> str:
        """Extract accession from sequence description."""
        # Try UniProt format: sp|P12345|NAME
        match = re.search(r'[st][pr]\|([A-Z0-9]+)\|', description)
        if match:
            return match.group(1)
        
        # Try simple accession at start
        parts = description.split()
        if parts:
            return parts[0]
        
        return description


@dataclass
class BlockDeletionRemover:
    """Removes sequences with excessive block deletions.
    
    From AF3 Section 2.2: Remove sequences with large insertions relative
    to the query, which create problematic block deletions in the MSA.
    
    Attributes:
        config: Block deletion configuration.
    """
    
    config: BlockDeletionConfig = field(default_factory=BlockDeletionConfig)
    
    def process(self, msa: MSA) -> MSA:
        """Remove sequences with excessive block deletions.
        
        Args:
            msa: Input MSA.
            
        Returns:
            MSA with problematic sequences removed.
        """
        if msa.depth == 0:
            return msa
        
        filtered_sequences: list[MSASequence] = []
        
        for i, seq in enumerate(msa.sequences):
            # Always keep query
            if i == 0:
                filtered_sequences.append(seq)
                continue
            
            if self._is_acceptable(seq):
                filtered_sequences.append(seq)
        
        logger.debug(
            "Block deletion removal: %d -> %d sequences",
            msa.depth,
            len(filtered_sequences),
        )
        
        return MSA(
            sequences=filtered_sequences,
            query_sequence=msa.query_sequence,
            database=msa.database,
        )
    
    def _is_acceptable(self, seq: MSASequence) -> bool:
        """Check if sequence has acceptable deletion pattern.
        
        Args:
            seq: Sequence to check.
            
        Returns:
            True if sequence is acceptable.
        """
        if not seq.deletions:
            return True
        
        total_deletions = sum(seq.deletions)
        
        # Check total deletion count
        if total_deletions > self.config.max_total_deletions:
            return False
        
        # Check deletion fraction
        if len(seq.sequence) > 0:
            deletion_fraction = total_deletions / len(seq.sequence)
            if deletion_fraction > self.config.threshold:
                return False
        
        # Check for large consecutive deletions (block deletions)
        if self._has_block_deletion(seq.deletions):
            return False
        
        return True
    
    def _has_block_deletion(self, deletions: Sequence[int]) -> bool:
        """Check if deletions contain a block deletion.
        
        Args:
            deletions: Deletion counts at each position.
            
        Returns:
            True if a block deletion is detected.
        """
        consecutive_count = 0
        
        for d in deletions:
            if d > 0:
                consecutive_count += 1
                if consecutive_count >= self.config.min_deletion_length:
                    return True
            else:
                consecutive_count = 0
        
        return False
    
    def remove_gappy_columns(
        self,
        msa: MSA,
        gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    ) -> MSA:
        """Remove columns with too many gaps.
        
        Args:
            msa: Input MSA.
            gap_threshold: Maximum fraction of gaps allowed per column.
            
        Returns:
            MSA with gappy columns removed.
        """
        if msa.depth == 0 or msa.width == 0:
            return msa
        
        # Compute gap fraction per column
        gap_counts = np.zeros(msa.width, dtype=np.float32)
        for seq in msa.sequences:
            for j, char in enumerate(seq.sequence):
                if char == "-":
                    gap_counts[j] += 1
        
        gap_fractions = gap_counts / msa.depth
        keep_columns = gap_fractions < gap_threshold
        
        # Filter columns
        new_sequences = []
        for seq in msa.sequences:
            new_seq_str = "".join(
                char for j, char in enumerate(seq.sequence) if keep_columns[j]
            )
            new_deletions = None
            if seq.deletions:
                new_deletions = [
                    d for j, d in enumerate(seq.deletions) if keep_columns[j]
                ]
            
            new_sequences.append(MSASequence(
                sequence=new_seq_str,
                description=seq.description,
                accession=seq.accession,
                species=seq.species,
                e_value=seq.e_value,
                deletions=new_deletions,
            ))
        
        return MSA(
            sequences=new_sequences,
            query_sequence=msa.query_sequence[:sum(keep_columns)] if msa.query_sequence else None,
            database=msa.database,
        )


@dataclass
class SpeciesPairer:
    """Pairs sequences across chains by species.
    
    From AF3 Section 2.3: For multi-chain complexes, sequences from
    different chains are paired if they come from the same species.
    
    Attributes:
        config: Species pairing configuration.
    """
    
    config: SpeciesPairingConfig = field(default_factory=SpeciesPairingConfig)
    
    def extract_species(self, description: str) -> Optional[str]:
        """Extract species from sequence description.
        
        Args:
            description: Sequence description line.
            
        Returns:
            Extracted species name or None.
        """
        for pattern in self.config.species_patterns:
            match = re.search(pattern, description)
            if match:
                return match.group(1).strip()
        return None
    
    def group_by_species(
        self,
        msa: MSA,
        skip_query: bool = True,
    ) -> Dict[str, List[MSASequence]]:
        """Group MSA sequences by species.
        
        Args:
            msa: Input MSA.
            skip_query: Whether to skip the query sequence.
            
        Returns:
            Dictionary mapping species to list of sequences.
        """
        species_groups: Dict[str, List[MSASequence]] = defaultdict(list)
        
        start_idx = 1 if skip_query else 0
        for seq in msa.sequences[start_idx:]:
            species = seq.species or self.extract_species(seq.description)
            if species:
                species_groups[species].append(seq)
        
        return dict(species_groups)
    
    def find_common_species(
        self,
        chain_msas: Mapping[str, MSA],
    ) -> FrozenSet[str]:
        """Find species present in all chain MSAs.
        
        Args:
            chain_msas: MSAs for each chain.
            
        Returns:
            Set of species present in all chains.
        """
        if not chain_msas:
            return frozenset()
        
        chain_species_sets: list[set[str]] = []
        
        for msa in chain_msas.values():
            species_groups = self.group_by_species(msa)
            chain_species_sets.append(set(species_groups.keys()))
        
        if not chain_species_sets:
            return frozenset()
        
        common_species = chain_species_sets[0]
        for species_set in chain_species_sets[1:]:
            common_species &= species_set
        
        return frozenset(common_species)
    
    def pair_sequences(
        self,
        chain_msas: Mapping[str, MSA],
        chain_order: Sequence[str],
    ) -> List[Dict[str, MSASequence]]:
        """Pair sequences across chains by species.
        
        Args:
            chain_msas: MSAs for each chain.
            chain_order: Order of chains for consistent output.
            
        Returns:
            List of paired sequence dictionaries.
        """
        # Find species present in all chains
        common_species = self.find_common_species(chain_msas)
        
        if not common_species:
            logger.warning("No common species found for pairing")
            return []
        
        # Group sequences by species for each chain
        chain_species_groups: Dict[str, Dict[str, List[MSASequence]]] = {}
        for chain_id, msa in chain_msas.items():
            chain_species_groups[chain_id] = self.group_by_species(msa)
        
        # Create paired rows
        paired_rows: List[Dict[str, MSASequence]] = []
        
        for species in sorted(common_species):
            # Get sequences for each chain
            chain_sequences: Dict[str, List[MSASequence]] = {}
            all_chains_have_species = True
            
            for chain_id in chain_order:
                groups = chain_species_groups.get(chain_id, {})
                seqs = groups.get(species, [])
                if not seqs and self.config.require_all_chains:
                    all_chains_have_species = False
                    break
                chain_sequences[chain_id] = seqs
            
            if not all_chains_have_species:
                continue
            
            # Create paired rows (use first sequence for each chain)
            paired_row: Dict[str, MSASequence] = {}
            for chain_id in chain_order:
                seqs = chain_sequences.get(chain_id, [])
                if seqs:
                    paired_row[chain_id] = seqs[0]
            
            if len(paired_row) == len(chain_order):
                paired_rows.append(paired_row)
            
            if len(paired_rows) >= self.config.max_paired_rows:
                break
        
        logger.debug("Created %d paired rows from %d species", len(paired_rows), len(common_species))
        return paired_rows
    
    def create_paired_msa_array(
        self,
        paired_rows: Sequence[Dict[str, MSASequence]],
        chain_lengths: Mapping[str, int],
        chain_order: Sequence[str],
        query_sequences: Mapping[str, str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create concatenated MSA array from paired sequences.
        
        Args:
            paired_rows: List of paired sequence dictionaries.
            chain_lengths: Length of each chain.
            chain_order: Order of chains.
            query_sequences: Query sequence for each chain.
            
        Returns:
            Tuple of (msa_array, deletion_matrix).
        """
        total_length = sum(chain_lengths[c] for c in chain_order)
        num_rows = len(paired_rows) + 1  # +1 for query row
        
        msa_array = np.zeros((num_rows, total_length), dtype=np.int32)
        deletion_matrix = np.zeros((num_rows, total_length), dtype=np.int32)
        
        # Build offset map
        offset = 0
        chain_offsets: Dict[str, int] = {}
        for chain_id in chain_order:
            chain_offsets[chain_id] = offset
            offset += chain_lengths[chain_id]
        
        # Fill query row
        for chain_id in chain_order:
            query = query_sequences.get(chain_id, "")
            start = chain_offsets[chain_id]
            length = chain_lengths[chain_id]
            
            for j, char in enumerate(query[:length]):
                msa_array[0, start + j] = _residue_to_int(char)
        
        # Fill paired rows
        for i, paired_row in enumerate(paired_rows, start=1):
            for chain_id in chain_order:
                seq = paired_row.get(chain_id)
                if seq is None:
                    # Fill with gaps
                    start = chain_offsets[chain_id]
                    length = chain_lengths[chain_id]
                    msa_array[i, start:start + length] = 31  # Gap token
                    continue
                
                start = chain_offsets[chain_id]
                length = chain_lengths[chain_id]
                
                for j, char in enumerate(seq.sequence[:length]):
                    msa_array[i, start + j] = _residue_to_int(char)
                
                if seq.deletions:
                    for j, d in enumerate(seq.deletions[:length]):
                        deletion_matrix[i, start + j] = d
        
        return msa_array, deletion_matrix


@dataclass
class UniRef90Mapper:
    """Maps UniProt hits to UniRef90 cluster representatives.
    
    From AF3 Section 2.2: Use UniRef90 clustering to reduce redundancy
    and improve MSA diversity.
    
    Attributes:
        config: Mapping configuration.
        _mapping_cache: Cached mapping dictionary.
    """
    
    config: UniRef90MappingConfig = field(default_factory=UniRef90MappingConfig)
    _mapping_cache: Optional[Dict[str, str]] = field(default=None, repr=False)
    
    def load_mapping(self) -> Dict[str, str]:
        """Load UniProt-to-UniRef90 mapping from file.
        
        Returns:
            Dictionary mapping UniProt IDs to UniRef90 cluster IDs.
        """
        if self._mapping_cache is not None:
            return self._mapping_cache
        
        if self.config.mapping_file is None or not self.config.mapping_file.exists():
            logger.warning("UniRef90 mapping file not found")
            return {}
        
        mapping: Dict[str, str] = {}
        
        with open(self.config.mapping_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    uniprot_id, uniref90_id = parts[0], parts[1]
                    mapping[uniprot_id] = uniref90_id
        
        self._mapping_cache = mapping
        logger.info("Loaded %d UniRef90 mappings", len(mapping))
        return mapping
    
    def map_sequence(self, accession: str) -> Optional[str]:
        """Map a UniProt accession to UniRef90 cluster.
        
        Args:
            accession: UniProt accession.
            
        Returns:
            UniRef90 cluster ID or None.
        """
        mapping = self.load_mapping()
        return mapping.get(accession)
    
    def process(self, msa: MSA) -> MSA:
        """Add UniRef90 cluster information to MSA sequences.
        
        Args:
            msa: Input MSA.
            
        Returns:
            MSA with UniRef90 cluster annotations.
        """
        mapping = self.load_mapping()
        
        updated_sequences: list[MSASequence] = []
        for seq in msa.sequences:
            accession = seq.accession or _extract_accession(seq.description)
            cluster_id = mapping.get(accession)
            
            # Create updated sequence with cluster info in description
            if cluster_id:
                new_description = f"{seq.description} UniRef90:{cluster_id}"
            else:
                new_description = seq.description
            
            updated_sequences.append(MSASequence(
                sequence=seq.sequence,
                description=new_description,
                accession=seq.accession,
                species=seq.species,
                e_value=seq.e_value,
                deletions=seq.deletions,
            ))
        
        return MSA(
            sequences=updated_sequences,
            query_sequence=msa.query_sequence,
            database=msa.database,
        )
    
    def deduplicate_by_cluster(self, msa: MSA) -> MSA:
        """Deduplicate MSA by UniRef90 cluster.
        
        Args:
            msa: Input MSA.
            
        Returns:
            MSA with one sequence per UniRef90 cluster.
        """
        mapping = self.load_mapping()
        deduplicator = MSADeduplicator()
        return deduplicator.deduplicate_by_cluster(msa, mapping)


@dataclass
class RNARealigner:
    """Realigns RNA MSAs using hmmalign.
    
    From AF3 Section 2.2: RNA MSAs are realigned to the query using
    hmmalign for better structural alignment.
    
    Attributes:
        config: Realignment configuration.
    """
    
    config: RNARealignmentConfig = field(default_factory=RNARealignmentConfig)
    
    def process(self, msa: MSA) -> MSA:
        """Realign RNA MSA using hmmalign.
        
        Args:
            msa: Input RNA MSA.
            
        Returns:
            Realigned MSA.
        """
        if msa.depth <= 1:
            return msa
        
        try:
            return self._run_hmmalign(msa)
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning("hmmalign failed, returning original MSA: %s", e)
            return msa
    
    def _run_hmmalign(self, msa: MSA) -> MSA:
        """Run hmmalign on the MSA.
        
        Args:
            msa: Input MSA.
            
        Returns:
            Realigned MSA.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Write query sequence for HMM building
            query_fasta = tmpdir_path / "query.fasta"
            with open(query_fasta, "w") as f:
                f.write(f">query\n{msa.sequences[0].sequence}\n")
            
            # Build HMM from query
            hmm_file = tmpdir_path / "query.hmm"
            subprocess.run(
                [
                    str(self.config.hmmbuild_binary),
                    "--rna",
                    str(hmm_file),
                    str(query_fasta),
                ],
                check=True,
                capture_output=True,
            )
            
            # Write all sequences
            seqs_fasta = tmpdir_path / "sequences.fasta"
            with open(seqs_fasta, "w") as f:
                for i, seq in enumerate(msa.sequences):
                    desc = seq.description or f"seq_{i}"
                    f.write(f">{desc}\n{seq.sequence}\n")
            
            # Run hmmalign
            aligned_sto = tmpdir_path / "aligned.sto"
            subprocess.run(
                [
                    str(self.config.hmmalign_binary),
                    "--rna",
                    "--outformat", "Stockholm",
                    "-o", str(aligned_sto),
                    str(hmm_file),
                    str(seqs_fasta),
                ],
                check=True,
                capture_output=True,
            )
            
            # Parse aligned output
            with open(aligned_sto) as f:
                content = f.read()
            
            return MSA.from_stockholm(content, database=msa.database)
    
    def realign_to_reference(
        self,
        msa: MSA,
        reference_hmm: Path,
    ) -> MSA:
        """Realign MSA to a reference HMM.
        
        Args:
            msa: Input MSA.
            reference_hmm: Path to reference HMM file.
            
        Returns:
            Realigned MSA.
        """
        if not reference_hmm.exists():
            raise FileNotFoundError(f"Reference HMM not found: {reference_hmm}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Write sequences
            seqs_fasta = tmpdir_path / "sequences.fasta"
            with open(seqs_fasta, "w") as f:
                for i, seq in enumerate(msa.sequences):
                    desc = seq.description or f"seq_{i}"
                    f.write(f">{desc}\n{seq.sequence}\n")
            
            # Run hmmalign
            aligned_sto = tmpdir_path / "aligned.sto"
            subprocess.run(
                [
                    str(self.config.hmmalign_binary),
                    "--rna",
                    "--outformat", "Stockholm",
                    "-o", str(aligned_sto),
                    str(reference_hmm),
                    str(seqs_fasta),
                ],
                check=True,
                capture_output=True,
            )
            
            with open(aligned_sto) as f:
                content = f.read()
            
            return MSA.from_stockholm(content, database=msa.database)


@dataclass
class BFDSearcher:
    """Integrates reduced BFD database search.
    
    From AF3 Section 2.2: Search against reduced BFD for additional
    diverse sequences.
    
    Attributes:
        bfd_database_path: Path to reduced BFD database.
        hhblits_binary: Path to HHblits executable.
        n_iterations: Number of HHblits iterations.
        e_value_threshold: E-value threshold for hits.
        max_sequences: Maximum sequences to return.
    """
    
    bfd_database_path: Optional[Path] = None
    hhblits_binary: Path = field(default_factory=lambda: Path("hhblits"))
    n_iterations: int = 2
    e_value_threshold: float = 1e-3
    max_sequences: int = 5000
    
    def search(self, query: str) -> MSA:
        """Search BFD database with query sequence.
        
        Args:
            query: Query sequence string.
            
        Returns:
            MSA from BFD search.
        """
        if self.bfd_database_path is None or not self.bfd_database_path.exists():
            logger.warning("BFD database not found")
            return MSA(sequences=[MSASequence(sequence=query, description="query")])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Write query
            query_fasta = tmpdir_path / "query.fasta"
            with open(query_fasta, "w") as f:
                f.write(f">query\n{query}\n")
            
            # Run HHblits
            output_a3m = tmpdir_path / "output.a3m"
            subprocess.run(
                [
                    str(self.hhblits_binary),
                    "-i", str(query_fasta),
                    "-d", str(self.bfd_database_path),
                    "-oa3m", str(output_a3m),
                    "-n", str(self.n_iterations),
                    "-e", str(self.e_value_threshold),
                    "-maxseq", str(self.max_sequences),
                    "-cpu", "4",
                ],
                check=True,
                capture_output=True,
            )
            
            with open(output_a3m) as f:
                content = f.read()
            
            return MSA.from_a3m(content, database="bfd")
    
    def filter_by_coverage(
        self,
        msa: MSA,
        min_coverage: float = 0.5,
    ) -> MSA:
        """Filter BFD hits by query coverage.
        
        Args:
            msa: MSA from BFD search.
            min_coverage: Minimum fraction of query covered.
            
        Returns:
            Filtered MSA.
        """
        if msa.depth == 0:
            return msa
        
        query_length = len(msa.sequences[0].sequence.replace("-", ""))
        filtered_sequences = [msa.sequences[0]]  # Keep query
        
        for seq in msa.sequences[1:]:
            aligned_positions = sum(
                1 for c in seq.sequence if c != "-" and c != "X"
            )
            coverage = aligned_positions / max(query_length, 1)
            
            if coverage >= min_coverage:
                filtered_sequences.append(seq)
        
        return MSA(
            sequences=filtered_sequences,
            query_sequence=msa.query_sequence,
            database=msa.database,
        )


@dataclass(frozen=True)
class MSAProcessingPipeline:
    """Complete MSA processing pipeline.
    
    Implements the full AF3 MSA processing workflow from Section 2.2-2.3.
    
    Attributes:
        deduplicator: Sequence deduplication processor.
        block_deletion_remover: Block deletion filter.
        species_pairer: Species-based pairing processor.
        uniref90_mapper: UniRef90 mapping processor.
        rna_realigner: RNA realignment processor.
        bfd_searcher: BFD database searcher.
    """
    
    deduplicator: MSADeduplicator = field(default_factory=MSADeduplicator)
    block_deletion_remover: BlockDeletionRemover = field(default_factory=BlockDeletionRemover)
    species_pairer: SpeciesPairer = field(default_factory=SpeciesPairer)
    uniref90_mapper: Optional[UniRef90Mapper] = None
    rna_realigner: Optional[RNARealigner] = None
    bfd_searcher: Optional[BFDSearcher] = None
    
    def process_single_chain(
        self,
        msa: MSA,
        is_rna: bool = False,
        max_rows: int = MAX_MSA_ROWS,
    ) -> MSA:
        """Process MSA for a single chain.
        
        Args:
            msa: Input MSA.
            is_rna: Whether this is an RNA sequence.
            max_rows: Maximum MSA rows.
            
        Returns:
            Processed MSA.
        """
        # Step 1: Deduplicate
        processed = self.deduplicator.process(msa)
        
        # Step 2: Remove block deletions
        processed = self.block_deletion_remover.process(processed)
        
        # Step 3: Map to UniRef90 clusters (if available)
        if self.uniref90_mapper is not None:
            processed = self.uniref90_mapper.process(processed)
        
        # Step 4: Realign RNA MSAs (if applicable)
        if is_rna and self.rna_realigner is not None:
            processed = self.rna_realigner.process(processed)
        
        # Step 5: Crop to max rows
        if processed.depth > max_rows:
            processed = processed.crop(max_rows)
        
        return processed
    
    def process_multi_chain(
        self,
        chain_msas: Dict[str, MSA],
        chain_lengths: Dict[str, int],
        chain_order: List[str],
        max_paired_rows: int = MAX_PAIRED_ROWS,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Process and pair MSAs for multi-chain complex.
        
        From AF3 Section 2.3: Create paired MSA with query row,
        up to 8191 paired rows, and dense stacking.
        
        Args:
            chain_msas: MSAs for each chain.
            chain_lengths: Length of each chain.
            chain_order: Order of chains.
            max_paired_rows: Maximum paired rows.
            
        Returns:
            Tuple of (msa_array, deletion_matrix, species_list).
        """
        # Process each chain's MSA
        processed_msas: Dict[str, MSA] = {}
        for chain_id, msa in chain_msas.items():
            processed_msas[chain_id] = self.process_single_chain(msa)
        
        # Pair sequences by species
        paired_rows = self.species_pairer.pair_sequences(
            processed_msas,
            chain_order,
        )
        
        # Limit to max paired rows
        paired_rows = paired_rows[:max_paired_rows]
        
        # Extract query sequences
        query_sequences = {
            chain_id: msa.sequences[0].sequence if msa.sequences else ""
            for chain_id, msa in processed_msas.items()
        }
        
        # Create concatenated arrays
        msa_array, deletion_matrix = self.species_pairer.create_paired_msa_array(
            paired_rows,
            chain_lengths,
            chain_order,
            query_sequences,
        )
        
        # Extract species list
        species_list = [
            self.species_pairer.extract_species(
                next(iter(row.values())).description
            ) or "unknown"
            for row in paired_rows
        ]
        
        return msa_array, deletion_matrix, species_list
    
    def add_bfd_sequences(
        self,
        msa: MSA,
        query: str,
    ) -> MSA:
        """Add sequences from BFD search.
        
        Args:
            msa: Existing MSA.
            query: Query sequence for BFD search.
            
        Returns:
            MSA with BFD sequences added.
        """
        if self.bfd_searcher is None:
            return msa
        
        bfd_msa = self.bfd_searcher.search(query)
        bfd_msa = self.bfd_searcher.filter_by_coverage(bfd_msa)
        
        # Merge, skipping BFD query (already have it)
        merged_sequences = list(msa.sequences)
        for seq in bfd_msa.sequences[1:]:
            merged_sequences.append(seq)
        
        return MSA(
            sequences=merged_sequences,
            query_sequence=msa.query_sequence,
            database=f"{msa.database},bfd",
        )


def _residue_to_int(char: str) -> int:
    """Convert residue character to integer code.
    
    Uses 32 classes as per AF3 Table 5 (restype feature).
    
    Args:
        char: Single character residue.
        
    Returns:
        Integer code (0-31).
    """
    mapping = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
        'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
        'X': 20,  # Unknown amino acid
        '-': 31,  # Gap
    }
    return mapping.get(char.upper(), 20)


def _extract_accession(description: str) -> str:
    """Extract accession from sequence description.
    
    Args:
        description: Sequence description line.
        
    Returns:
        Extracted accession or original description.
    """
    match = re.search(r'[st][pr]\|([A-Z0-9]+)\|', description)
    if match:
        return match.group(1)
    
    parts = description.split()
    return parts[0] if parts else description


def create_processing_pipeline(
    *,
    uniref90_mapping_file: Optional[Path] = None,
    bfd_database_path: Optional[Path] = None,
    enable_rna_realignment: bool = True,
) -> MSAProcessingPipeline:
    """Factory function to create a configured processing pipeline.
    
    Args:
        uniref90_mapping_file: Path to UniRef90 mapping file.
        bfd_database_path: Path to BFD database.
        enable_rna_realignment: Whether to enable RNA realignment.
        
    Returns:
        Configured MSAProcessingPipeline instance.
    """
    uniref90_mapper = None
    if uniref90_mapping_file is not None:
        uniref90_mapper = UniRef90Mapper(
            config=UniRef90MappingConfig(mapping_file=uniref90_mapping_file)
        )
    
    rna_realigner = RNARealigner() if enable_rna_realignment else None
    
    bfd_searcher = None
    if bfd_database_path is not None:
        bfd_searcher = BFDSearcher(bfd_database_path=bfd_database_path)
    
    return MSAProcessingPipeline(
        deduplicator=MSADeduplicator(),
        block_deletion_remover=BlockDeletionRemover(),
        species_pairer=SpeciesPairer(),
        uniref90_mapper=uniref90_mapper,
        rna_realigner=rna_realigner,
        bfd_searcher=bfd_searcher,
    )


# =============================================================================
# Convenience Functions for Common Operations
# =============================================================================

def deduplicate_msa(
    sequences: List[str],
    preserve_first: bool = True,
) -> List[str]:
    """Deduplicate MSA sequences.
    
    Simple convenience function for removing duplicate sequences.
    
    Args:
        sequences: List of sequences to deduplicate.
        preserve_first: Whether to always keep the first sequence (query).
        
    Returns:
        List of unique sequences.
    """
    seen = set()
    result = []
    
    for i, seq in enumerate(sequences):
        # Normalize for comparison
        normalized = seq.upper().replace("-", "")
        
        if i == 0 and preserve_first:
            result.append(seq)
            seen.add(normalized)
        elif normalized not in seen:
            result.append(seq)
            seen.add(normalized)
    
    return result


def compute_msa_profile(
    sequences: List[str],
    alphabet: str = "ACDEFGHIKLMNPQRSTVWY-",
) -> List[Dict[str, float]]:
    """Compute amino acid profile from MSA.
    
    For each position, compute the frequency of each residue type.
    
    Args:
        sequences: List of aligned sequences (same length).
        alphabet: Valid characters to count.
        
    Returns:
        List of dictionaries mapping residue to frequency at each position.
    """
    if not sequences:
        return []
    
    seq_len = len(sequences[0])
    profile = []
    
    for pos in range(seq_len):
        counts = defaultdict(int)
        total = 0
        
        for seq in sequences:
            if pos < len(seq):
                char = seq[pos].upper()
                if char in alphabet:
                    counts[char] += 1
                    total += 1
        
        # Normalize to frequencies
        freqs = {}
        if total > 0:
            for char, count in counts.items():
                freqs[char] = count / total
        
        profile.append(freqs)
    
    return profile


def compute_deletion_matrix(
    sequences: List[str],
) -> np.ndarray:
    """Compute deletion matrix from MSA.
    
    From AF3 Table 5: deletion counts are transformed using arctan.
    
    Args:
        sequences: List of aligned sequences.
        
    Returns:
        Array of shape (num_seqs, seq_len) with deletion counts.
    """
    if not sequences:
        return np.zeros((0, 0), dtype=np.int32)
    
    num_seqs = len(sequences)
    seq_len = len(sequences[0])
    
    deletions = np.zeros((num_seqs, seq_len), dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        deletion_count = 0
        for j, char in enumerate(seq):
            if char == "-":
                deletion_count += 1
            else:
                if j > 0:
                    deletions[i, j - 1] = deletion_count
                deletion_count = 0
    
    return deletions


def transform_deletion_value(deletion_count: int) -> float:
    """Transform deletion count to feature value.
    
    From AF3 Table 5: 2/π * arctan(d/3)
    
    Args:
        deletion_count: Raw deletion count.
        
    Returns:
        Transformed value in [0, 1].
    """
    return (2 / np.pi) * np.arctan(deletion_count / 3)


def subsample_msa(
    sequences: List[str],
    max_sequences: int,
    preserve_query: bool = True,
    seed: Optional[int] = None,
) -> List[str]:
    """Subsample MSA to maximum size.
    
    From AF3 Section 2.2: MSA is subsampled from size n to
    k = Uniform[1, n] by selecting sequences at random.
    
    Args:
        sequences: Full MSA sequences.
        max_sequences: Maximum number of sequences.
        preserve_query: Keep query (first) sequence.
        seed: Random seed for reproducibility.
        
    Returns:
        Subsampled sequences.
    """
    if len(sequences) <= max_sequences:
        return sequences
    
    rng = np.random.RandomState(seed)
    
    if preserve_query:
        # Keep query, sample from rest
        query = sequences[0]
        rest = sequences[1:]
        
        # Sample size uniformly from [1, n]
        n = len(rest)
        k = rng.randint(1, min(n, max_sequences - 1) + 1)
        
        indices = rng.choice(n, size=k, replace=False)
        sampled = [rest[i] for i in sorted(indices)]
        
        return [query] + sampled
    else:
        n = len(sequences)
        k = rng.randint(1, min(n, max_sequences) + 1)
        
        indices = rng.choice(n, size=k, replace=False)
        return [sequences[i] for i in sorted(indices)]


def concatenate_msas(
    msas: List[List[str]],
    gap_char: str = "-",
) -> List[str]:
    """Concatenate multiple MSAs horizontally.
    
    Used for multi-chain complex MSA construction.
    
    Args:
        msas: List of MSAs to concatenate.
        gap_char: Character for gaps.
        
    Returns:
        Concatenated MSA.
    """
    if not msas:
        return []
    
    # Get maximum depth
    max_depth = max(len(msa) for msa in msas)
    
    # Get lengths
    lengths = [len(msa[0]) if msa else 0 for msa in msas]
    
    result = []
    for row_idx in range(max_depth):
        row_parts = []
        for msa_idx, msa in enumerate(msas):
            if row_idx < len(msa):
                row_parts.append(msa[row_idx])
            else:
                # Pad with gaps
                row_parts.append(gap_char * lengths[msa_idx])
        result.append("".join(row_parts))
    
    return result


def extract_species_from_description(
    description: str,
    patterns: Optional[Tuple[str, ...]] = None,
) -> Optional[str]:
    """Extract species name from sequence description.
    
    Args:
        description: Sequence description/header.
        patterns: Regex patterns to try.
        
    Returns:
        Species name if found, None otherwise.
    """
    if patterns is None:
        patterns = (
            r'OS=([^=]+?)(?:\s+OX=|\s+GN=|\s+PE=|\s*$)',
            r'\[([^\]]+)\]',
            r'_([A-Z]{3,})(?:\s|$)',
        )
    
    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            species = match.group(1).strip()
            if species:
                return species
    
    return None
