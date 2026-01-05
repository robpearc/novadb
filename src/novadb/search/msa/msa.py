"""Multiple Sequence Alignment (MSA) data structures.

This module provides data classes for representing MSAs and their associated
metadata, following the conventions used in AlphaFold3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MSASequence:
    """Represents a single sequence in an MSA.
    
    Attributes:
        sequence: The aligned sequence (with gaps as '-')
        description: FASTA description line
        accession: Accession number (if available)
        species: Species name for pairing (if available)
        e_value: E-value from search (if available)
        deletions: Deletion counts at each position
    """
    sequence: str
    description: str = ""
    accession: str = ""
    species: str = ""
    e_value: Optional[float] = None
    deletions: Optional[List[int]] = None

    @property
    def length(self) -> int:
        """Length of the aligned sequence."""
        return len(self.sequence)

    @property
    def num_gaps(self) -> int:
        """Number of gap characters in the sequence."""
        return self.sequence.count("-")

    @property
    def coverage(self) -> float:
        """Fraction of non-gap positions."""
        if self.length == 0:
            return 0.0
        return 1.0 - (self.num_gaps / self.length)


@dataclass
class MSA:
    """Represents a Multiple Sequence Alignment.
    
    Based on AF3 supplement Section 2.2-2.3:
    - First row is always the query sequence
    - Max 16,384 rows (Nmsa)
    - Includes deletion information
    
    Attributes:
        sequences: List of MSASequence objects
        query_sequence: The original query sequence (unaligned)
        database: Source database name
    """
    sequences: List[MSASequence] = field(default_factory=list)
    query_sequence: str = ""
    database: str = ""

    def __post_init__(self):
        if self.sequences and not self.query_sequence:
            # Extract query from first sequence (remove gaps)
            self.query_sequence = self.sequences[0].sequence.replace("-", "")

    @property
    def depth(self) -> int:
        """Number of sequences in the MSA."""
        return len(self.sequences)

    @property
    def width(self) -> int:
        """Width of the alignment (including gaps)."""
        if self.sequences:
            return self.sequences[0].length
        return 0

    @property
    def query_length(self) -> int:
        """Length of the query sequence."""
        return len(self.query_sequence)

    def get_array(self) -> np.ndarray:
        """Convert MSA to numpy array of character codes.
        
        Returns:
            Array of shape (depth, width) with integer residue codes
        """
        if not self.sequences:
            return np.zeros((0, 0), dtype=np.int32)

        # Use simple character mapping
        residue_to_int = {c: i for i, c in enumerate("ARNDCQEGHILKMFPSTWYV-X")}

        array = np.zeros((self.depth, self.width), dtype=np.int32)
        for i, seq in enumerate(self.sequences):
            for j, char in enumerate(seq.sequence.upper()):
                array[i, j] = residue_to_int.get(char, residue_to_int["X"])

        return array

    def get_deletion_matrix(self) -> np.ndarray:
        """Get deletion counts as a matrix.
        
        Returns:
            Array of shape (depth, width) with deletion counts
        """
        if not self.sequences:
            return np.zeros((0, 0), dtype=np.int32)

        matrix = np.zeros((self.depth, self.width), dtype=np.int32)
        for i, seq in enumerate(self.sequences):
            if seq.deletions:
                matrix[i, :len(seq.deletions)] = seq.deletions

        return matrix

    def get_has_deletion(self) -> np.ndarray:
        """Get binary mask for positions with deletions.
        
        From AF3 supplement Table 5: has_deletion feature.
        
        Returns:
            Array of shape (depth, width) with 1 where deletions exist
        """
        return (self.get_deletion_matrix() > 0).astype(np.int32)

    def get_deletion_value(self) -> np.ndarray:
        """Get transformed deletion values.
        
        From AF3 supplement Table 5: deletion_value = 2/π * arctan(d/3)
        
        Returns:
            Array of shape (depth, width) with transformed deletion counts
        """
        deletions = self.get_deletion_matrix().astype(np.float32)
        return (2 / np.pi) * np.arctan(deletions / 3)

    def get_profile(self) -> np.ndarray:
        """Compute sequence profile (residue distribution at each position).
        
        From AF3 supplement Table 5: Distribution across restypes in main MSA.
        
        Returns:
            Array of shape (width, 32) with residue probabilities
        """
        if not self.sequences:
            return np.zeros((0, 32), dtype=np.float32)

        # 32 classes as per AF3: 20 AA + unknown, 5 RNA, 5 DNA, gap
        num_classes = 32
        counts = np.zeros((self.width, num_classes), dtype=np.float32)

        # Simplified mapping to 32 classes
        aa_mapping = {c: i for i, c in enumerate("ARNDCQEGHILKMFPSTWYV")}

        for seq in self.sequences:
            for j, char in enumerate(seq.sequence.upper()):
                if char in aa_mapping:
                    counts[j, aa_mapping[char]] += 1
                elif char == "-":
                    counts[j, 31] += 1  # Gap
                else:
                    counts[j, 20] += 1  # Unknown

        # Normalize to get probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        profile = counts / row_sums

        return profile

    def get_deletion_mean(self) -> np.ndarray:
        """Compute mean deletion count at each position.
        
        From AF3 supplement Table 5: Mean number of deletions at each position.
        
        Returns:
            Array of shape (width,) with mean deletion counts
        """
        deletion_matrix = self.get_deletion_matrix()
        if deletion_matrix.size == 0:
            return np.zeros(0, dtype=np.float32)
        return deletion_matrix.mean(axis=0).astype(np.float32)

    def deduplicate(self) -> "MSA":
        """Remove duplicate sequences, keeping the first occurrence.
        
        Returns:
            New MSA with unique sequences
        """
        seen = set()
        unique_sequences = []

        for seq in self.sequences:
            if seq.sequence not in seen:
                seen.add(seq.sequence)
                unique_sequences.append(seq)

        return MSA(
            sequences=unique_sequences,
            query_sequence=self.query_sequence,
            database=self.database,
        )

    def crop(self, max_sequences: int) -> "MSA":
        """Crop MSA to maximum number of sequences.
        
        Args:
            max_sequences: Maximum number of sequences to keep
            
        Returns:
            New MSA with at most max_sequences rows
        """
        return MSA(
            sequences=self.sequences[:max_sequences],
            query_sequence=self.query_sequence,
            database=self.database,
        )

    def subsample(self, k: int, random_state: Optional[int] = None) -> "MSA":
        """Randomly subsample sequences.
        
        From AF3 supplement Section 2.2: During training, the main MSA for each
        sequence is subsampled from size n to size k = Uniform[1, n].
        
        Args:
            k: Number of sequences to sample
            random_state: Random seed for reproducibility
            
        Returns:
            New MSA with k sequences (first sequence always included)
        """
        if k >= self.depth:
            return self

        rng = np.random.default_rng(random_state)

        # Always keep the query (first sequence)
        indices = [0]

        # Randomly sample remaining sequences
        remaining_indices = list(range(1, self.depth))
        sampled = rng.choice(remaining_indices, size=min(k - 1, len(remaining_indices)), replace=False)
        indices.extend(sorted(sampled))

        return MSA(
            sequences=[self.sequences[i] for i in indices],
            query_sequence=self.query_sequence,
            database=self.database,
        )

    @classmethod
    def from_a3m(cls, content: str, database: str = "") -> "MSA":
        """Parse MSA from A3M format.
        
        A3M is a compact format where lowercase letters represent insertions.
        
        Args:
            content: A3M file content
            database: Source database name
            
        Returns:
            Parsed MSA object
        """
        sequences = []
        current_desc = ""
        current_seq = ""

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_seq:
                    seq, deletions = cls._parse_a3m_sequence(current_seq)
                    sequences.append(MSASequence(
                        sequence=seq,
                        description=current_desc,
                        deletions=deletions,
                    ))
                current_desc = line[1:]
                current_seq = ""
            else:
                current_seq += line

        # Add last sequence
        if current_seq:
            seq, deletions = cls._parse_a3m_sequence(current_seq)
            sequences.append(MSASequence(
                sequence=seq,
                description=current_desc,
                deletions=deletions,
            ))

        return cls(sequences=sequences, database=database)

    @staticmethod
    def _parse_a3m_sequence(seq: str) -> Tuple[str, List[int]]:
        """Parse A3M sequence, extracting deletions from lowercase characters.
        
        In A3M format:
        - Uppercase = aligned residues
        - Lowercase = insertions (deletions relative to query)
        - '-' = gaps
        
        Returns:
            Tuple of (sequence with only uppercase/gaps, deletion counts)
        """
        result = []
        deletions = []
        deletion_count = 0

        for char in seq:
            if char.isupper() or char == "-":
                deletions.append(deletion_count)
                result.append(char)
                deletion_count = 0
            elif char.islower():
                deletion_count += 1

        return "".join(result), deletions

    @classmethod
    def from_stockholm(cls, content: str, database: str = "") -> "MSA":
        """Parse MSA from Stockholm format.
        
        Args:
            content: Stockholm file content
            database: Source database name
            
        Returns:
            Parsed MSA object
        """
        sequences_dict: Dict[str, str] = {}
        sequence_order = []

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line == "//":
                continue

            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                seq = parts[1]

                if name not in sequences_dict:
                    sequences_dict[name] = ""
                    sequence_order.append(name)
                sequences_dict[name] += seq

        sequences = [
            MSASequence(sequence=sequences_dict[name], description=name)
            for name in sequence_order
        ]

        return cls(sequences=sequences, database=database)

    def to_a3m(self) -> str:
        """Convert MSA to A3M format string.
        
        Returns:
            A3M formatted string
        """
        lines = []
        for i, seq in enumerate(self.sequences):
            desc = seq.description or f"sequence_{i}"
            lines.append(f">{desc}")
            lines.append(seq.sequence)
        return "\n".join(lines)

    def merge(self, other: "MSA") -> "MSA":
        """Merge with another MSA.
        
        Args:
            other: MSA to merge with
            
        Returns:
            New MSA with sequences from both
        """
        # Verify compatibility
        if self.width > 0 and other.width > 0 and self.width != other.width:
            raise ValueError(f"MSA widths don't match: {self.width} vs {other.width}")

        return MSA(
            sequences=self.sequences + other.sequences,
            query_sequence=self.query_sequence,
            database=f"{self.database},{other.database}",
        )
