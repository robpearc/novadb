"""Sequence clustering for dataset curation.

Implements sequence clustering from AlphaFold3 Section 2.5:
- Protein chains: 40% sequence identity (MMseqs2)
- Nucleic acids: 100% sequence identity
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import os


@dataclass
class ClusterResult:
    """Result of sequence clustering."""
    cluster_id: str
    representative_id: str
    members: List[str]
    size: int


class SequenceClusterer:
    """Cluster sequences using MMseqs2.
    
    From AF3 Section 2.5:
    - Proteins: 40% sequence identity
    - Nucleic acids: 100% sequence identity
    """

    def __init__(
        self,
        mmseqs_binary: str = "mmseqs",
        tmp_dir: Optional[str] = None,
    ):
        self.mmseqs = mmseqs_binary
        self.tmp_dir = tmp_dir or tempfile.gettempdir()

    def cluster_proteins(
        self,
        fasta_file: str,
        output_prefix: str,
        min_seq_id: float = 0.4,
        coverage: float = 0.8,
        coverage_mode: int = 1,
    ) -> Dict[str, ClusterResult]:
        """Cluster protein sequences.
        
        Args:
            fasta_file: Path to input FASTA file
            output_prefix: Prefix for output files
            min_seq_id: Minimum sequence identity (default 0.4 = 40%)
            coverage: Minimum coverage (default 0.8 = 80%)
            coverage_mode: Coverage mode (0=bidirectional, 1=target, 2=query)
            
        Returns:
            Dictionary mapping sequence IDs to cluster results
        """
        return self._run_mmseqs(
            fasta_file,
            output_prefix,
            min_seq_id=min_seq_id,
            coverage=coverage,
            coverage_mode=coverage_mode,
        )

    def cluster_nucleic_acids(
        self,
        fasta_file: str,
        output_prefix: str,
        min_seq_id: float = 1.0,
    ) -> Dict[str, ClusterResult]:
        """Cluster nucleic acid sequences.
        
        Args:
            fasta_file: Path to input FASTA file
            output_prefix: Prefix for output files
            min_seq_id: Minimum sequence identity (default 1.0 = 100%)
            
        Returns:
            Dictionary mapping sequence IDs to cluster results
        """
        return self._run_mmseqs(
            fasta_file,
            output_prefix,
            min_seq_id=min_seq_id,
            coverage=0.8,
            coverage_mode=1,
        )

    def _run_mmseqs(
        self,
        fasta_file: str,
        output_prefix: str,
        min_seq_id: float,
        coverage: float,
        coverage_mode: int,
    ) -> Dict[str, ClusterResult]:
        """Run MMseqs2 clustering."""
        db_path = f"{output_prefix}_db"
        cluster_path = f"{output_prefix}_cluster"
        result_path = f"{output_prefix}_cluster.tsv"

        try:
            # Create database
            subprocess.run(
                [self.mmseqs, "createdb", fasta_file, db_path],
                check=True,
                capture_output=True,
            )

            # Run clustering
            subprocess.run(
                [
                    self.mmseqs,
                    "cluster",
                    db_path,
                    cluster_path,
                    self.tmp_dir,
                    "--min-seq-id",
                    str(min_seq_id),
                    "-c",
                    str(coverage),
                    "--cov-mode",
                    str(coverage_mode),
                ],
                check=True,
                capture_output=True,
            )

            # Create TSV output
            subprocess.run(
                [
                    self.mmseqs,
                    "createtsv",
                    db_path,
                    db_path,
                    cluster_path,
                    result_path,
                ],
                check=True,
                capture_output=True,
            )

            # Parse results
            return self._parse_cluster_tsv(result_path)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"MMseqs2 failed: {e.stderr.decode()}")

    def _parse_cluster_tsv(
        self, tsv_path: str
    ) -> Dict[str, ClusterResult]:
        """Parse MMseqs2 cluster TSV output."""
        clusters: Dict[str, List[str]] = {}

        with open(tsv_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rep_id, member_id = parts[0], parts[1]
                    if rep_id not in clusters:
                        clusters[rep_id] = []
                    clusters[rep_id].append(member_id)

        # Convert to ClusterResult
        results = {}
        for i, (rep_id, members) in enumerate(clusters.items()):
            cluster_id = f"cluster_{i}"
            result = ClusterResult(
                cluster_id=cluster_id,
                representative_id=rep_id,
                members=members,
                size=len(members),
            )
            for member in members:
                results[member] = result

        return results


class IdentityClusterer:
    """Simple 100% identity clustering for nucleic acids.
    
    Doesn't require external tools.
    """

    def cluster(
        self, sequences: Dict[str, str]
    ) -> Dict[str, ClusterResult]:
        """Cluster sequences by 100% identity.
        
        Args:
            sequences: Dictionary of sequence_id -> sequence
            
        Returns:
            Dictionary mapping sequence IDs to cluster results
        """
        # Group by sequence
        seq_to_ids: Dict[str, List[str]] = {}
        for seq_id, seq in sequences.items():
            seq_upper = seq.upper().replace("-", "")
            if seq_upper not in seq_to_ids:
                seq_to_ids[seq_upper] = []
            seq_to_ids[seq_upper].append(seq_id)

        # Create cluster results
        results = {}
        for i, (seq, members) in enumerate(seq_to_ids.items()):
            cluster_id = f"cluster_{i}"
            rep_id = members[0]

            result = ClusterResult(
                cluster_id=cluster_id,
                representative_id=rep_id,
                members=members,
                size=len(members),
            )

            for member in members:
                results[member] = result

        return results


def compute_sequence_identity(
    seq1: str,
    seq2: str,
) -> float:
    """Compute sequence identity between two sequences.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Sequence identity as fraction (0-1)
    """
    if len(seq1) != len(seq2):
        return 0.0

    matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != "-")
    aligned = sum(1 for a, b in zip(seq1, seq2) if a != "-" or b != "-")

    if aligned == 0:
        return 0.0

    return matches / aligned


def extract_sequences_from_structures(
    structures: List,  # List[Structure]
    chain_type_filter: Optional[str] = None,
) -> Dict[str, str]:
    """Extract sequences from structures for clustering.
    
    Args:
        structures: List of Structure objects
        chain_type_filter: Optional filter for chain type
        
    Returns:
        Dictionary of sequence_id -> sequence
    """
    from novadb.data.parsers.structure import ChainType

    # One-letter codes for amino acids
    aa_3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    sequences = {}

    for structure in structures:
        for chain_id, chain in structure.chains.items():
            # Filter by chain type if specified
            if chain_type_filter:
                if chain.chain_type and chain.chain_type.name != chain_type_filter:
                    continue

            # Build sequence
            seq = []
            for residue in chain.residues:
                if chain.chain_type == ChainType.PROTEIN:
                    letter = aa_3to1.get(residue.name, "X")
                elif chain.chain_type in (ChainType.RNA, ChainType.DNA):
                    # Nucleotides already use single letters
                    letter = residue.name[-1] if len(residue.name) > 0 else "N"
                else:
                    continue

                seq.append(letter)

            if seq:
                seq_id = f"{structure.pdb_id}_{chain_id}"
                sequences[seq_id] = "".join(seq)

    return sequences


def write_fasta(
    sequences: Dict[str, str],
    output_path: str,
) -> None:
    """Write sequences to FASTA file.
    
    Args:
        sequences: Dictionary of sequence_id -> sequence
        output_path: Path to output FASTA file
    """
    with open(output_path, "w") as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n")
            # Write in 80-character lines
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")
