"""Jackhmmer sequence search implementation.

This module provides a wrapper for the Jackhmmer tool from HMMER suite,
configured as described in AlphaFold3 supplement Section 2.2.

Default flags: -N 1 -E 0.0001 --incE 0.0001 --F1 0.0005 --F2 0.00005 --F3 0.0000005
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from novadb.search.msa.msa import MSA, MSASequence
from novadb.config import JackhmmerConfig


@dataclass
class JackhmmerResult:
    """Result from a Jackhmmer search."""
    msa: MSA
    database: str
    num_hits: int
    execution_time: float


class JackhmmerSearch:
    """Jackhmmer sequence search for protein MSA generation.
    
    From AF3 supplement Table 1, Jackhmmer is used to search:
    - UniRef90 (--seq_limit 100000, max 10,000 sequences)
    - UniProt (--seq_limit 500000, max 50,000 sequences)
    - Reduced BFD (--seq_limit 50000, max 5,000 sequences)
    - MGnify (--seq_limit 50000, max 5,000 sequences)
    """

    def __init__(
        self,
        binary_path: str = "jackhmmer",
        config: Optional[JackhmmerConfig] = None,
    ):
        """Initialize Jackhmmer search.
        
        Args:
            binary_path: Path to jackhmmer binary
            config: Search configuration
        """
        self.binary_path = binary_path
        self.config = config or JackhmmerConfig()
        self._verify_binary()

    def _verify_binary(self) -> None:
        """Verify that jackhmmer binary is available."""
        try:
            result = subprocess.run(
                [self.binary_path, "-h"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Jackhmmer binary not functional: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Jackhmmer binary not found at: {self.binary_path}. "
                "Please install HMMER: conda install -c bioconda hmmer"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Jackhmmer binary timed out during verification")

    def search(
        self,
        sequence: str,
        database_path: Union[str, Path],
        database_name: str = "unknown",
        seq_limit: Optional[int] = None,
        max_sequences: Optional[int] = None,
        z_value: Optional[int] = None,
        n_cpu: int = 4,
    ) -> JackhmmerResult:
        """Run Jackhmmer search against a database.
        
        Args:
            sequence: Query protein sequence
            database_path: Path to FASTA database
            database_name: Name of the database for tracking
            seq_limit: Override --seq_limit parameter
            max_sequences: Maximum sequences to return
            z_value: Database size for E-value calculation (-Z flag)
            n_cpu: Number of CPU cores to use
            
        Returns:
            JackhmmerResult with MSA and metadata
        """
        import time
        start_time = time.time()

        database_path = Path(database_path)
        if not database_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")

        # Create temporary files for input and output
        with tempfile.TemporaryDirectory() as tmpdir:
            query_file = Path(tmpdir) / "query.fasta"
            output_file = Path(tmpdir) / "output.sto"

            # Write query sequence
            with open(query_file, "w") as f:
                f.write(f">query\n{sequence}\n")

            # Build command
            cmd = self._build_command(
                query_file=query_file,
                database_path=database_path,
                output_file=output_file,
                seq_limit=seq_limit,
                z_value=z_value,
                n_cpu=n_cpu,
            )

            # Run Jackhmmer
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Jackhmmer failed: {result.stderr}")

            # Parse output
            if output_file.exists():
                with open(output_file) as f:
                    msa = MSA.from_stockholm(f.read(), database=database_name)
            else:
                # No hits found
                msa = MSA(
                    sequences=[MSASequence(sequence=sequence, description="query")],
                    query_sequence=sequence,
                    database=database_name,
                )

        # Deduplicate and crop
        msa = msa.deduplicate()
        if max_sequences:
            msa = msa.crop(max_sequences)

        execution_time = time.time() - start_time

        return JackhmmerResult(
            msa=msa,
            database=database_name,
            num_hits=msa.depth - 1,  # Exclude query
            execution_time=execution_time,
        )

    def _build_command(
        self,
        query_file: Path,
        database_path: Path,
        output_file: Path,
        seq_limit: Optional[int],
        z_value: Optional[int],
        n_cpu: int,
    ) -> List[str]:
        """Build Jackhmmer command with AF3-specified flags."""
        cmd = [
            self.binary_path,
            "-N", str(self.config.num_iterations),
            "-E", str(self.config.e_value),
            "--incE", str(self.config.inc_e_value),
            "--F1", str(self.config.f1),
            "--F2", str(self.config.f2),
            "--F3", str(self.config.f3),
            "--cpu", str(n_cpu),
            "-A", str(output_file),  # Output Stockholm alignment
        ]

        if seq_limit:
            cmd.extend(["--seq_limit", str(seq_limit)])

        if z_value:
            cmd.extend(["-Z", str(z_value)])

        cmd.extend([str(query_file), str(database_path)])

        return cmd

    def search_uniref90(
        self,
        sequence: str,
        database_path: Union[str, Path],
        n_cpu: int = 4,
    ) -> JackhmmerResult:
        """Search UniRef90 with AF3 parameters.
        
        From Table 1: --seq_limit 100000, max 10,000 sequences
        """
        return self.search(
            sequence=sequence,
            database_path=database_path,
            database_name="uniref90",
            seq_limit=self.config.uniref90_seq_limit,
            max_sequences=self.config.uniref90_max_seqs,
            n_cpu=n_cpu,
        )

    def search_uniprot(
        self,
        sequence: str,
        database_path: Union[str, Path],
        z_value: int = 138515945,  # UniProt v2021_04 size
        n_cpu: int = 4,
    ) -> JackhmmerResult:
        """Search UniProt with AF3 parameters.
        
        From Table 1: --seq_limit 500000, max 50,000 sequences
        Requires -Z flag for database size
        """
        return self.search(
            sequence=sequence,
            database_path=database_path,
            database_name="uniprot",
            seq_limit=self.config.uniprot_seq_limit,
            max_sequences=self.config.uniprot_max_seqs,
            z_value=z_value,
            n_cpu=n_cpu,
        )

    def search_reduced_bfd(
        self,
        sequence: str,
        database_path: Union[str, Path],
        n_cpu: int = 4,
    ) -> JackhmmerResult:
        """Search Reduced BFD with AF3 parameters.
        
        From Table 1: --seq_limit 50000, max 5,000 sequences
        """
        return self.search(
            sequence=sequence,
            database_path=database_path,
            database_name="reduced_bfd",
            seq_limit=self.config.reduced_bfd_seq_limit,
            max_sequences=self.config.reduced_bfd_max_seqs,
            n_cpu=n_cpu,
        )

    def search_mgnify(
        self,
        sequence: str,
        database_path: Union[str, Path],
        n_cpu: int = 4,
    ) -> JackhmmerResult:
        """Search MGnify with AF3 parameters.
        
        From Table 1: --seq_limit 50000, max 5,000 sequences
        """
        return self.search(
            sequence=sequence,
            database_path=database_path,
            database_name="mgnify",
            seq_limit=self.config.mgnify_seq_limit,
            max_sequences=self.config.mgnify_max_seqs,
            n_cpu=n_cpu,
        )
