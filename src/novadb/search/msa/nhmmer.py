"""Nhmmer sequence search implementation for RNA.

This module provides a wrapper for nhmmer from the HMMER suite,
configured as described in AlphaFold3 supplement Section 2.2.

Default flags: -E 0.001 --incE 0.001 --rna --watson --F3 0.00005
For short sequences (<50 nt): --F3 0.02

RNA databases are pre-processed by filtering to RNA entries only,
then clustering with: --min-seq-id 0.9 -c 0.8
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from novadb.search.msa.msa import MSA, MSASequence
from novadb.config import NhmmerConfig


@dataclass
class NhmmerResult:
    """Result from an nhmmer search."""
    msa: MSA
    database: str
    num_hits: int
    execution_time: float


class NhmmerSearch:
    """Nhmmer nucleotide HMM search for RNA MSA generation.
    
    From AF3 supplement Table 2, nhmmer is used to search:
    - Rfam (max 10,000 sequences)
    - RNACentral (max 10,000 sequences)
    - Nucleotide collection (max 10,000 sequences)
    
    RNA hit sequences are realigned to the query with hmmalign.
    """

    def __init__(
        self,
        binary_path: str = "nhmmer",
        hmmalign_path: str = "hmmalign",
        hmmbuild_path: str = "hmmbuild",
        config: Optional[NhmmerConfig] = None,
    ):
        """Initialize nhmmer search.
        
        Args:
            binary_path: Path to nhmmer binary
            hmmalign_path: Path to hmmalign binary for realignment
            hmmbuild_path: Path to hmmbuild binary
            config: Search configuration
        """
        self.binary_path = binary_path
        self.hmmalign_path = hmmalign_path
        self.hmmbuild_path = hmmbuild_path
        self.config = config or NhmmerConfig()
        self._verify_binary()

    def _verify_binary(self) -> None:
        """Verify that nhmmer binary is available."""
        try:
            result = subprocess.run(
                [self.binary_path, "-h"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"nhmmer binary not functional: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                f"nhmmer binary not found at: {self.binary_path}. "
                "Please install HMMER: conda install -c bioconda hmmer"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("nhmmer binary timed out during verification")

    def search(
        self,
        sequence: str,
        database_path: Union[str, Path],
        database_name: str = "unknown",
        max_sequences: Optional[int] = None,
        n_cpu: int = 4,
        realign: bool = True,
    ) -> NhmmerResult:
        """Run nhmmer search against an RNA database.
        
        Args:
            sequence: Query RNA sequence
            database_path: Path to FASTA database
            database_name: Name of the database for tracking
            max_sequences: Maximum sequences to return
            n_cpu: Number of CPU cores to use
            realign: Whether to realign hits with hmmalign
            
        Returns:
            NhmmerResult with MSA and metadata
        """
        import time
        start_time = time.time()

        database_path = Path(database_path)
        if not database_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")

        # Determine if this is a short sequence
        is_short = len(sequence) < self.config.short_seq_threshold

        # Create temporary files for input and output
        with tempfile.TemporaryDirectory() as tmpdir:
            query_file = Path(tmpdir) / "query.fasta"
            output_file = Path(tmpdir) / "output.sto"
            hits_file = Path(tmpdir) / "hits.fasta"

            # Write query sequence
            with open(query_file, "w") as f:
                f.write(f">query\n{sequence}\n")

            # Build command
            cmd = self._build_command(
                query_file=query_file,
                database_path=database_path,
                output_file=output_file,
                is_short=is_short,
                n_cpu=n_cpu,
            )

            # Run nhmmer
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"nhmmer failed: {result.stderr}")

            # Parse output
            if output_file.exists():
                with open(output_file) as f:
                    content = f.read()
                    if content.strip() and content.strip() != "//":
                        msa = MSA.from_stockholm(content, database=database_name)
                    else:
                        msa = MSA(
                            sequences=[MSASequence(sequence=sequence, description="query")],
                            query_sequence=sequence,
                            database=database_name,
                        )
            else:
                # No hits found
                msa = MSA(
                    sequences=[MSASequence(sequence=sequence, description="query")],
                    query_sequence=sequence,
                    database=database_name,
                )

            # Realign with hmmalign if requested and we have hits
            if realign and msa.depth > 1:
                msa = self._realign_hits(msa, query_file, tmpdir)

        # Crop to max sequences
        if max_sequences:
            msa = msa.crop(max_sequences)

        execution_time = time.time() - start_time

        return NhmmerResult(
            msa=msa,
            database=database_name,
            num_hits=msa.depth - 1,
            execution_time=execution_time,
        )

    def _build_command(
        self,
        query_file: Path,
        database_path: Path,
        output_file: Path,
        is_short: bool,
        n_cpu: int,
    ) -> List[str]:
        """Build nhmmer command with AF3-specified flags."""
        f3 = self.config.f3_short if is_short else self.config.f3

        cmd = [
            self.binary_path,
            "-E", str(self.config.e_value),
            "--incE", str(self.config.inc_e_value),
            "--rna",
            "--watson",
            "--F3", str(f3),
            "--cpu", str(n_cpu),
            "-A", str(output_file),  # Output Stockholm alignment
            str(query_file),
            str(database_path),
        ]

        return cmd

    def _realign_hits(
        self,
        msa: MSA,
        query_file: Path,
        tmpdir: str,
    ) -> MSA:
        """Realign hits to query using hmmalign.
        
        From AF3 supplement Section 2.2:
        "RNA hit sequences are realigned to the query with hmmalign."
        """
        tmpdir = Path(tmpdir)
        hits_file = tmpdir / "hits.fasta"
        hmm_file = tmpdir / "query.hmm"
        aligned_file = tmpdir / "realigned.sto"

        # Write hits to FASTA
        with open(hits_file, "w") as f:
            for seq in msa.sequences:
                f.write(f">{seq.description}\n{seq.sequence.replace('-', '')}\n")

        # Build HMM from query
        subprocess.run(
            [self.hmmbuild_path, str(hmm_file), str(query_file)],
            capture_output=True,
            check=True,
        )

        # Align hits to HMM
        result = subprocess.run(
            [
                self.hmmalign_path,
                "--rna",
                "-o", str(aligned_file),
                str(hmm_file),
                str(hits_file),
            ],
            capture_output=True,
        )

        if result.returncode == 0 and aligned_file.exists():
            with open(aligned_file) as f:
                return MSA.from_stockholm(f.read(), database=msa.database)

        return msa  # Return original if realignment fails

    def search_rfam(
        self,
        sequence: str,
        database_path: Union[str, Path],
        n_cpu: int = 4,
    ) -> NhmmerResult:
        """Search Rfam with AF3 parameters.
        
        From Table 2: Max 10,000 sequences.
        Rfam v14.9 is used.
        """
        return self.search(
            sequence=sequence,
            database_path=database_path,
            database_name="rfam",
            max_sequences=self.config.rfam_max_seqs,
            n_cpu=n_cpu,
        )

    def search_rnacentral(
        self,
        sequence: str,
        database_path: Union[str, Path],
        n_cpu: int = 4,
    ) -> NhmmerResult:
        """Search RNACentral with AF3 parameters.
        
        From Table 2: Max 10,000 sequences.
        RNACentral v21.0 is used.
        """
        return self.search(
            sequence=sequence,
            database_path=database_path,
            database_name="rnacentral",
            max_sequences=self.config.rnacentral_max_seqs,
            n_cpu=n_cpu,
        )

    def search_nt(
        self,
        sequence: str,
        database_path: Union[str, Path],
        n_cpu: int = 4,
    ) -> NhmmerResult:
        """Search NCBI Nucleotide collection with AF3 parameters.
        
        From Table 2: Max 10,000 sequences.
        NT version 2023-02-23 is used.
        """
        return self.search(
            sequence=sequence,
            database_path=database_path,
            database_name="nt",
            max_sequences=self.config.nt_max_seqs,
            n_cpu=n_cpu,
        )
