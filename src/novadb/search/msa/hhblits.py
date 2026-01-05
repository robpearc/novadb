"""HHBlits sequence search implementation.

This module provides a wrapper for HHBlits from the HH-suite,
configured as described in AlphaFold3 supplement Section 2.2.

Default flags: -n 3 -e 0.001 -realign_max 100000 -maxfilt 100000 
               -min_prefilter_hits 1000 -p 20 -Z 500
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from novadb.search.msa.msa import MSA, MSASequence
from novadb.config import HHBlitsConfig


@dataclass
class HHBlitsResult:
    """Result from an HHBlits search."""
    msa: MSA
    database: str
    num_hits: int
    execution_time: float


class HHBlitsSearch:
    """HHBlits HMM-HMM search for protein MSA generation.
    
    From AF3 supplement Table 1, HHBlits is used to search:
    - Uniclust30 + BFD (combined database)
    
    This search typically yields deeper alignments than Jackhmmer
    for distantly related sequences.
    """

    def __init__(
        self,
        binary_path: str = "hhblits",
        config: Optional[HHBlitsConfig] = None,
    ):
        """Initialize HHBlits search.
        
        Args:
            binary_path: Path to hhblits binary
            config: Search configuration
        """
        self.binary_path = binary_path
        self.config = config or HHBlitsConfig()
        self._verify_binary()

    def _verify_binary(self) -> None:
        """Verify that hhblits binary is available."""
        try:
            result = subprocess.run(
                [self.binary_path, "-h"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # HHBlits returns non-zero even with -h, so just check it runs
        except FileNotFoundError:
            raise RuntimeError(
                f"HHBlits binary not found at: {self.binary_path}. "
                "Please install HH-suite: conda install -c bioconda hhsuite"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("HHBlits binary timed out during verification")

    def search(
        self,
        sequence: str,
        database_path: Union[str, Path],
        database_name: str = "uniclust30_bfd",
        n_cpu: int = 4,
    ) -> HHBlitsResult:
        """Run HHBlits search against a database.
        
        Args:
            sequence: Query protein sequence
            database_path: Path to HHBlits database (without extension)
            database_name: Name of the database for tracking
            n_cpu: Number of CPU cores to use
            
        Returns:
            HHBlitsResult with MSA and metadata
        """
        import time
        start_time = time.time()

        database_path = Path(database_path)

        # Create temporary files for input and output
        with tempfile.TemporaryDirectory() as tmpdir:
            query_file = Path(tmpdir) / "query.fasta"
            output_file = Path(tmpdir) / "output.a3m"

            # Write query sequence
            with open(query_file, "w") as f:
                f.write(f">query\n{sequence}\n")

            # Build command
            cmd = self._build_command(
                query_file=query_file,
                database_path=database_path,
                output_file=output_file,
                n_cpu=n_cpu,
            )

            # Run HHBlits
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                # HHBlits may return non-zero for various reasons, check output
                if not output_file.exists():
                    raise RuntimeError(f"HHBlits failed: {result.stderr}")

            # Parse output
            if output_file.exists():
                with open(output_file) as f:
                    msa = MSA.from_a3m(f.read(), database=database_name)
            else:
                # No hits found
                msa = MSA(
                    sequences=[MSASequence(sequence=sequence, description="query")],
                    query_sequence=sequence,
                    database=database_name,
                )

        execution_time = time.time() - start_time

        return HHBlitsResult(
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
        n_cpu: int,
    ) -> List[str]:
        """Build HHBlits command with AF3-specified flags."""
        cmd = [
            self.binary_path,
            "-n", str(self.config.num_iterations),
            "-e", str(self.config.e_value),
            "-realign_max", str(self.config.realign_max),
            "-maxfilt", str(self.config.maxfilt),
            "-min_prefilter_hits", str(self.config.min_prefilter_hits),
            "-p", str(self.config.min_prob),
            "-Z", str(self.config.max_seqs),
            "-cpu", str(n_cpu),
            "-i", str(query_file),
            "-d", str(database_path),
            "-oa3m", str(output_file),
        ]

        return cmd

    def search_uniclust30_bfd(
        self,
        sequence: str,
        uniclust30_path: Union[str, Path],
        bfd_path: Optional[Union[str, Path]] = None,
        n_cpu: int = 4,
    ) -> HHBlitsResult:
        """Search Uniclust30 + BFD with AF3 parameters.
        
        From Table 1: Combined Uniclust30 (v2018_08) + BFD database.
        
        Note: BFD is typically a very large database. If bfd_path is provided,
        it will search both databases sequentially.
        """
        # First search Uniclust30
        result = self.search(
            sequence=sequence,
            database_path=uniclust30_path,
            database_name="uniclust30",
            n_cpu=n_cpu,
        )

        # Optionally search BFD if provided
        if bfd_path:
            bfd_result = self.search(
                sequence=sequence,
                database_path=bfd_path,
                database_name="bfd",
                n_cpu=n_cpu,
            )

            # Merge MSAs
            merged_msa = result.msa.merge(bfd_result.msa)
            merged_msa = merged_msa.deduplicate()

            return HHBlitsResult(
                msa=merged_msa,
                database="uniclust30_bfd",
                num_hits=merged_msa.depth - 1,
                execution_time=result.execution_time + bfd_result.execution_time,
            )

        return result
