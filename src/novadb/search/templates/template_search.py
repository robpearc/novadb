"""Template search for structure prediction.

Implements template search from AlphaFold3 Section 2.4:
- Build HMM from query MSA using hmmbuild
- Search against PDB70 or full PDB using hmmsearch
- Filter templates by date and quality

From AF3 supplement Section 2.4:
- Templates must be released at least 60 days before the example's release date
  for PDB training data
- For distillation data, template cutoff is 2018-04-30
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

# Template release date offset from AF3 Section 2.4
TEMPLATE_RELEASE_OFFSET_DAYS = 60


@dataclass
class TemplateHit:
    """A template search hit."""
    template_id: str  # e.g., "1abc_A"
    pdb_id: str
    chain_id: str
    hit_sequence: str
    query_start: int
    query_end: int
    template_start: int
    template_end: int
    e_value: float
    score: float
    identity: float
    aligned_query: str
    aligned_template: str
    release_date: Optional[date] = None


@dataclass
class TemplateSearchResult:
    """Result of template search."""
    query_id: str
    query_sequence: str
    hits: List[TemplateHit] = field(default_factory=list)

    @property
    def num_hits(self) -> int:
        return len(self.hits)


class TemplateSearcher:
    """Search for structural templates using HMM-based methods.
    
    From AF3 Section 2.4:
    - Build HMM from query MSA
    - Search against PDB70 database
    - Filter by release date for distillation
    """

    def __init__(
        self,
        hmmbuild_binary: str = "hmmbuild",
        hmmsearch_binary: str = "hmmsearch",
        pdb_database: str = "",
        pdb_mmcif_dir: str = "",
        max_templates: int = 20,
        max_subsequence_ratio: float = 0.95,
        min_align_ratio: float = 0.1,
    ):
        """Initialize template searcher.
        
        Args:
            hmmbuild_binary: Path to hmmbuild
            hmmsearch_binary: Path to hmmsearch
            pdb_database: Path to PDB sequence database (FASTA)
            pdb_mmcif_dir: Path to directory of mmCIF files
            max_templates: Maximum templates to return
            max_subsequence_ratio: Max ratio for subsequence filter
            min_align_ratio: Minimum alignment coverage
        """
        self.hmmbuild = hmmbuild_binary
        self.hmmsearch = hmmsearch_binary
        self.pdb_database = pdb_database
        self.pdb_mmcif_dir = pdb_mmcif_dir
        self.max_templates = max_templates
        self.max_subsequence_ratio = max_subsequence_ratio
        self.min_align_ratio = min_align_ratio

        # Cache for release dates
        self._release_dates: Dict[str, date] = {}

    def search(
        self,
        query_sequence: str,
        msa_a3m: str,
        max_template_date: Optional[date] = None,
        example_release_date: Optional[date] = None,
    ) -> TemplateSearchResult:
        """Search for templates.

        Args:
            query_sequence: Query protein sequence
            msa_a3m: MSA in A3M format
            max_template_date: Maximum release date for templates (explicit cutoff)
            example_release_date: Release date of the training example.
                If provided, templates must be released at least 60 days
                before this date (from AF3 Section 2.4).

        Returns:
            TemplateSearchResult with hits

        Note:
            From AF3 Section 2.4: For PDB training data, templates are filtered
            to those released at least 60 days before the example's release date.
            For distillation data, a fixed cutoff of 2018-04-30 is used.
        """
        # Apply 60-day offset if example_release_date is provided
        if example_release_date is not None and max_template_date is None:
            max_template_date = example_release_date - timedelta(days=TEMPLATE_RELEASE_OFFSET_DAYS)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write MSA to file
            msa_path = tmp_path / "query.a3m"
            msa_path.write_text(msa_a3m)

            # Build HMM
            hmm_path = tmp_path / "query.hmm"
            self._run_hmmbuild(msa_path, hmm_path)

            # Search database
            output_path = tmp_path / "search.out"
            self._run_hmmsearch(hmm_path, output_path)

            # Parse results
            hits = self._parse_hmmsearch_output(output_path, query_sequence)

        # Filter by date if specified
        if max_template_date is not None:
            hits = self._filter_by_date(hits, max_template_date)

        # Apply additional filters
        hits = self._filter_hits(hits, query_sequence)

        # Take top hits
        hits = hits[: self.max_templates]

        return TemplateSearchResult(
            query_id="query",
            query_sequence=query_sequence,
            hits=hits,
        )

    def _run_hmmbuild(self, msa_path: Path, hmm_path: Path) -> None:
        """Run hmmbuild to create HMM from MSA."""
        cmd = [
            self.hmmbuild,
            "--amino",  # Protein sequences
            str(hmm_path),
            str(msa_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"hmmbuild failed: {result.stderr}")

    def _run_hmmsearch(self, hmm_path: Path, output_path: Path) -> None:
        """Run hmmsearch against PDB database.

        From AF3 Section 2.4, hmmsearch flags:
        --noali --F1 0.1 --F2 0.1 --F3 0.1 --E 100 --incE 100
        --domE 100 --incdomE 100 -Z 10000
        """
        cmd = [
            self.hmmsearch,
            "--noali",  # Don't show alignments in main output
            # AF3 filter thresholds (Section 2.4)
            "--F1", "0.1",
            "--F2", "0.1",
            "--F3", "0.1",
            # AF3 E-value thresholds
            "-E", "100",
            "--incE", "100",
            "--domE", "100",
            "--incdomE", "100",
            # Report top Z sequences (AF3 uses 10000 for search, then filters to 20)
            "-Z", "10000",
            "-A", str(output_path.with_suffix(".sto")),  # Alignment output
            "-o", str(output_path),  # Main output
            "--tblout", str(output_path.with_suffix(".tbl")),  # Table output
            str(hmm_path),
            self.pdb_database,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"hmmsearch failed: {result.stderr}")

    def _parse_hmmsearch_output(
        self,
        output_path: Path,
        query_sequence: str,
    ) -> List[TemplateHit]:
        """Parse hmmsearch output files."""
        hits = []

        # Parse table output for scores
        tbl_path = output_path.with_suffix(".tbl")
        if not tbl_path.exists():
            return hits

        scores = {}
        with open(tbl_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    target_name = parts[0]
                    e_value = float(parts[4])
                    score = float(parts[5])
                    scores[target_name] = (e_value, score)

        # Parse alignment output
        sto_path = output_path.with_suffix(".sto")
        if sto_path.exists():
            hits = self._parse_stockholm_alignments(
                sto_path, query_sequence, scores
            )

        return hits

    def _parse_stockholm_alignments(
        self,
        sto_path: Path,
        query_sequence: str,
        scores: Dict[str, Tuple[float, float]],
    ) -> List[TemplateHit]:
        """Parse Stockholm alignment file."""
        hits = []

        with open(sto_path, "r") as f:
            content = f.read()

        # Simple parsing - would need more robust implementation
        blocks = content.split("//")

        for block in blocks:
            if not block.strip():
                continue

            lines = block.strip().split("\n")
            alignments = {}

            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    seq_id = parts[0]
                    seq = parts[1]
                    if seq_id not in alignments:
                        alignments[seq_id] = ""
                    alignments[seq_id] += seq

            # Extract hit information
            for seq_id, aligned_seq in alignments.items():
                if "/" in seq_id:
                    # Template format: 1abc_A/1-100
                    base_id = seq_id.split("/")[0]
                else:
                    base_id = seq_id

                if "_" in base_id:
                    pdb_id = base_id[:4].lower()
                    chain_id = base_id[5:]
                else:
                    pdb_id = base_id[:4].lower()
                    chain_id = "A"

                e_value, score = scores.get(base_id, (1.0, 0.0))

                # Calculate identity
                query_aligned = alignments.get("query", aligned_seq)
                identity = self._calculate_identity(query_aligned, aligned_seq)

                hit = TemplateHit(
                    template_id=base_id,
                    pdb_id=pdb_id,
                    chain_id=chain_id,
                    hit_sequence=aligned_seq.replace("-", ""),
                    query_start=0,
                    query_end=len(query_sequence),
                    template_start=0,
                    template_end=len(aligned_seq.replace("-", "")),
                    e_value=e_value,
                    score=score,
                    identity=identity,
                    aligned_query=query_aligned,
                    aligned_template=aligned_seq,
                )
                hits.append(hit)

        return hits

    def _calculate_identity(
        self, seq1: str, seq2: str
    ) -> float:
        """Calculate sequence identity between aligned sequences."""
        if len(seq1) != len(seq2):
            return 0.0

        matches = 0
        aligned = 0

        for a, b in zip(seq1, seq2):
            if a != "-" and b != "-":
                aligned += 1
                if a == b:
                    matches += 1

        return matches / aligned if aligned > 0 else 0.0

    def _filter_by_date(
        self,
        hits: List[TemplateHit],
        max_date: date,
    ) -> List[TemplateHit]:
        """Filter templates by release date."""
        filtered = []

        for hit in hits:
            release = self._get_release_date(hit.pdb_id)
            if release is None or release <= max_date:
                hit.release_date = release
                filtered.append(hit)

        return filtered

    def _get_release_date(self, pdb_id: str) -> Optional[date]:
        """Get release date for a PDB structure."""
        if pdb_id in self._release_dates:
            return self._release_dates[pdb_id]

        # Try to read from mmCIF file
        mmcif_path = Path(self.pdb_mmcif_dir) / f"{pdb_id}.cif"
        if not mmcif_path.exists():
            mmcif_path = Path(self.pdb_mmcif_dir) / f"{pdb_id.lower()}.cif"

        if mmcif_path.exists():
            try:
                with open(mmcif_path, "r") as f:
                    for line in f:
                        if "_pdbx_database_status.recvd_initial_deposition_date" in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                date_str = parts[-1]
                                release = date.fromisoformat(date_str)
                                self._release_dates[pdb_id] = release
                                return release
            except Exception:
                pass

        return None

    def _filter_hits(
        self,
        hits: List[TemplateHit],
        query_sequence: str,
    ) -> List[TemplateHit]:
        """Apply quality filters to hits.
        
        From AF3: Filter for alignment quality and coverage.
        """
        filtered = []
        query_len = len(query_sequence)

        for hit in hits:
            # Filter by alignment coverage
            align_len = hit.query_end - hit.query_start
            coverage = align_len / query_len

            if coverage < self.min_align_ratio:
                continue

            # Filter subsequences (template much shorter than query)
            template_len = len(hit.hit_sequence)
            if template_len / query_len > self.max_subsequence_ratio:
                # Template is almost as long, likely just a hit on self
                if hit.identity > 0.95:
                    continue

            filtered.append(hit)

        # Sort by score descending
        filtered.sort(key=lambda h: h.score, reverse=True)

        return filtered


def realign_template(
    query_sequence: str,
    template_sequence: str,
    gap_open: float = -10.0,
    gap_extend: float = -1.0,
) -> Tuple[str, str]:
    """Realign template to query using Needleman-Wunsch.
    
    Args:
        query_sequence: Query sequence
        template_sequence: Template sequence
        gap_open: Gap opening penalty
        gap_extend: Gap extension penalty
        
    Returns:
        Tuple of (aligned_query, aligned_template)
    """
    # Simple Needleman-Wunsch implementation
    # In practice, would use Bio.pairwise2 or similar

    m = len(query_sequence)
    n = len(template_sequence)

    # BLOSUM62-like simple scoring
    def score(a: str, b: str) -> float:
        if a == b:
            return 4.0
        return -1.0

    # Initialize DP matrix
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = gap_open + (i - 1) * gap_extend
    for j in range(1, n + 1):
        dp[0][j] = gap_open + (j - 1) * gap_extend

    # Fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1][j - 1] + score(query_sequence[i - 1], template_sequence[j - 1])
            delete = dp[i - 1][j] + gap_extend
            insert = dp[i][j - 1] + gap_extend
            dp[i][j] = max(match, delete, insert)

    # Traceback
    aligned_query = []
    aligned_template = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag = dp[i - 1][j - 1] + score(query_sequence[i - 1], template_sequence[j - 1])
            if dp[i][j] == diag:
                aligned_query.append(query_sequence[i - 1])
                aligned_template.append(template_sequence[j - 1])
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + gap_extend:
            aligned_query.append(query_sequence[i - 1])
            aligned_template.append("-")
            i -= 1
        else:
            aligned_query.append("-")
            aligned_template.append(template_sequence[j - 1])
            j -= 1

    return "".join(reversed(aligned_query)), "".join(reversed(aligned_template))
