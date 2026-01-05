"""MSA processing and pairing for multi-chain complexes.

This module implements MSA processing as described in AlphaFold3 
supplement Section 2.3:
- Construct MSA with up to 16,384 rows (Nmsa ≤ 16,384)
- First row is the query sequence
- Next rows (up to 8,191) from UniProt pairing by species
- Remaining rows from dense stacking of individual chain MSAs
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from novadb.search.msa.msa import MSA, MSASequence
from novadb.config import MSAProcessingConfig


@dataclass
class PairedMSA:
    """Represents a paired MSA for a multi-chain complex.
    
    From AF3 supplement Section 2.3:
    The MSA is constructed by pairing sequences from different chains
    based on species information, then filling with dense stacking.
    """
    msa: np.ndarray  # Shape: (Nmsa, Ntoken, num_classes)
    deletion_matrix: np.ndarray  # Shape: (Nmsa, Ntoken)
    chain_lengths: List[int]  # Length of each chain
    species: List[str]  # Species for each MSA row


class MSAProcessor:
    """Processes MSAs for multi-chain complexes.
    
    Implements the MSA processing pipeline from AF3 Section 2.3:
    1. Parse and clean individual MSAs
    2. Pair sequences across chains by species
    3. Stack remaining sequences densely
    4. Crop to maximum MSA depth
    """

    def __init__(self, config: Optional[MSAProcessingConfig] = None):
        """Initialize the MSA processor.
        
        Args:
            config: MSA processing configuration
        """
        self.config = config or MSAProcessingConfig()

    def process_single_chain(
        self,
        msas: List[MSA],
        query_sequence: str,
    ) -> MSA:
        """Process MSAs for a single chain.
        
        Combines MSAs from different databases into a single MSA,
        with deduplication and cropping.
        
        Args:
            msas: List of MSAs from different databases
            query_sequence: The query sequence
            
        Returns:
            Combined and processed MSA
        """
        if not msas:
            return MSA(
                sequences=[MSASequence(sequence=query_sequence, description="query")],
                query_sequence=query_sequence,
            )

        # Start with query sequence
        combined = MSA(
            sequences=[MSASequence(sequence=query_sequence, description="query")],
            query_sequence=query_sequence,
        )

        # Stack MSAs from different databases
        for msa in msas:
            # Skip query sequence (already added)
            for seq in msa.sequences[1:]:
                combined.sequences.append(seq)

        # Deduplicate
        combined = combined.deduplicate()

        # Crop to max rows
        combined = combined.crop(self.config.max_msa_rows)

        return combined

    def process_multi_chain(
        self,
        chain_msas: Dict[str, List[MSA]],
        chain_sequences: Dict[str, str],
        uniprot_msas: Optional[Dict[str, MSA]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process MSAs for a multi-chain complex.
        
        From AF3 supplement Section 2.3:
        1. First row is query sequences (concatenated)
        2. Next up to 8,191 rows from UniProt pairing by species
        3. Remaining rows filled densely from individual chain MSAs
        
        Args:
            chain_msas: Dict mapping chain_id to list of MSAs (non-UniProt)
            chain_sequences: Dict mapping chain_id to query sequence
            uniprot_msas: Dict mapping chain_id to UniProt MSA (for pairing)
            
        Returns:
            Tuple of (msa_array, deletion_matrix)
            - msa_array: Shape (Nmsa, total_tokens, 32)
            - deletion_matrix: Shape (Nmsa, total_tokens)
        """
        chain_ids = sorted(chain_sequences.keys())
        chain_lengths = [len(chain_sequences[cid]) for cid in chain_ids]
        total_length = sum(chain_lengths)

        # Compute chain offsets
        chain_offsets = {}
        offset = 0
        for cid in chain_ids:
            chain_offsets[cid] = offset
            offset += len(chain_sequences[cid])

        # Initialize output arrays
        max_rows = self.config.max_msa_rows
        msa_array = np.zeros((max_rows, total_length), dtype=np.int32)
        deletion_matrix = np.zeros((max_rows, total_length), dtype=np.int32)

        # Row 0: Query sequences
        row = 0
        for cid in chain_ids:
            seq = chain_sequences[cid]
            start = chain_offsets[cid]
            for j, char in enumerate(seq):
                msa_array[row, start + j] = self._residue_to_int(char)
        row += 1

        # Rows 1 to max_paired_rows: Species-paired sequences from UniProt
        if uniprot_msas:
            paired_rows = self._pair_by_species(
                uniprot_msas,
                chain_ids,
                chain_offsets,
                chain_lengths,
                max_rows=self.config.max_paired_rows,
            )
            
            for paired_row_data in paired_rows:
                if row >= max_rows:
                    break
                msa_array[row] = paired_row_data["msa"]
                deletion_matrix[row] = paired_row_data["deletions"]
                row += 1

        # Remaining rows: Dense stacking from other MSAs
        # From AF3 Section 2.3: Use dense stacking with sequences sorted by gap count
        remaining_rows = max_rows - row

        if remaining_rows > 0:
            row = self._dense_stack(
                msa_array=msa_array,
                deletion_matrix=deletion_matrix,
                start_row=row,
                max_rows=max_rows,
                chain_msas=chain_msas,
                chain_sequences=chain_sequences,
                chain_ids=chain_ids,
                chain_offsets=chain_offsets,
                chain_lengths=chain_lengths,
            )

        # Trim to actual number of rows
        msa_array = msa_array[:row]
        deletion_matrix = deletion_matrix[:row]

        return msa_array, deletion_matrix

    def _dense_stack(
        self,
        msa_array: np.ndarray,
        deletion_matrix: np.ndarray,
        start_row: int,
        max_rows: int,
        chain_msas: Dict[str, List[MSA]],
        chain_sequences: Dict[str, str],
        chain_ids: List[str],
        chain_offsets: Dict[str, int],
        chain_lengths: List[int],
    ) -> int:
        """Dense stacking of MSA sequences from individual chains.

        From AF3 Section 2.3: After species-paired rows, remaining MSA rows
        are filled using dense stacking. This selects sequences with the
        highest coverage (fewest gaps) and interleaves across chains.

        Args:
            msa_array: MSA array to fill
            deletion_matrix: Deletion matrix to fill
            start_row: Starting row index
            max_rows: Maximum number of rows
            chain_msas: MSAs for each chain
            chain_sequences: Query sequences for each chain
            chain_ids: Ordered chain IDs
            chain_offsets: Position offset for each chain
            chain_lengths: Length of each chain

        Returns:
            Final row index after stacking
        """
        row = start_row

        # Collect sequences from all chains with their gap counts
        # Format: List of (chain_id, sequence, deletions, gap_count, coverage)
        all_sequences = []

        for cid in chain_ids:
            chain_len = chain_lengths[chain_ids.index(cid)]

            # Get combined MSA for this chain
            combined_msa = self.process_single_chain(
                chain_msas.get(cid, []),
                chain_sequences[cid],
            )

            # Skip query sequence (already in row 0)
            for seq in combined_msa.sequences[1:]:
                seq_str = seq.sequence[:chain_len]
                gap_count = seq_str.count('-') + seq_str.count('.')
                coverage = 1.0 - (gap_count / max(1, len(seq_str)))

                all_sequences.append({
                    'chain_id': cid,
                    'sequence': seq,
                    'gap_count': gap_count,
                    'coverage': coverage,
                    'length': chain_len,
                })

        # Sort by coverage (highest first = fewest gaps)
        all_sequences.sort(key=lambda x: (-x['coverage'], x['chain_id']))

        # Interleave sequences from different chains
        # Group by chain, then round-robin select
        chain_queues: Dict[str, List] = {cid: [] for cid in chain_ids}
        for seq_info in all_sequences:
            chain_queues[seq_info['chain_id']].append(seq_info)

        # Round-robin selection across chains
        chain_indices = {cid: 0 for cid in chain_ids}
        active_chains = [cid for cid in chain_ids if chain_queues[cid]]

        while row < max_rows and active_chains:
            chains_to_remove = []

            for cid in active_chains:
                if row >= max_rows:
                    break

                queue = chain_queues[cid]
                idx = chain_indices[cid]

                if idx >= len(queue):
                    chains_to_remove.append(cid)
                    continue

                seq_info = queue[idx]
                chain_indices[cid] += 1

                # Fill this row for this chain
                chain_start = chain_offsets[cid]
                chain_len = seq_info['length']
                seq = seq_info['sequence']

                for j, char in enumerate(seq.sequence[:chain_len]):
                    msa_array[row, chain_start + j] = self._residue_to_int(char)

                if seq.deletions:
                    deletion_matrix[row, chain_start:chain_start + len(seq.deletions)] = (
                        np.array(seq.deletions[:chain_len], dtype=np.int32)
                    )

                row += 1

            # Remove exhausted chains
            for cid in chains_to_remove:
                active_chains.remove(cid)

        return row

    def _pair_by_species(
        self,
        uniprot_msas: Dict[str, MSA],
        chain_ids: List[str],
        chain_offsets: Dict[str, int],
        chain_lengths: List[int],
        max_rows: int,
    ) -> List[Dict]:
        """Pair sequences from different chains by species.
        
        From AF3 Section 2.3 and AlphaFold-Multimer:
        Sequences are paired if they come from the same species.
        
        Args:
            uniprot_msas: UniProt MSAs for each chain
            chain_ids: Ordered list of chain IDs
            chain_offsets: Offset for each chain in the concatenated MSA
            chain_lengths: Length of each chain
            max_rows: Maximum paired rows to return
            
        Returns:
            List of paired row dictionaries with 'msa' and 'deletions'
        """
        # Group sequences by species for each chain
        chain_species_seqs: Dict[str, Dict[str, List[MSASequence]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for cid in chain_ids:
            msa = uniprot_msas.get(cid)
            if not msa:
                continue

            for seq in msa.sequences[1:]:  # Skip query
                species = self._extract_species(seq.description)
                if species:
                    chain_species_seqs[cid][species].append(seq)

        # Find species present in all chains
        if not chain_species_seqs:
            return []

        all_species = set()
        for cid in chain_ids:
            if cid in chain_species_seqs:
                if not all_species:
                    all_species = set(chain_species_seqs[cid].keys())
                else:
                    all_species &= set(chain_species_seqs[cid].keys())

        # Create paired rows
        paired_rows = []
        total_length = sum(chain_lengths)

        for species in sorted(all_species):
            # Get sequences for each chain
            chain_seqs = []
            for cid in chain_ids:
                seqs = chain_species_seqs[cid].get(species, [])
                if not seqs:
                    break
                chain_seqs.append((cid, seqs))

            if len(chain_seqs) != len(chain_ids):
                continue

            # Create all combinations (simplified: just take first match per chain)
            row_msa = np.zeros(total_length, dtype=np.int32)
            row_deletions = np.zeros(total_length, dtype=np.int32)

            for i, (cid, seqs) in enumerate(chain_seqs):
                seq = seqs[0]  # Take first sequence for this species
                start = chain_offsets[cid]
                length = chain_lengths[i]

                for j, char in enumerate(seq.sequence[:length]):
                    row_msa[start + j] = self._residue_to_int(char)

                if seq.deletions:
                    row_deletions[start:start + len(seq.deletions)] = (
                        np.array(seq.deletions[:length], dtype=np.int32)
                    )

            paired_rows.append({
                "msa": row_msa,
                "deletions": row_deletions,
                "species": species,
            })

            if len(paired_rows) >= max_rows:
                break

        return paired_rows

    def _extract_species(self, description: str) -> Optional[str]:
        """Extract species name from sequence description.
        
        Handles various formats:
        - UniProt: "sp|P12345|PROT_HUMAN ... OS=Homo sapiens ..."
        - General: "... [Homo sapiens]"
        """
        # Try UniProt format: OS=Species Name
        match = re.search(r'OS=([^=]+?)(?:\s+OX=|\s+GN=|\s+PE=|\s*$)', description)
        if match:
            return match.group(1).strip()

        # Try bracket format: [Species Name]
        match = re.search(r'\[([^\]]+)\]', description)
        if match:
            return match.group(1).strip()

        # Try to extract from UniProt ID: _SPECIES
        match = re.search(r'_([A-Z]+)\s', description)
        if match:
            return match.group(1)

        return None

    def _residue_to_int(self, char: str) -> int:
        """Convert residue character to integer code.
        
        Uses 32 classes as per AF3 Table 5 (restype feature).
        """
        mapping = {
            # 20 amino acids (0-19)
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
            # Unknown amino acid (20)
            'X': 20,
            # Gap (31)
            '-': 31,
        }
        return mapping.get(char.upper(), 20)  # Default to unknown

    def compute_profile(self, msa: np.ndarray) -> np.ndarray:
        """Compute sequence profile from MSA array.
        
        From AF3 Table 5: Distribution across restypes in the main MSA.
        
        Args:
            msa: MSA array of shape (Nmsa, Ntoken) with integer codes
            
        Returns:
            Profile of shape (Ntoken, 32) with residue probabilities
        """
        num_classes = 32
        depth, width = msa.shape

        profile = np.zeros((width, num_classes), dtype=np.float32)

        for j in range(width):
            for code in range(num_classes):
                profile[j, code] = np.sum(msa[:, j] == code)

        # Normalize
        row_sums = profile.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        profile = profile / row_sums

        return profile

    def compute_deletion_mean(self, deletion_matrix: np.ndarray) -> np.ndarray:
        """Compute mean deletion count at each position.
        
        From AF3 Table 5: Mean number of deletions at each position.
        
        Args:
            deletion_matrix: Deletion counts of shape (Nmsa, Ntoken)
            
        Returns:
            Mean deletions of shape (Ntoken,)
        """
        return deletion_matrix.mean(axis=0).astype(np.float32)

    def subsample_msa(
        self,
        msa: np.ndarray,
        deletion_matrix: np.ndarray,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly subsample MSA during training.
        
        From AF3 Section 2.2:
        "During training, the main MSA for each sequence is subsampled 
        from size n to size k = Uniform[1, n]"
        
        Args:
            msa: MSA array of shape (Nmsa, Ntoken)
            deletion_matrix: Deletion matrix of shape (Nmsa, Ntoken)
            random_state: Random seed
            
        Returns:
            Tuple of subsampled (msa, deletion_matrix)
        """
        rng = np.random.default_rng(random_state)
        n = msa.shape[0]
        k = rng.integers(1, n + 1)

        # Always keep first row (query)
        indices = [0]

        # Randomly sample remaining rows
        if k > 1 and n > 1:
            remaining = list(range(1, n))
            sampled = rng.choice(remaining, size=min(k - 1, len(remaining)), replace=False)
            indices.extend(sorted(sampled))

        return msa[indices], deletion_matrix[indices]
