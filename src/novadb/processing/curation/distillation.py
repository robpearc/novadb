"""Distillation dataset generation for AlphaFold3.

Implements distillation data generation from AF3 Section 2.5:
1. Disordered protein PDB distillation - Low pLDDT regions in PDB
2. Protein monomer distillation - MGnify sequences predicted by AF2
3. RNA distillation - Rfam families predicted by AF2-multimer
4. Transcription factor distillation - JASPAR-based TF-DNA predictions
5. Short protein distillation - Short sequences for protein scaffolding
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class DistillationSource(Enum):
    """Source of distillation data."""
    MGNIFY = auto()          # MGnify protein sequences
    RFAM = auto()            # Rfam RNA families
    JASPAR = auto()          # JASPAR transcription factor binding sites
    PDB_DISORDERED = auto()  # Disordered regions from PDB
    SHORT_PROTEIN = auto()   # Short protein sequences


@dataclass
class DistillationSample:
    """A sample from distillation dataset.
    
    Attributes:
        sample_id: Unique identifier
        source: Source of the sample
        sequences: Dictionary of chain_id -> sequence
        chain_types: Dictionary of chain_id -> type (protein/rna/dna)
        predicted_structure_path: Path to predicted structure
        confidence_scores: pLDDT or similar confidence
        metadata: Additional metadata
    """
    sample_id: str
    source: DistillationSource
    sequences: Dict[str, str]
    chain_types: Dict[str, str]
    predicted_structure_path: Optional[str] = None
    confidence_scores: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_chains(self) -> int:
        return len(self.sequences)
    
    @property
    def total_length(self) -> int:
        return sum(len(seq) for seq in self.sequences.values())
    
    def mean_confidence(self) -> Optional[float]:
        """Compute mean confidence score across all chains."""
        if self.confidence_scores is None:
            return None
        all_scores = []
        for scores in self.confidence_scores.values():
            all_scores.extend(scores.tolist())
        return np.mean(all_scores) if all_scores else None


@dataclass
class DistillationConfig:
    """Configuration for distillation dataset generation."""
    
    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("distillation"))
    
    # Prediction model
    predictor_type: str = "alphafold2"  # alphafold2, esmfold, etc.
    predictor_model_path: Optional[str] = None
    
    # Filtering
    min_confidence: float = 70.0  # Minimum pLDDT
    max_sequence_length: int = 2000
    min_sequence_length: int = 16
    
    # Sampling
    max_samples_per_source: int = 100000
    random_seed: int = 42
    
    # Dataset weights (from Table 4)
    disordered_weight: float = 0.02
    monomer_weight: float = 0.495
    short_protein_weight: float = 0.005
    rna_weight: float = 0.05


class BaseDistillationGenerator(ABC):
    """Base class for distillation data generators."""
    
    def __init__(self, config: Optional[DistillationConfig] = None):
        self.config = config or DistillationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
    
    @property
    @abstractmethod
    def source(self) -> DistillationSource:
        """Source type for this generator."""
        pass
    
    @abstractmethod
    async def generate_samples(
        self,
        input_path: Union[str, Path],
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DistillationSample]:
        """Generate distillation samples.
        
        Args:
            input_path: Path to input data
            max_samples: Maximum samples to generate
            progress_callback: Called with (current, total)
            
        Returns:
            List of distillation samples
        """
        pass
    
    def filter_sample(self, sample: DistillationSample) -> bool:
        """Check if sample passes filtering criteria."""
        # Check length
        if sample.total_length > self.config.max_sequence_length:
            return False
        if sample.total_length < self.config.min_sequence_length:
            return False
        
        # Check confidence
        if self.config.min_confidence > 0:
            mean_conf = sample.mean_confidence()
            if mean_conf is not None and mean_conf < self.config.min_confidence:
                return False
        
        return True


class MGnifyDistillationGenerator(BaseDistillationGenerator):
    """Generate protein monomer distillation from MGnify.
    
    From AF3 Section 2.5.3:
    - Use representative sequences from MGnify clusters
    - Predict structures using AlphaFold2
    - Filter by pLDDT confidence
    """
    
    @property
    def source(self) -> DistillationSource:
        return DistillationSource.MGNIFY
    
    async def generate_samples(
        self,
        input_path: Union[str, Path],
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DistillationSample]:
        """Generate samples from MGnify FASTA file.
        
        Args:
            input_path: Path to MGnify FASTA file
            max_samples: Maximum samples to generate
            progress_callback: Progress callback
            
        Returns:
            List of distillation samples
        """
        max_samples = max_samples or self.config.max_samples_per_source
        samples = []
        
        # Read sequences from FASTA
        sequences = self._read_fasta(Path(input_path))
        total = len(sequences)
        
        logger.info(f"Processing {total} MGnify sequences")
        
        # Shuffle for random sampling
        seq_ids = list(sequences.keys())
        self.rng.shuffle(seq_ids)
        
        for i, seq_id in enumerate(seq_ids):
            if len(samples) >= max_samples:
                break
            
            sequence = sequences[seq_id]
            
            # Create sample
            sample = DistillationSample(
                sample_id=f"mgnify_{seq_id}",
                source=self.source,
                sequences={"A": sequence},
                chain_types={"A": "protein"},
                metadata={
                    "source_id": seq_id,
                    "source_db": "mgnify",
                },
            )
            
            # Filter
            if self.filter_sample(sample):
                samples.append(sample)
            
            if progress_callback and i % 1000 == 0:
                progress_callback(i, total)
        
        logger.info(f"Generated {len(samples)} MGnify distillation samples")
        return samples
    
    def _read_fasta(self, fasta_path: Path) -> Dict[str, str]:
        """Read sequences from FASTA file."""
        sequences = {}
        current_id = None
        current_seq = []
        
        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    # Parse header
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
        
        if current_id:
            sequences[current_id] = "".join(current_seq)
        
        return sequences


class RfamDistillationGenerator(BaseDistillationGenerator):
    """Generate RNA distillation from Rfam families.
    
    From AF3 Section 2.5.5:
    - Use representative sequences from Rfam families
    - Generate complexes using Stockholm annotations
    - Predict structures using AF2-multimer
    """
    
    @property
    def source(self) -> DistillationSource:
        return DistillationSource.RFAM
    
    async def generate_samples(
        self,
        input_path: Union[str, Path],
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DistillationSample]:
        """Generate samples from Rfam Stockholm files.
        
        Args:
            input_path: Path to Rfam seed alignments directory
            max_samples: Maximum samples to generate
            progress_callback: Progress callback
            
        Returns:
            List of distillation samples
        """
        max_samples = max_samples or self.config.max_samples_per_source
        samples = []
        
        input_dir = Path(input_path)
        sto_files = list(input_dir.glob("**/*.sto")) + list(input_dir.glob("**/*.stockholm"))
        total = len(sto_files)
        
        logger.info(f"Processing {total} Rfam families")
        
        for i, sto_file in enumerate(sto_files):
            if len(samples) >= max_samples:
                break
            
            try:
                family_samples = self._process_stockholm(sto_file)
                for sample in family_samples:
                    if self.filter_sample(sample):
                        samples.append(sample)
                        if len(samples) >= max_samples:
                            break
            except Exception as e:
                logger.warning(f"Failed to process {sto_file}: {e}")
            
            if progress_callback:
                progress_callback(i, total)
        
        logger.info(f"Generated {len(samples)} Rfam distillation samples")
        return samples
    
    def _process_stockholm(self, sto_path: Path) -> List[DistillationSample]:
        """Process a Stockholm alignment file."""
        samples = []
        sequences = {}
        family_id = sto_path.stem
        
        with open(sto_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line == "//":
                    break
                
                parts = line.split()
                if len(parts) >= 2:
                    seq_id = parts[0]
                    sequence = parts[1].replace(".", "-").replace("~", "-")
                    sequences[seq_id] = sequence.replace("-", "")
        
        # Take representative sequences
        for seq_id, sequence in list(sequences.items())[:5]:
            sample = DistillationSample(
                sample_id=f"rfam_{family_id}_{seq_id}",
                source=self.source,
                sequences={"A": sequence},
                chain_types={"A": "rna"},
                metadata={
                    "rfam_family": family_id,
                    "source_id": seq_id,
                },
            )
            samples.append(sample)
        
        return samples


class JASPARDistillationGenerator(BaseDistillationGenerator):
    """Generate TF-DNA distillation from JASPAR.
    
    From AF3 Section 2.5.6:
    - Use transcription factor binding profiles from JASPAR
    - Generate TF-DNA complexes
    - Predict structures with AF3 or AF2-multimer
    """
    
    @property
    def source(self) -> DistillationSource:
        return DistillationSource.JASPAR
    
    async def generate_samples(
        self,
        input_path: Union[str, Path],
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DistillationSample]:
        """Generate samples from JASPAR database.
        
        Args:
            input_path: Path to JASPAR profiles (JSON or MEME format)
            max_samples: Maximum samples to generate
            progress_callback: Progress callback
            
        Returns:
            List of distillation samples
        """
        max_samples = max_samples or self.config.max_samples_per_source
        samples = []
        
        input_file = Path(input_path)
        
        if input_file.suffix == ".json":
            profiles = self._load_jaspar_json(input_file)
        else:
            profiles = self._load_jaspar_meme(input_file)
        
        total = len(profiles)
        logger.info(f"Processing {total} JASPAR profiles")
        
        for i, profile in enumerate(profiles):
            if len(samples) >= max_samples:
                break
            
            # Generate DNA sequence from motif
            dna_sequence = self._sample_dna_from_motif(profile["matrix"])
            
            sample = DistillationSample(
                sample_id=f"jaspar_{profile['id']}",
                source=self.source,
                sequences={
                    "A": profile.get("protein_sequence", ""),
                    "B": dna_sequence,
                },
                chain_types={
                    "A": "protein",
                    "B": "dna",
                },
                metadata={
                    "jaspar_id": profile["id"],
                    "tf_name": profile.get("name", ""),
                    "motif_length": len(dna_sequence),
                },
            )
            
            if self.filter_sample(sample):
                samples.append(sample)
            
            if progress_callback:
                progress_callback(i, total)
        
        logger.info(f"Generated {len(samples)} JASPAR distillation samples")
        return samples
    
    def _load_jaspar_json(self, json_path: Path) -> List[Dict]:
        """Load JASPAR profiles from JSON."""
        with open(json_path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    
    def _load_jaspar_meme(self, meme_path: Path) -> List[Dict]:
        """Load JASPAR profiles from MEME format."""
        profiles = []
        current_profile = None
        
        with open(meme_path) as f:
            for line in f:
                if line.startswith("MOTIF"):
                    if current_profile:
                        profiles.append(current_profile)
                    parts = line.strip().split()
                    current_profile = {
                        "id": parts[1] if len(parts) > 1 else "",
                        "name": parts[2] if len(parts) > 2 else "",
                        "matrix": [],
                    }
                elif line.startswith("letter-probability"):
                    continue
                elif current_profile and line.strip() and not line.startswith("URL"):
                    try:
                        probs = [float(x) for x in line.strip().split()]
                        if len(probs) == 4:
                            current_profile["matrix"].append(probs)
                    except ValueError:
                        pass
        
        if current_profile:
            profiles.append(current_profile)
        
        return profiles
    
    def _sample_dna_from_motif(self, matrix: List[List[float]]) -> str:
        """Sample DNA sequence from position weight matrix."""
        bases = ["A", "C", "G", "T"]
        sequence = []
        
        for position in matrix:
            # Normalize probabilities
            probs = np.array(position)
            probs = probs / probs.sum()
            
            # Sample base
            base_idx = self.rng.choice(4, p=probs)
            sequence.append(bases[base_idx])
        
        return "".join(sequence)


class PDBDisorderedDistillationGenerator(BaseDistillationGenerator):
    """Generate distillation from disordered PDB regions.
    
    From AF3 Section 2.5.2:
    - Identify disordered regions in PDB structures
    - Use AF2 predictions for comparison
    - Generate training data from low-confidence predictions
    """
    
    @property
    def source(self) -> DistillationSource:
        return DistillationSource.PDB_DISORDERED
    
    async def generate_samples(
        self,
        input_path: Union[str, Path],
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DistillationSample]:
        """Generate samples from PDB with disordered regions.
        
        Args:
            input_path: Path to disorder annotation file
            max_samples: Maximum samples to generate
            progress_callback: Progress callback
            
        Returns:
            List of distillation samples
        """
        max_samples = max_samples or self.config.max_samples_per_source
        samples = []
        
        # Load disorder annotations
        annotations = self._load_disorder_annotations(Path(input_path))
        total = len(annotations)
        
        logger.info(f"Processing {total} PDB entries with disorder annotations")
        
        for i, annotation in enumerate(annotations):
            if len(samples) >= max_samples:
                break
            
            # Create sample with disorder information
            sample = DistillationSample(
                sample_id=f"pdb_disorder_{annotation['pdb_id']}_{annotation['chain_id']}",
                source=self.source,
                sequences={"A": annotation["sequence"]},
                chain_types={"A": "protein"},
                metadata={
                    "pdb_id": annotation["pdb_id"],
                    "chain_id": annotation["chain_id"],
                    "disordered_regions": annotation["disordered_regions"],
                    "disorder_fraction": annotation["disorder_fraction"],
                },
            )
            
            if self.filter_sample(sample):
                samples.append(sample)
            
            if progress_callback:
                progress_callback(i, total)
        
        logger.info(f"Generated {len(samples)} disordered PDB distillation samples")
        return samples
    
    def _load_disorder_annotations(self, annotation_path: Path) -> List[Dict]:
        """Load disorder annotations from file.
        
        Expected format: TSV with columns:
        pdb_id, chain_id, sequence, disordered_regions, disorder_fraction
        """
        annotations = []
        
        with open(annotation_path) as f:
            header = f.readline().strip().split("\t")
            
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    annotations.append({
                        "pdb_id": parts[0],
                        "chain_id": parts[1],
                        "sequence": parts[2],
                        "disordered_regions": parts[3],
                        "disorder_fraction": float(parts[4]),
                    })
        
        return annotations


class ShortProteinDistillationGenerator(BaseDistillationGenerator):
    """Generate short protein distillation.
    
    From AF3 Section 2.5.4:
    - Short protein sequences (<50 residues)
    - Used for scaffold prediction
    """
    
    @property
    def source(self) -> DistillationSource:
        return DistillationSource.SHORT_PROTEIN
    
    async def generate_samples(
        self,
        input_path: Union[str, Path],
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DistillationSample]:
        """Generate samples from short protein sequences.
        
        Args:
            input_path: Path to short protein FASTA
            max_samples: Maximum samples to generate
            progress_callback: Progress callback
            
        Returns:
            List of distillation samples
        """
        max_samples = max_samples or self.config.max_samples_per_source
        samples = []
        
        # Override length filter for short proteins
        original_min = self.config.min_sequence_length
        self.config.min_sequence_length = 10
        
        sequences = self._read_fasta(Path(input_path))
        
        # Filter to short sequences
        short_sequences = {
            k: v for k, v in sequences.items()
            if len(v) <= 50 and len(v) >= 10
        }
        
        total = len(short_sequences)
        logger.info(f"Processing {total} short protein sequences")
        
        seq_ids = list(short_sequences.keys())
        self.rng.shuffle(seq_ids)
        
        for i, seq_id in enumerate(seq_ids):
            if len(samples) >= max_samples:
                break
            
            sequence = short_sequences[seq_id]
            
            sample = DistillationSample(
                sample_id=f"short_{seq_id}",
                source=self.source,
                sequences={"A": sequence},
                chain_types={"A": "protein"},
                metadata={
                    "source_id": seq_id,
                    "length": len(sequence),
                },
            )
            
            if self.filter_sample(sample):
                samples.append(sample)
            
            if progress_callback and i % 1000 == 0:
                progress_callback(i, total)
        
        self.config.min_sequence_length = original_min
        
        logger.info(f"Generated {len(samples)} short protein distillation samples")
        return samples
    
    def _read_fasta(self, fasta_path: Path) -> Dict[str, str]:
        """Read sequences from FASTA file."""
        sequences = {}
        current_id = None
        current_seq = []
        
        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
        
        if current_id:
            sequences[current_id] = "".join(current_seq)
        
        return sequences


class DistillationPipeline:
    """Complete distillation data generation pipeline.
    
    Combines all distillation sources with proper weighting.
    
    Example:
        ```python
        config = DistillationConfig(
            output_dir=Path("distillation_data"),
            min_confidence=70.0,
        )
        
        pipeline = DistillationPipeline(config)
        
        # Generate all distillation datasets
        all_samples = await pipeline.generate_all(
            mgnify_fasta="mgnify_clusters.fasta",
            rfam_dir="rfam_seed/",
            jaspar_json="jaspar_profiles.json",
            disorder_annotations="pdb_disorder.tsv",
        )
        
        # Sample with proper weights
        batch = pipeline.sample(all_samples, batch_size=32)
        ```
    """
    
    def __init__(self, config: Optional[DistillationConfig] = None):
        self.config = config or DistillationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        
        # Initialize generators
        self.generators = {
            DistillationSource.MGNIFY: MGnifyDistillationGenerator(self.config),
            DistillationSource.RFAM: RfamDistillationGenerator(self.config),
            DistillationSource.JASPAR: JASPARDistillationGenerator(self.config),
            DistillationSource.PDB_DISORDERED: PDBDisorderedDistillationGenerator(self.config),
            DistillationSource.SHORT_PROTEIN: ShortProteinDistillationGenerator(self.config),
        }
        
        # Dataset storage
        self.datasets: Dict[DistillationSource, List[DistillationSample]] = {
            source: [] for source in DistillationSource
        }
    
    async def generate_all(
        self,
        mgnify_fasta: Optional[Union[str, Path]] = None,
        rfam_dir: Optional[Union[str, Path]] = None,
        jaspar_json: Optional[Union[str, Path]] = None,
        disorder_annotations: Optional[Union[str, Path]] = None,
        short_protein_fasta: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[DistillationSource, List[DistillationSample]]:
        """Generate all distillation datasets.
        
        Args:
            mgnify_fasta: Path to MGnify sequences
            rfam_dir: Path to Rfam seed alignments
            jaspar_json: Path to JASPAR profiles
            disorder_annotations: Path to disorder annotations
            short_protein_fasta: Path to short protein sequences
            progress_callback: Called with (source_name, current, total)
            
        Returns:
            Dictionary of source -> samples
        """
        if mgnify_fasta:
            logger.info("Generating MGnify distillation...")
            samples = await self.generators[DistillationSource.MGNIFY].generate_samples(
                mgnify_fasta,
                progress_callback=lambda c, t: progress_callback("mgnify", c, t) if progress_callback else None,
            )
            self.datasets[DistillationSource.MGNIFY] = samples
        
        if rfam_dir:
            logger.info("Generating Rfam distillation...")
            samples = await self.generators[DistillationSource.RFAM].generate_samples(
                rfam_dir,
                progress_callback=lambda c, t: progress_callback("rfam", c, t) if progress_callback else None,
            )
            self.datasets[DistillationSource.RFAM] = samples
        
        if jaspar_json:
            logger.info("Generating JASPAR distillation...")
            samples = await self.generators[DistillationSource.JASPAR].generate_samples(
                jaspar_json,
                progress_callback=lambda c, t: progress_callback("jaspar", c, t) if progress_callback else None,
            )
            self.datasets[DistillationSource.JASPAR] = samples
        
        if disorder_annotations:
            logger.info("Generating disordered PDB distillation...")
            samples = await self.generators[DistillationSource.PDB_DISORDERED].generate_samples(
                disorder_annotations,
                progress_callback=lambda c, t: progress_callback("disordered", c, t) if progress_callback else None,
            )
            self.datasets[DistillationSource.PDB_DISORDERED] = samples
        
        if short_protein_fasta:
            logger.info("Generating short protein distillation...")
            samples = await self.generators[DistillationSource.SHORT_PROTEIN].generate_samples(
                short_protein_fasta,
                progress_callback=lambda c, t: progress_callback("short_protein", c, t) if progress_callback else None,
            )
            self.datasets[DistillationSource.SHORT_PROTEIN] = samples
        
        return self.datasets
    
    def get_weights(self) -> Dict[DistillationSource, float]:
        """Get dataset weights from config."""
        return {
            DistillationSource.MGNIFY: self.config.monomer_weight,
            DistillationSource.RFAM: self.config.rna_weight,
            DistillationSource.JASPAR: 0.0,  # Included in monomer_weight
            DistillationSource.PDB_DISORDERED: self.config.disordered_weight,
            DistillationSource.SHORT_PROTEIN: self.config.short_protein_weight,
        }
    
    def sample(
        self,
        datasets: Optional[Dict[DistillationSource, List[DistillationSample]]] = None,
        batch_size: int = 32,
    ) -> List[DistillationSample]:
        """Sample from distillation datasets with proper weighting.
        
        Args:
            datasets: Dataset dictionary (uses self.datasets if None)
            batch_size: Number of samples
            
        Returns:
            List of sampled distillation samples
        """
        datasets = datasets or self.datasets
        weights = self.get_weights()
        
        # Filter to non-empty datasets
        valid_sources = [
            source for source, samples in datasets.items()
            if samples and weights.get(source, 0) > 0
        ]
        
        if not valid_sources:
            return []
        
        # Normalize weights
        source_weights = np.array([weights[s] for s in valid_sources])
        source_weights /= source_weights.sum()
        
        # Sample
        samples = []
        for _ in range(batch_size):
            # Select source
            source_idx = self.rng.choice(len(valid_sources), p=source_weights)
            source = valid_sources[source_idx]
            
            # Sample from source
            source_samples = datasets[source]
            sample_idx = self.rng.integers(0, len(source_samples))
            samples.append(source_samples[sample_idx])
        
        return samples
    
    def save_datasets(
        self,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Save all datasets to disk.
        
        Args:
            output_dir: Output directory (uses config default if None)
        """
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for source, samples in self.datasets.items():
            if not samples:
                continue
            
            # Save metadata
            metadata_file = output_dir / f"{source.name.lower()}_metadata.json"
            with open(metadata_file, "w") as f:
                metadata = [
                    {
                        "sample_id": s.sample_id,
                        "sequences": s.sequences,
                        "chain_types": s.chain_types,
                        "metadata": s.metadata,
                    }
                    for s in samples
                ]
                json.dump(metadata, f, indent=2)
            
            # Save sequences as FASTA
            fasta_file = output_dir / f"{source.name.lower()}_sequences.fasta"
            with open(fasta_file, "w") as f:
                for sample in samples:
                    for chain_id, sequence in sample.sequences.items():
                        f.write(f">{sample.sample_id}_{chain_id}\n")
                        for i in range(0, len(sequence), 80):
                            f.write(sequence[i:i+80] + "\n")
            
            logger.info(f"Saved {len(samples)} {source.name} samples to {output_dir}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all datasets."""
        stats = {}
        
        for source, samples in self.datasets.items():
            if not samples:
                continue
            
            lengths = [s.total_length for s in samples]
            confidences = [s.mean_confidence() for s in samples if s.mean_confidence() is not None]
            
            stats[source.name] = {
                "count": len(samples),
                "total_sequences": sum(s.num_chains for s in samples),
                "mean_length": np.mean(lengths) if lengths else 0,
                "median_length": np.median(lengths) if lengths else 0,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "mean_confidence": np.mean(confidences) if confidences else None,
            }
        
        return stats
