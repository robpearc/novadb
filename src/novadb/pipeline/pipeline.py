"""Data processing pipeline orchestration.

Implements the full data processing pipeline from AlphaFold3:
1. Parse structures (mmCIF)
2. Filter structures (date, resolution, composition)
3. Genetic search (MSA generation)
4. Template search
5. Tokenize structures
6. Extract features
7. Crop to training size
8. Store processed data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import traceback

from novadb.config import Config
from novadb.data.parsers.mmcif_parser import MMCIFParser
from novadb.data.parsers.structure import Structure
from novadb.processing.tokenization.tokenizer import Tokenizer, TokenizedStructure
from novadb.processing.features.features import FeatureExtractor, InputFeatures
from novadb.processing.cropping import Cropper, CropConfig
from novadb.processing.curation.filtering import StructureFilter, FilterResult
from novadb.processing.curation.sampling import DatasetEntry, DatasetSampler
from novadb.search.msa.msa import MSA
from novadb.search.msa.msa_processor import MSAProcessor
from novadb.search.msa.jackhmmer import JackhmmerSearch
from novadb.search.msa.hhblits import HHBlitsSearch
from novadb.search.templates.template_search import TemplateSearcher, TemplateSearchResult
from novadb.storage.backends import StorageBackend, create_storage
from novadb.storage.serialization import DataSerializer, DatasetWriter


logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""
    total_structures: int = 0
    parsed_successfully: int = 0
    passed_filters: int = 0
    processed_successfully: int = 0
    failed: int = 0
    skipped: int = 0

    filter_failures: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class ProcessedSample:
    """A fully processed training sample."""
    pdb_id: str
    features: InputFeatures
    tokenized: TokenizedStructure
    msa: Optional[MSA] = None
    templates: Optional[TemplateSearchResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataPipeline:
    """Full data processing pipeline.
    
    Orchestrates the entire data processing workflow from raw
    mmCIF files to processed features ready for training.
    """

    def __init__(
        self,
        config: Config,
        storage: Optional[StorageBackend] = None,
    ):
        """Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
            storage: Optional storage backend (created from config if not provided)
        """
        self.config = config
        self.storage = storage or create_storage(config.storage)

        # Initialize components
        self.parser = MMCIFParser()
        self.structure_filter = StructureFilter(
            filtering_config=config.filtering,
            date_config=config.dates,
        )
        self.tokenizer = Tokenizer()
        self.feature_extractor = FeatureExtractor(
            max_msa_sequences=config.msa_processing.max_msa_rows,
            max_templates=config.template_search.max_templates,
        )
        self.cropper = Cropper(CropConfig(
            contiguous_weight=config.cropping.pdb_contiguous_weight,
            spatial_weight=config.cropping.pdb_spatial_weight,
            spatial_interface_weight=config.cropping.pdb_spatial_interface_weight,
        ))

        # MSA search tools (optional - require external databases)
        self.jackhmmer = None
        self.hhblits = None
        self.msa_processor = MSAProcessor(config=config.msa_processing)

        # Template search (optional)
        self.template_searcher = None

        # Serializer
        self.serializer = DataSerializer(self.storage)

        # Stats
        self.stats = PipelineStats()

    def setup_msa_search(
        self,
        jackhmmer_binary: str = "jackhmmer",
        hhblits_binary: str = "hhblits",
    ) -> None:
        """Set up MSA search tools.
        
        Args:
            jackhmmer_binary: Path to jackhmmer binary
            hhblits_binary: Path to hhblits binary
        """
        self.jackhmmer = JackhmmerSearch(
            binary_path=jackhmmer_binary,
            config=self.config.jackhmmer,
        )
        self.hhblits = HHBlitsSearch(
            binary_path=hhblits_binary,
            config=self.config.hhblits,
        )

    def setup_template_search(
        self,
        hmmbuild_binary: str = "hmmbuild",
        hmmsearch_binary: str = "hmmsearch",
        pdb_database: str = "",
        pdb_mmcif_dir: str = "",
    ) -> None:
        """Set up template search.
        
        Args:
            hmmbuild_binary: Path to hmmbuild binary
            hmmsearch_binary: Path to hmmsearch binary
            pdb_database: Path to PDB sequence database
            pdb_mmcif_dir: Path to mmCIF files directory
        """
        self.template_searcher = TemplateSearcher(
            hmmbuild_binary=hmmbuild_binary,
            hmmsearch_binary=hmmsearch_binary,
            pdb_database=pdb_database,
            pdb_mmcif_dir=pdb_mmcif_dir,
            max_templates=self.config.templates.max_templates,
        )

    def process_structure(
        self,
        mmcif_path: str,
        run_msa: bool = True,
        run_templates: bool = True,
    ) -> Optional[ProcessedSample]:
        """Process a single structure.
        
        Args:
            mmcif_path: Path to mmCIF file
            run_msa: Whether to run MSA search
            run_templates: Whether to run template search
            
        Returns:
            ProcessedSample if successful, None otherwise
        """
        self.stats.total_structures += 1

        try:
            # Parse structure
            structure = self.parser.parse(mmcif_path)
            self.stats.parsed_successfully += 1

            # Apply filters
            filter_result = self.structure_filter.filter(structure)
            if not filter_result.passed:
                reason = filter_result.reason.split(":")[0]
                self.stats.filter_failures[reason] = (
                    self.stats.filter_failures.get(reason, 0) + 1
                )
                return None

            self.stats.passed_filters += 1

            # Filter ligands
            structure = self.structure_filter.filter_ligands(structure)

            # Tokenize
            tokenized = self.tokenizer.tokenize(structure)

            # Run MSA search if enabled
            msa = None
            if run_msa and self.jackhmmer is not None:
                msa = self._run_msa_search(structure)

            # Run template search if enabled
            templates = None
            if run_templates and self.template_searcher is not None and msa is not None:
                templates = self._run_template_search(structure, msa)

            # Crop if needed
            if len(tokenized.tokens) > self.config.cropping.max_tokens:
                tokenized = self.cropper.crop_to_token_limit(
                    tokenized,
                    self.config.cropping.max_tokens,
                )

            # Extract features
            features = self.feature_extractor.extract(
                tokenized,
                msa=msa,
                templates=None,  # Would need template structures
            )

            self.stats.processed_successfully += 1

            return ProcessedSample(
                pdb_id=structure.pdb_id,
                features=features,
                tokenized=tokenized,
                msa=msa,
                templates=templates,
                metadata={
                    "resolution": structure.resolution,
                    "release_date": str(structure.release_date),
                    "method": structure.method,
                    "num_chains": len(structure.chains),
                    "num_tokens": len(tokenized.tokens),
                },
            )

        except Exception as e:
            self.stats.failed += 1
            self.stats.errors.append(f"{mmcif_path}: {str(e)}")
            logger.error(f"Failed to process {mmcif_path}: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _run_msa_search(self, structure: Structure) -> Optional[MSA]:
        """Run MSA search for protein chains."""
        # Get first protein chain sequence
        for chain in structure.chains.values():
            if chain.chain_type and chain.chain_type.name == "PROTEIN":
                sequence = chain.get_sequence()
                if len(sequence) >= 10:  # Minimum length
                    try:
                        # Run Jackhmmer on UniRef90
                        msa = self.jackhmmer.search(
                            sequence,
                            self.config.databases.uniref90,
                        )
                        return msa
                    except Exception as e:
                        logger.warning(f"MSA search failed: {e}")
                        return None
        return None

    def _run_template_search(
        self,
        structure: Structure,
        msa: MSA,
    ) -> Optional[TemplateSearchResult]:
        """Run template search."""
        try:
            max_date = self.config.dates.training_cutoff
            return self.template_searcher.search(
                query_sequence=msa.query_sequence,
                msa_a3m=msa.to_a3m(),
                max_template_date=max_date,
            )
        except Exception as e:
            logger.warning(f"Template search failed: {e}")
            return None

    def process_directory(
        self,
        mmcif_dir: str,
        output_prefix: str = "processed",
        max_structures: Optional[int] = None,
    ) -> PipelineStats:
        """Process all structures in a directory.
        
        Args:
            mmcif_dir: Directory containing mmCIF files
            output_prefix: Prefix for output files
            max_structures: Maximum number of structures to process
            
        Returns:
            Pipeline statistics
        """
        mmcif_path = Path(mmcif_dir)
        cif_files = list(mmcif_path.glob("*.cif")) + list(mmcif_path.glob("*.cif.gz"))

        if max_structures:
            cif_files = cif_files[:max_structures]

        logger.info(f"Processing {len(cif_files)} structures from {mmcif_dir}")

        writer = DatasetWriter(self.storage, shard_size=100)

        for i, cif_file in enumerate(cif_files):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(cif_files)}")

            sample = self.process_structure(str(cif_file))
            if sample is not None:
                writer.add(sample.features)

        metadata = writer.finalize()
        logger.info(f"Finished processing. Stats: {self.stats}")

        return self.stats

    def process_pdb_list(
        self,
        pdb_ids: List[str],
        mmcif_dir: str,
        output_prefix: str = "processed",
    ) -> Generator[ProcessedSample, None, None]:
        """Process a list of PDB IDs.
        
        Args:
            pdb_ids: List of PDB IDs to process
            mmcif_dir: Directory containing mmCIF files
            output_prefix: Prefix for output files
            
        Yields:
            ProcessedSample for each successfully processed structure
        """
        mmcif_path = Path(mmcif_dir)

        for pdb_id in pdb_ids:
            # Try different file patterns
            cif_file = None
            for pattern in [f"{pdb_id}.cif", f"{pdb_id.lower()}.cif", 
                           f"{pdb_id}.cif.gz", f"{pdb_id.lower()}.cif.gz"]:
                candidate = mmcif_path / pattern
                if candidate.exists():
                    cif_file = candidate
                    break

            if cif_file is None:
                logger.warning(f"Structure not found: {pdb_id}")
                self.stats.skipped += 1
                continue

            sample = self.process_structure(str(cif_file))
            if sample is not None:
                yield sample


class DistillationPipeline:
    """Pipeline for generating distillation data.
    
    From AF3 Section 2.5: Generate predictions for sequences
    without experimental structures.
    """

    def __init__(
        self,
        config: Config,
        prediction_model: str = "alphafold2",
    ):
        self.config = config
        self.prediction_model = prediction_model

        # Components
        self.tokenizer = Tokenizer()
        self.feature_extractor = FeatureExtractor()

    def generate_predictions(
        self,
        sequences: List[Tuple[str, str]],  # (id, sequence)
        output_dir: str,
    ) -> List[str]:
        """Generate structure predictions for sequences.
        
        This is a placeholder - actual implementation would
        call AlphaFold2 or similar.
        
        Args:
            sequences: List of (id, sequence) tuples
            output_dir: Directory for output structures
            
        Returns:
            List of output file paths
        """
        output_paths = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for seq_id, sequence in sequences:
            # Placeholder - would call AF2 here
            out_file = output_path / f"{seq_id}_pred.cif"
            # Would write predicted structure here
            output_paths.append(str(out_file))

        return output_paths


def create_pipeline(config_path: Optional[str] = None) -> DataPipeline:
    """Create a pipeline from configuration.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configured DataPipeline
    """
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    return DataPipeline(config)
