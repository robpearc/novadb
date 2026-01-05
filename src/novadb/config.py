"""Configuration management for NovaDB pipeline.

This module defines all configuration options for the data processing pipeline,
including database paths, processing parameters, and storage settings.
Based on AlphaFold3 supplementary materials Section 2.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class StorageBackend(str, Enum):
    """Supported storage backends for processed data."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class DatabasePaths(BaseModel):
    """Paths to genetic and template search databases.

    Based on Tables 1 and 2 in the AF3 supplement describing:
    - Protein sequence databases (UniRef90, UniProt, BFD, MGnify)
    - RNA sequence databases (Rfam, RNACentral, Nucleotide collection)
    - Template database (PDB sequences)
    """

    # PDB structure database
    pdb_mmcif_dir: Path = Field(
        description="Directory containing PDB mmCIF files"
    )
    pdb_seqres_fasta: Optional[Path] = Field(
        default=None,
        description="FASTA file of all PDB sequences for template search"
    )

    # Protein sequence databases
    uniref90: Optional[Path] = Field(
        default=None,
        description="UniRef90 FASTA database (jackhmmer search)"
    )
    uniprot: Optional[Path] = Field(
        default=None,
        description="UniProt FASTA database (jackhmmer search)"
    )
    uniclust30: Optional[Path] = Field(
        default=None,
        description="Uniclust30 database for HHBlits"
    )
    bfd: Optional[Path] = Field(
        default=None,
        description="BFD database (combined with Uniclust30 for HHBlits)"
    )
    reduced_bfd: Optional[Path] = Field(
        default=None,
        description="Reduced BFD FASTA database (jackhmmer search)"
    )
    mgnify: Optional[Path] = Field(
        default=None,
        description="MGnify FASTA database (jackhmmer search)"
    )

    # RNA sequence databases
    rfam: Optional[Path] = Field(
        default=None,
        description="Rfam covariance model database (nhmmer search)"
    )
    rnacentral: Optional[Path] = Field(
        default=None,
        description="RNACentral FASTA database (nhmmer search)"
    )
    nt: Optional[Path] = Field(
        default=None,
        description="NCBI Nucleotide collection FASTA database (nhmmer search)"
    )


class JackhmmerConfig(BaseModel):
    """Configuration for Jackhmmer searches.

    Default flags from AF3 supplement Section 2.2:
    -N 1 -E 0.0001 --incE 0.0001 --F1 0.0005 --F2 0.00005 --F3 0.0000005
    """

    num_iterations: int = Field(default=1, description="Number of iterations (-N)")
    e_value: float = Field(default=0.0001, description="E-value threshold (-E)")
    inc_e_value: float = Field(default=0.0001, description="Inclusion E-value (--incE)")
    f1: float = Field(default=0.0005, description="Stage 1 filter threshold (--F1)")
    f2: float = Field(default=0.00005, description="Stage 2 filter threshold (--F2)")
    f3: float = Field(default=0.0000005, description="Stage 3 filter threshold (--F3)")

    # Per-database sequence limits (Table 1)
    uniref90_seq_limit: int = Field(default=100000, description="UniRef90 --seq_limit")
    uniprot_seq_limit: int = Field(default=500000, description="UniProt --seq_limit")
    reduced_bfd_seq_limit: int = Field(default=50000, description="Reduced BFD --seq_limit")
    mgnify_seq_limit: int = Field(default=50000, description="MGnify --seq_limit")

    # Max sequences to keep per database (Table 1)
    uniref90_max_seqs: int = Field(default=10000, description="Max UniRef90 sequences")
    uniprot_max_seqs: int = Field(default=50000, description="Max UniProt sequences")
    reduced_bfd_max_seqs: int = Field(default=5000, description="Max Reduced BFD sequences")
    mgnify_max_seqs: int = Field(default=5000, description="Max MGnify sequences")


class HHBlitsConfig(BaseModel):
    """Configuration for HHBlits searches.

    Default flags from AF3 supplement Section 2.2:
    -n 3 -e 0.001 -realign_max 100000 -maxfilt 100000 -min_prefilter_hits 1000 -p 20 -Z 500
    """

    num_iterations: int = Field(default=3, description="Number of iterations (-n)")
    e_value: float = Field(default=0.001, description="E-value threshold (-e)")
    realign_max: int = Field(default=100000, description="Max realignments (--realign_max)")
    maxfilt: int = Field(default=100000, description="Max filter hits (--maxfilt)")
    min_prefilter_hits: int = Field(default=1000, description="Min prefilter hits")
    min_prob: int = Field(default=20, description="Minimum probability (-p)")
    max_seqs: int = Field(default=500, description="Max output sequences (-Z)")


class NhmmerConfig(BaseModel):
    """Configuration for nhmmer searches for RNA.

    Default flags from AF3 supplement Section 2.2:
    -E 0.001 --incE 0.001 --rna --watson --F3 0.00005
    For short sequences (<50 nt): --F3 0.02
    """

    e_value: float = Field(default=0.001, description="E-value threshold (-E)")
    inc_e_value: float = Field(default=0.001, description="Inclusion E-value (--incE)")
    f3: float = Field(default=0.00005, description="Stage 3 filter threshold (--F3)")
    f3_short: float = Field(default=0.02, description="F3 for sequences <50 nt")
    short_seq_threshold: int = Field(default=50, description="Threshold for short sequences")

    # Per-database max sequences (Table 2)
    rfam_max_seqs: int = Field(default=10000, description="Max Rfam sequences")
    rnacentral_max_seqs: int = Field(default=10000, description="Max RNACentral sequences")
    nt_max_seqs: int = Field(default=10000, description="Max NT sequences")


class TemplateSearchConfig(BaseModel):
    """Configuration for template search.

    From AF3 supplement Section 2.4:
    - Uses hmmbuild + hmmsearch with specific flags
    - Templates filtered by release date and sequence identity
    """

    max_msa_for_hmm: int = Field(
        default=300,
        description="Max MSA sequences for HMM building during training"
    )
    hmmsearch_flags: str = Field(
        default="--noali --F1 0.1 --F2 0.1 --F3 0.1 --E 100 --incE 100 --domE 100 --incdomE 100",
        description="Hmmsearch command flags"
    )
    max_templates: int = Field(default=20, description="Max templates from search")
    templates_to_use: int = Field(default=4, description="Templates used by model")
    min_template_length: int = Field(default=10, description="Min template residues")
    min_query_coverage: float = Field(default=0.10, description="Min template coverage")
    max_sequence_identity: float = Field(default=0.95, description="Max identity for filtering")
    template_release_offset_days: int = Field(
        default=60,
        description="Template release must be >= this many days before example"
    )


class MSAProcessingConfig(BaseModel):
    """Configuration for MSA processing.

    From AF3 supplement Section 2.3:
    - Max 16,384 MSA rows
    - First row is query
    - Next 8,191 rows from UniProt pairing by species
    - Remaining rows from dense stacking
    """

    max_msa_rows: int = Field(default=16384, description="Maximum MSA rows")
    max_paired_rows: int = Field(default=8191, description="Max UniProt paired rows")


class FilteringConfig(BaseModel):
    """Configuration for structure filtering.

    From AF3 supplement Section 2.5.4 - PDB filtering constraints.
    """

    # Target-level filters
    max_resolution: float = Field(default=9.0, description="Max resolution in Angstroms")
    min_resolved_residues: int = Field(default=4, description="Min resolved residues per chain")
    max_chains_training: int = Field(default=300, description="Max chains for training")
    max_chains_evaluation: int = Field(default=1000, description="Max chains for evaluation")

    # Bioassembly-level filters
    remove_hydrogens: bool = Field(default=True, description="Remove hydrogen atoms")
    remove_unknown_polymers: bool = Field(default=True, description="Remove all-unknown chains")
    clash_threshold: float = Field(default=0.30, description="Fraction of atoms for clash")
    clash_distance: float = Field(default=1.7, description="Distance for clash detection (Å)")

    # Chain-level filters
    max_ca_distance: float = Field(default=10.0, description="Max Cα-Cα distance for proteins")
    max_chains_for_crop: int = Field(default=20, description="Max chains after cropping")
    crop_interface_distance: float = Field(default=15.0, description="Interface distance (Å)")


class ClusteringConfig(BaseModel):
    """Configuration for training set clustering.

    From AF3 supplement Section 2.5.3:
    - 40% sequence identity for proteins
    - 100% identity for nucleic acids
    - 100% identity for peptides (<10 residues)
    - CCD identity for ligands
    """

    protein_identity_threshold: float = Field(
        default=0.40, description="Sequence identity threshold for proteins"
    )
    nucleic_acid_identity_threshold: float = Field(
        default=1.0, description="Sequence identity threshold for nucleic acids"
    )
    peptide_identity_threshold: float = Field(
        default=1.0, description="Sequence identity threshold for peptides"
    )
    peptide_length_threshold: int = Field(
        default=10, description="Max residues to be considered a peptide"
    )


class CroppingConfig(BaseModel):
    """Configuration for structure cropping.

    From AF3 supplement Section 2.7 and Table 4:
    - Contiguous, spatial, and spatial interface cropping strategies
    - Dataset-specific weights for strategy selection
    """

    # Cropping weights for Weighted PDB / Disordered PDB
    pdb_contiguous_weight: float = Field(default=0.20)
    pdb_spatial_weight: float = Field(default=0.40)
    pdb_spatial_interface_weight: float = Field(default=0.40)

    # Cropping weights for distillation sets
    distill_contiguous_weight: float = Field(default=0.25)
    distill_spatial_weight: float = Field(default=0.75)

    # Interface detection
    interface_distance_threshold: float = Field(
        default=15.0, description="Distance for interface token detection (Å)"
    )


class SamplingWeightsConfig(BaseModel):
    """Configuration for training sample weighting.

    From AF3 supplement Equation 1 and Table 3.
    """

    # Dataset sampling weights (Table 3)
    weighted_pdb: float = Field(default=0.50)
    disordered_pdb_distillation: float = Field(default=0.02)
    protein_monomer_distillation: float = Field(default=0.495)
    short_protein_distillation: float = Field(default=0.005)
    rna_distillation: float = Field(default=0.05)
    tf_negatives: float = Field(default=0.011)
    tf_positives: float = Field(default=0.021)

    # Chain/interface type weights (Equation 1)
    beta_chain: float = Field(default=0.5)
    beta_interface: float = Field(default=1.0)
    alpha_protein: float = Field(default=3.0)
    alpha_nucleic: float = Field(default=3.0)
    alpha_ligand: float = Field(default=1.0)


class TokenizationConfig(BaseModel):
    """Configuration for tokenization.

    From AF3 supplement Section 2.6:
    - Standard residues: 1 token per residue
    - Modified residues/ligands: 1 token per heavy atom
    """

    max_atoms_per_token: int = Field(
        default=23, description="Max atoms per token (largest standard residue)"
    )


class StorageConfig(BaseModel):
    """Configuration for data storage backends."""

    backend: StorageBackend = Field(default=StorageBackend.LOCAL)

    # Local storage
    local_path: Optional[Path] = Field(
        default=None, description="Local storage directory"
    )

    # AWS S3
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    s3_region: Optional[str] = Field(default=None, description="AWS region")
    s3_prefix: Optional[str] = Field(default="", description="S3 key prefix")

    # Google Cloud Storage
    gcs_bucket: Optional[str] = Field(default=None, description="GCS bucket name")
    gcs_project: Optional[str] = Field(default=None, description="GCP project ID")
    gcs_prefix: Optional[str] = Field(default="", description="GCS prefix")

    # Azure Blob Storage
    azure_container: Optional[str] = Field(default=None, description="Azure container")
    azure_connection_string: Optional[str] = Field(
        default=None, description="Azure connection string"
    )
    azure_prefix: Optional[str] = Field(default="", description="Azure prefix")


class DateConfig(BaseModel):
    """Date cutoffs for training and evaluation.

    From AF3 supplement:
    - Training cutoff: 2021-09-30
    - Template cutoff for distillation: 2018-04-30
    - Evaluation set: 2022-05-01 to 2023-01-12
    """

    training_cutoff: date = Field(
        default=date(2021, 9, 30),
        description="Max release date for training structures"
    )
    template_cutoff_distillation: date = Field(
        default=date(2018, 4, 30),
        description="Max template date for distillation sets"
    )
    evaluation_start: date = Field(
        default=date(2022, 5, 1),
        description="Start date for evaluation set"
    )
    evaluation_end: date = Field(
        default=date(2023, 1, 12),
        description="End date for evaluation set"
    )


class Config(BaseSettings):
    """Main configuration for the NovaDB pipeline."""

    # Sub-configurations
    databases: DatabasePaths
    jackhmmer: JackhmmerConfig = Field(default_factory=JackhmmerConfig)
    hhblits: HHBlitsConfig = Field(default_factory=HHBlitsConfig)
    nhmmer: NhmmerConfig = Field(default_factory=NhmmerConfig)
    template_search: TemplateSearchConfig = Field(default_factory=TemplateSearchConfig)
    msa_processing: MSAProcessingConfig = Field(default_factory=MSAProcessingConfig)
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    cropping: CroppingConfig = Field(default_factory=CroppingConfig)
    sampling: SamplingWeightsConfig = Field(default_factory=SamplingWeightsConfig)
    tokenization: TokenizationConfig = Field(default_factory=TokenizationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    dates: DateConfig = Field(default_factory=DateConfig)

    # Processing parameters
    num_workers: int = Field(default=4, description="Number of parallel workers")
    batch_size: int = Field(default=32, description="Batch size for processing")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    model_config = {"env_prefix": "NOVADB_"}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()
