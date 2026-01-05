# NovaDB: AlphaFold3-Style Dataset Curation Pipeline

A comprehensive Python framework for replicating the dataset curation, processing, and feature engineering pipeline described in the AlphaFold3 supplementary materials. This pipeline prepares structural biology data for training structure prediction models.

## Overview

NovaDB implements the complete data processing pipeline for biomolecular structure prediction, including:

- **Data Parsing**: mmCIF file parsing with structure cleanup and bioassembly expansion
- **Genetic Search**: MSA generation using Jackhmmer, HHBlits, and nhmmer
- **Template Search**: Template hit retrieval and processing for protein chains
- **Tokenization**: Flexible tokenization for proteins, nucleic acids, and ligands
- **Feature Engineering**: Comprehensive feature generation for model training
- **Dataset Curation**: PDB filtering, clustering, and distillation set preparation
- **Cloud Storage**: Integration with AWS S3, Google Cloud Storage, and Azure Blob Storage

## Architecture

```
novadb/
├── data/                    # Data handling and storage
│   ├── parsers/            # mmCIF and structure parsers
│   ├── databases/          # Database interfaces (PDB, UniProt, etc.)
│   └── storage/            # Cloud and local storage adapters
├── search/                  # Sequence search tools
│   ├── msa/                # MSA generation (Jackhmmer, HHBlits, nhmmer)
│   └── templates/          # Template search and processing
├── processing/              # Data processing pipeline
│   ├── tokenization/       # Tokenization schemes
│   ├── cropping/           # Spatial and contiguous cropping
│   └── filtering/          # Structure and chain filtering
├── features/                # Feature engineering
│   ├── token/              # Per-token features
│   ├── msa/                # MSA-derived features
│   ├── template/           # Template features
│   ├── bond/               # Bond and connectivity features
│   └── reference/          # Reference conformer features
├── curation/                # Dataset curation
│   ├── clustering/         # Sequence and interface clustering
│   ├── weighting/          # Sample weighting schemes
│   └── distillation/       # Distillation dataset handling
└── pipeline/                # Pipeline orchestration
    ├── config/             # Configuration management
    └── runner/             # Pipeline execution
```

## Key Components

### 1. Data Parsing (Section 2.1 of AF3 Supplement)

- Parse mmCIF files with metadata extraction
- Structure cleanup: alternative locations, MSE→MET conversion, water removal
- Bioassembly expansion for biologically relevant complexes
- Arginine naming ambiguity resolution

### 2. Genetic Search (Sections 2.2-2.3)

**Protein MSA Generation:**
- UniRef90, UniProt, Uniclust30+BFD, Reduced BFD, MGnify searches
- Jackhmmer and HHBlits with specific flag configurations
- MSA deduplication and species-based pairing

**RNA MSA Generation:**
- Rfam, RNACentral, Nucleotide collection searches
- nhmmer with RNA-specific parameters
- Sequence realignment with hmmalign

### 3. Template Search (Section 2.4)

- HMM-based template search from UniRef90 MSA
- Release date filtering and duplicate removal
- Template structure featurization

### 4. Training Data (Section 2.5)

**Datasets:**
- Weighted PDB (ground truth structures)
- Disordered protein PDB distillation
- Protein monomer distillation (from MGnify)
- RNA distillation (from Rfam)
- Transcription factor distillation (JASPAR-based)

**Clustering:**
- 40% sequence identity for proteins
- 100% identity for nucleic acids
- CCD-based clustering for ligands
- Interface-based clustering

### 5. Tokenization (Section 2.6)

- Standard amino acids: 1 residue = 1 token
- Standard nucleotides: 1 nucleotide = 1 token
- Modified residues/ligands: 1 atom = 1 token
- Token center atoms: Cα for proteins, C1' for nucleotides

### 6. Cropping (Section 2.7)

- Contiguous cropping for sequence locality
- Spatial cropping around reference atoms
- Spatial interface cropping for interaction sites

### 7. Feature Engineering (Section 2.8)

**Token Features:**
- Position indices (residue_index, token_index)
- Chain identifiers (asym_id, entity_id, sym_id)
- Residue types and molecule type masks

**Reference Features:**
- Atom positions from RDKit conformers
- Element types, charges, atom names
- Space UIDs for residue identification

**MSA Features:**
- Processed MSA with deletion information
- Sequence profiles and deletion means

**Template Features:**
- Template residue types and distances
- Backbone frames and unit vectors

**Bond Features:**
- Polymer-ligand and ligand-ligand bonds
- Token-level bond matrices

## Installation

```bash
# Clone the repository
git clone https://github.com/novadb/novadb.git
cd novadb

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e ".[all]"

# Install HMMER tools (required for sequence searches)
# On macOS:
brew install hmmer

# On Ubuntu/Debian:
sudo apt-get install hmmer

# On conda:
conda install -c bioconda hmmer hhsuite
```

## Configuration

Create a configuration file `config.yaml`:

```yaml
# Database paths
databases:
  pdb_dir: /path/to/pdb/mmCIF
  uniref90: /path/to/uniref90/uniref90.fasta
  uniprot: /path/to/uniprot/uniprot.fasta
  bfd: /path/to/bfd
  mgnify: /path/to/mgnify/mgnify.fasta
  rfam: /path/to/rfam/Rfam.cm
  rnacentral: /path/to/rnacentral/rnacentral.fasta
  nt: /path/to/nt/nt.fasta

# Processing parameters
processing:
  max_msa_sequences: 16384
  max_templates: 4
  max_tokens: 384  # or 640, 768 for fine-tuning stages
  max_chains: 20
  resolution_cutoff: 9.0

# Clustering parameters
clustering:
  protein_identity: 0.4
  nucleic_identity: 1.0
  peptide_identity: 1.0

# Storage configuration
storage:
  backend: s3  # or 'gcs', 'azure', 'local'
  bucket: novadb-training-data
  region: us-east-1

# Training date cutoffs
dates:
  training_cutoff: "2021-09-30"
  template_cutoff: "2021-09-30"
```

## Usage

### Command Line Interface

```bash
# Run the full pipeline
novadb-pipeline run --config config.yaml --input /path/to/pdb

# Download required databases
novadb-download databases --output /path/to/databases

# Process a single structure
novadb process --input 1abc.cif --output features/

# Generate MSAs for a list of sequences
novadb msa --fasta sequences.fasta --output msas/

# Curate training dataset
novadb curate --pdb-dir /path/to/pdb --output training_set/
```

### Python API

```python
from novadb import Pipeline, Config
from novadb.data.parsers import MMCIFParser
from novadb.features import FeatureBuilder
from novadb.storage import S3Storage

# Load configuration
config = Config.from_yaml("config.yaml")

# Initialize storage
storage = S3Storage(bucket="novadb-training-data")

# Create pipeline
pipeline = Pipeline(config, storage)

# Process a structure
with open("1abc.cif") as f:
    structure = MMCIFParser().parse(f)

features = pipeline.process(structure)
pipeline.save(features, "1abc")

# Or run batch processing
pipeline.run_batch(
    input_dir="/path/to/pdb",
    output_dir="s3://novadb-training-data/features/"
)
```

## Data Flow

```
mmCIF Files
    │
    ▼
┌──────────────────┐
│   Parse & Clean  │  ← Structure cleanup, bioassembly expansion
└────────┬─────────┘
         │
    ▼         ▼
┌─────────┐ ┌─────────────┐
│   MSA   │ │  Templates  │  ← Genetic/template search
│ Search  │ │   Search    │
└────┬────┘ └──────┬──────┘
     │             │
     ▼             ▼
┌──────────────────────┐
│    Tokenization      │  ← Residue/atom to token mapping
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Feature Engineering │  ← Generate all input features
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Cropping & Filter  │  ← Spatial/contiguous cropping
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Storage (S3/GCS)   │  ← Save processed features
└──────────────────────┘
```

## References

This implementation is based on the AlphaFold3 supplementary materials:

- Abramson, J., et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." Nature (2024).
- Supplementary Information for AlphaFold 3

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.
