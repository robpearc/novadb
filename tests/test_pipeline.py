"""Comprehensive test suite for NovaDB AlphaFold3 data pipeline.

Tests cover:
- Structure parsing (mmCIF)
- Tokenization
- Feature extraction
- Cropping strategies
- MSA processing
- Clustering
- Bond detection
- Storage backends
"""

import io
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_mmcif_content() -> str:
    """Minimal valid mmCIF content for testing."""
    return """data_TEST
#
_entry.id   TEST
#
_cell.length_a          50.000
_cell.length_b          60.000
_cell.length_c          70.000
_cell.angle_alpha       90.00
_cell.angle_beta        90.00
_cell.angle_gamma       90.00
#
_exptl.method           'X-RAY DIFFRACTION'
#
_refine.ls_d_res_high   2.00
#
_pdbx_database_status.recvd_initial_deposition_date  2020-01-01
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
ATOM   1  N  N   ALA A 1 1 ? 0.000  0.000  0.000  1.00 10.00 1 A
ATOM   2  C  CA  ALA A 1 1 ? 1.458  0.000  0.000  1.00 10.00 1 A
ATOM   3  C  C   ALA A 1 1 ? 2.009  1.420  0.000  1.00 10.00 1 A
ATOM   4  O  O   ALA A 1 1 ? 1.246  2.380  0.000  1.00 10.00 1 A
ATOM   5  C  CB  ALA A 1 1 ? 1.986 -0.728 -1.232  1.00 10.00 1 A
ATOM   6  N  N   GLY A 1 2 ? 3.320  1.560  0.000  1.00 10.00 2 A
ATOM   7  C  CA  GLY A 1 2 ? 3.970  2.870  0.000  1.00 10.00 2 A
ATOM   8  C  C   GLY A 1 2 ? 5.480  2.750  0.000  1.00 10.00 2 A
ATOM   9  O  O   GLY A 1 2 ? 6.040  1.660  0.000  1.00 10.00 2 A
#
"""


@pytest.fixture
def sample_structure():
    """Create a sample Structure object for testing."""
    from novadb.data.parsers.structure import Structure, Chain, Residue, Atom, ChainType
    
    # Create atoms
    atoms_ala = {
        "N": Atom(name="N", element="N", coords=np.array([0.0, 0.0, 0.0])),
        "CA": Atom(name="CA", element="C", coords=np.array([1.458, 0.0, 0.0])),
        "C": Atom(name="C", element="C", coords=np.array([2.009, 1.420, 0.0])),
        "O": Atom(name="O", element="O", coords=np.array([1.246, 2.380, 0.0])),
        "CB": Atom(name="CB", element="C", coords=np.array([1.986, -0.728, -1.232])),
    }
    
    atoms_gly = {
        "N": Atom(name="N", element="N", coords=np.array([3.320, 1.560, 0.0])),
        "CA": Atom(name="CA", element="C", coords=np.array([3.970, 2.870, 0.0])),
        "C": Atom(name="C", element="C", coords=np.array([5.480, 2.750, 0.0])),
        "O": Atom(name="O", element="O", coords=np.array([6.040, 1.660, 0.0])),
    }
    
    # Create residues
    res_ala = Residue(name="ALA", seq_id=1, atoms=atoms_ala)
    res_gly = Residue(name="GLY", seq_id=2, atoms=atoms_gly)
    
    # Create chain
    chain = Chain(
        chain_id="A",
        chain_type=ChainType.PROTEIN,
        entity_id="1",
        residues=[res_ala, res_gly],
    )
    
    # Create structure
    structure = Structure(
        pdb_id="TEST",
        chains={"A": chain},
        resolution=2.0,
        method="X-RAY DIFFRACTION",
        release_date="2020-01-01",
    )
    
    return structure


@pytest.fixture
def sample_msa() -> Dict[str, str]:
    """Sample MSA sequences for testing."""
    return {
        "query": "AGHK",
        "seq1": "AGHK",
        "seq2": "AGHR",
        "seq3": "AGHK",
        "seq4": "ADHK",
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Structure Parsing Tests
# =============================================================================

class TestMMCIFParser:
    """Tests for mmCIF parser."""
    
    def test_parse_basic_structure(self, sample_mmcif_content, temp_dir):
        """Test parsing a basic mmCIF file."""
        from novadb.data.parsers.mmcif_parser import MMCIFParser
        
        # Write test file
        mmcif_path = temp_dir / "test.cif"
        mmcif_path.write_text(sample_mmcif_content)
        
        # Parse
        parser = MMCIFParser()
        structure = parser.parse(str(mmcif_path))
        
        # Verify
        assert structure is not None
        assert structure.pdb_id == "TEST"
        assert "A" in structure.chains
        assert len(structure.chains["A"].residues) == 2
        assert structure.chains["A"].residues[0].name == "ALA"
        assert structure.chains["A"].residues[1].name == "GLY"
    
    def test_parse_atom_coordinates(self, sample_mmcif_content, temp_dir):
        """Test that atom coordinates are parsed correctly."""
        from novadb.data.parsers.mmcif_parser import MMCIFParser
        
        mmcif_path = temp_dir / "test.cif"
        mmcif_path.write_text(sample_mmcif_content)
        
        parser = MMCIFParser()
        structure = parser.parse(str(mmcif_path))
        
        # Check CA atom of first residue
        res = structure.chains["A"].residues[0]
        ca = res.atoms.get("CA")
        assert ca is not None
        np.testing.assert_array_almost_equal(
            ca.coords, [1.458, 0.0, 0.0], decimal=3
        )
    
    def test_parse_metadata(self, sample_mmcif_content, temp_dir):
        """Test parsing structure metadata."""
        from novadb.data.parsers.mmcif_parser import MMCIFParser
        
        mmcif_path = temp_dir / "test.cif"
        mmcif_path.write_text(sample_mmcif_content)
        
        parser = MMCIFParser()
        structure = parser.parse(str(mmcif_path))
        
        assert structure.resolution == 2.0
        assert structure.method == "X-RAY DIFFRACTION"


class TestStructureModel:
    """Tests for Structure data model."""
    
    def test_chain_sequence(self, sample_structure):
        """Test getting chain sequence."""
        chain = sample_structure.chains["A"]
        seq = chain.get_sequence()
        assert seq == "AG"
    
    def test_residue_heavy_atoms(self, sample_structure):
        """Test getting heavy atom coordinates."""
        res = sample_structure.chains["A"].residues[0]  # ALA
        coords = res.heavy_atom_coords
        
        assert coords.shape[0] == 5  # N, CA, C, O, CB
        assert coords.shape[1] == 3
    
    def test_chain_type_classification(self, sample_structure):
        """Test chain type is correctly set."""
        from novadb.data.parsers.structure import ChainType
        
        chain = sample_structure.chains["A"]
        assert chain.chain_type == ChainType.PROTEIN


# =============================================================================
# Tokenization Tests
# =============================================================================

class TestTokenizer:
    """Tests for tokenization."""
    
    def test_tokenize_protein(self, sample_structure):
        """Test tokenizing a protein structure."""
        from novadb.processing.tokenization.tokenizer import Tokenizer
        
        tokenizer = Tokenizer()
        result = tokenizer.tokenize(sample_structure)
        
        # Should have 2 tokens (one per residue)
        assert len(result.tokens) == 2
        
        # Check token properties
        assert result.tokens[0].residue_name == "ALA"
        assert result.tokens[1].residue_name == "GLY"
    
    def test_token_center_atoms(self, sample_structure):
        """Test that center atoms are correctly identified."""
        from novadb.processing.tokenization.tokenizer import Tokenizer
        
        tokenizer = Tokenizer()
        result = tokenizer.tokenize(sample_structure)
        
        # For proteins, center atom should be CA
        for token in result.tokens:
            assert token.center_atom_name == "CA"
    
    def test_tokenize_preserves_atom_mapping(self, sample_structure):
        """Test that atom-to-token mapping is correct."""
        from novadb.processing.tokenization.tokenizer import Tokenizer
        
        tokenizer = Tokenizer()
        result = tokenizer.tokenize(sample_structure)
        
        # Each token should have its atoms
        ala_token = result.tokens[0]
        assert len(ala_token.atom_names) == 5  # N, CA, C, O, CB


# =============================================================================
# Feature Extraction Tests
# =============================================================================

class TestFeatureExtractor:
    """Tests for feature extraction."""
    
    def test_extract_basic_features(self, sample_structure):
        """Test extracting basic features."""
        from novadb.processing.tokenization.tokenizer import Tokenizer
        from novadb.processing.features.features import FeatureExtractor
        
        tokenizer = Tokenizer()
        tokenized = tokenizer.tokenize(sample_structure)
        
        extractor = FeatureExtractor()
        features = extractor.extract(tokenized)
        
        # Check feature shapes
        assert features.token_index.shape[0] == 2
        assert features.residue_index.shape[0] == 2
        assert features.restype.shape == (2, 32)  # One-hot with 32 classes
    
    def test_restype_encoding(self, sample_structure):
        """Test residue type encoding."""
        from novadb.processing.tokenization.tokenizer import Tokenizer
        from novadb.processing.features.features import FeatureExtractor
        
        tokenizer = Tokenizer()
        tokenized = tokenizer.tokenize(sample_structure)
        
        extractor = FeatureExtractor()
        features = extractor.extract(tokenized)
        
        # ALA should be encoded as index 0
        # GLY should be encoded as index 7 (3-letter alphabetical order)
        ala_idx = features.restype[0].argmax()
        gly_idx = features.restype[1].argmax()

        assert ala_idx == 0  # ALA
        assert gly_idx == 7  # GLY
    
    def test_reference_positions(self, sample_structure):
        """Test reference conformer positions."""
        from novadb.processing.tokenization.tokenizer import Tokenizer
        from novadb.processing.features.features import FeatureExtractor
        
        tokenizer = Tokenizer()
        tokenized = tokenizer.tokenize(sample_structure)
        
        extractor = FeatureExtractor()
        features = extractor.extract(tokenized)
        
        # Reference positions should be populated
        assert features.ref_pos is not None
        assert not np.all(features.ref_pos == 0)


# =============================================================================
# Cropping Tests
# =============================================================================

class TestCropping:
    """Tests for cropping strategies."""
    
    def test_contiguous_crop(self, sample_structure):
        """Test contiguous cropping."""
        from novadb.processing.tokenization.tokenizer import Tokenizer
        from novadb.processing.cropping import Cropper, CropConfig
        
        tokenizer = Tokenizer()
        tokenized = tokenizer.tokenize(sample_structure)
        
        config = CropConfig(max_tokens=1)
        cropper = Cropper(config)
        
        cropped = cropper.contiguous_crop(tokenized, 1)
        
        assert len(cropped.tokens) == 1
    
    def test_spatial_crop(self, sample_structure):
        """Test spatial cropping."""
        from novadb.processing.tokenization.tokenizer import Tokenizer
        from novadb.processing.cropping import Cropper, CropConfig
        
        tokenizer = Tokenizer()
        tokenized = tokenizer.tokenize(sample_structure)
        
        config = CropConfig(max_tokens=1)
        cropper = Cropper(config)
        
        cropped = cropper.spatial_crop(tokenized, 1)
        
        assert len(cropped.tokens) == 1
    
    def test_crop_preserves_connectivity(self, sample_structure):
        """Test that cropping preserves chain connectivity info."""
        from novadb.processing.tokenization.tokenizer import Tokenizer
        from novadb.processing.cropping import Cropper, CropConfig
        
        tokenizer = Tokenizer()
        tokenized = tokenizer.tokenize(sample_structure)
        
        config = CropConfig(max_tokens=2)
        cropper = Cropper(config)
        
        cropped = cropper.contiguous_crop(tokenized, 2)
        
        # Should maintain chain info
        assert all(t.chain_id == "A" for t in cropped.tokens)


# =============================================================================
# Clustering Tests
# =============================================================================

class TestClustering:
    """Tests for sequence clustering."""
    
    def test_identity_clustering(self):
        """Test 100% identity clustering."""
        from novadb.processing.curation.clustering import IdentityClusterer
        
        sequences = {
            "seq1": "AGHK",
            "seq2": "AGHK",
            "seq3": "MGHK",
            "seq4": "AGHK",
        }
        
        clusterer = IdentityClusterer()
        results = clusterer.cluster(sequences)
        
        # seq1, seq2, seq4 should be in same cluster
        assert results["seq1"].cluster_id == results["seq2"].cluster_id
        assert results["seq1"].cluster_id == results["seq4"].cluster_id
        
        # seq3 should be in different cluster
        assert results["seq3"].cluster_id != results["seq1"].cluster_id
    
    def test_sequence_identity_calculation(self):
        """Test sequence identity calculation."""
        from novadb.processing.curation.clustering import compute_sequence_identity
        
        # Identical sequences
        assert compute_sequence_identity("AGHK", "AGHK") == 1.0
        
        # 50% identity
        assert compute_sequence_identity("AGHK", "XGHX") == 0.5
        
        # 0% identity
        assert compute_sequence_identity("AAAA", "BBBB") == 0.0


class TestInterfaceClustering:
    """Tests for interface clustering."""
    
    def test_interface_detection(self, sample_structure):
        """Test detecting interfaces between chains."""
        from novadb.processing.curation.interface_clustering import InterfaceDetector
        
        detector = InterfaceDetector(contact_distance=8.0)
        interfaces = detector.detect_interfaces(sample_structure)
        
        # Single chain structure has no interfaces
        assert len(interfaces) == 0
    
    def test_interface_type(self):
        """Test interface type identification."""
        from novadb.data.parsers.structure import ChainType
        from novadb.processing.curation.interface_clustering import InterfaceType
        
        itype = InterfaceType(ChainType.PROTEIN, ChainType.LIGAND)
        
        assert "protein" in itype.name.lower()
        assert "ligand" in itype.name.lower()


# =============================================================================
# Bond Detection Tests
# =============================================================================

class TestBondDetection:
    """Tests for bond feature extraction."""
    
    def test_detect_backbone_bonds(self, sample_structure):
        """Test detecting backbone connectivity."""
        from novadb.processing.features.bond_features import BondDetector
        
        detector = BondDetector(include_polymer_backbone=True)
        bonds = detector.detect_bonds(sample_structure)
        
        # Should detect peptide bond between ALA and GLY
        backbone_bonds = [b for b in bonds if b.is_polymer_bond]
        assert len(backbone_bonds) >= 1
    
    def test_bond_feature_matrix(self, sample_structure):
        """Test token bond matrix generation."""
        from novadb.processing.tokenization.tokenizer import Tokenizer
        from novadb.processing.features.bond_features import BondFeatureExtractor
        
        tokenizer = Tokenizer()
        tokenized = tokenizer.tokenize(sample_structure)
        
        extractor = BondFeatureExtractor(include_polymer_backbone=True)
        features = extractor.extract_from_tokenized(sample_structure, tokenized)
        
        # Check matrix shape
        assert features.token_bonds.shape == (2, 2)
        
        # Matrix should be symmetric
        np.testing.assert_array_equal(
            features.token_bonds,
            features.token_bonds.T
        )


# =============================================================================
# Storage Backend Tests
# =============================================================================

class TestLocalStorage:
    """Tests for local filesystem storage."""
    
    def test_put_and_get(self, temp_dir):
        """Test storing and retrieving data."""
        from novadb.storage.backends import LocalStorage
        
        storage = LocalStorage(str(temp_dir))
        
        # Store data
        data = b"test data"
        storage.put("test/file.bin", data)
        
        # Retrieve data
        retrieved = storage.get("test/file.bin")
        assert retrieved == data
    
    def test_exists(self, temp_dir):
        """Test checking key existence."""
        from novadb.storage.backends import LocalStorage
        
        storage = LocalStorage(str(temp_dir))
        
        assert not storage.exists("nonexistent")
        
        storage.put("exists", b"data")
        assert storage.exists("exists")
    
    def test_delete(self, temp_dir):
        """Test deleting data."""
        from novadb.storage.backends import LocalStorage
        
        storage = LocalStorage(str(temp_dir))
        
        storage.put("to_delete", b"data")
        assert storage.exists("to_delete")
        
        storage.delete("to_delete")
        assert not storage.exists("to_delete")
    
    def test_list_keys(self, temp_dir):
        """Test listing keys."""
        from novadb.storage.backends import LocalStorage
        
        storage = LocalStorage(str(temp_dir))
        
        storage.put("dir1/file1", b"data")
        storage.put("dir1/file2", b"data")
        storage.put("dir2/file1", b"data")
        
        keys = storage.list("dir1")
        assert len(keys) == 2
        assert all("dir1" in k for k in keys)
    
    def test_json_storage(self, temp_dir):
        """Test JSON data storage."""
        from novadb.storage.backends import LocalStorage
        
        storage = LocalStorage(str(temp_dir))
        
        data = {"key": "value", "number": 42}
        storage.put_json("test.json", data)
        
        retrieved = storage.get_json("test.json")
        assert retrieved == data
    
    def test_numpy_storage(self, temp_dir):
        """Test numpy array storage."""
        from novadb.storage.backends import LocalStorage
        
        storage = LocalStorage(str(temp_dir))
        
        arrays = {
            "arr1": np.array([1, 2, 3]),
            "arr2": np.array([[1, 2], [3, 4]]),
        }
        storage.put_numpy("test.npz", arrays)
        
        retrieved = storage.get_numpy("test.npz")
        np.testing.assert_array_equal(retrieved["arr1"], arrays["arr1"])
        np.testing.assert_array_equal(retrieved["arr2"], arrays["arr2"])


# =============================================================================
# AF3 Constants Tests
# =============================================================================

class TestAF3Constants:
    """Tests for AF3 constants."""
    
    def test_crystallization_aids(self):
        """Test crystallization aid detection."""
        from novadb.constants.af3_constants import is_crystallization_aid
        
        assert is_crystallization_aid("SO4")
        assert is_crystallization_aid("GOL")
        assert not is_crystallization_aid("ATP")
    
    def test_glycan_detection(self):
        """Test glycan detection."""
        from novadb.constants.af3_constants import is_glycan
        
        # Common glycans
        assert is_glycan("NAG")  # N-acetylglucosamine
        assert is_glycan("MAN")  # Mannose
        assert not is_glycan("ALA")  # Amino acid
    
    def test_ion_detection(self):
        """Test ion detection."""
        from novadb.constants.af3_constants import is_ion
        
        assert is_ion("ZN")
        assert is_ion("MG")
        assert is_ion("CA")
        assert not is_ion("ALA")
    
    def test_standard_residues(self):
        """Test standard residue detection."""
        from novadb.constants.af3_constants import (
            is_standard_amino_acid,
            is_standard_nucleotide,
        )
        
        # Amino acids
        assert is_standard_amino_acid("ALA")
        assert is_standard_amino_acid("GLY")
        assert not is_standard_amino_acid("NAG")
        
        # Nucleotides
        assert is_standard_nucleotide("A")
        assert is_standard_nucleotide("DA")
        assert not is_standard_nucleotide("ALA")
    
    def test_clustering_thresholds(self):
        """Test clustering threshold retrieval."""
        from novadb.constants.af3_constants import get_clustering_threshold
        
        # Proteins: 40%
        assert get_clustering_threshold("ALA") == 0.4
        
        # Short peptides: 100%
        assert get_clustering_threshold("ALA", chain_length=5) == 1.0
        
        # Nucleotides: 100%
        assert get_clustering_threshold("A") == 1.0


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from novadb.config import Config

        config = Config()

        assert config.cropping.pdb_contiguous_weight > 0
        assert config.msa_processing.max_msa_rows > 0
        assert config.template_search.max_templates > 0

    def test_config_to_dict(self):
        """Test configuration serialization."""
        from novadb.config import Config

        config = Config()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert "cropping" in d
        assert "msa_processing" in d

    def test_config_from_dict(self):
        """Test configuration deserialization."""
        from novadb.config import Config

        d = {
            "cropping": {"pdb_contiguous_weight": 0.3},
            "msa_processing": {"max_msa_rows": 1000},
        }

        config = Config.from_dict(d)

        assert config.cropping.pdb_contiguous_weight == 0.3
        assert config.msa_processing.max_msa_rows == 1000


# =============================================================================
# MSA Processing Tests
# =============================================================================

class TestMSAProcessing:
    """Tests for MSA processing."""
    
    def test_msa_deduplication(self, sample_msa):
        """Test MSA sequence deduplication."""
        from novadb.search.msa.processing import deduplicate_msa
        
        sequences = list(sample_msa.values())
        deduplicated = deduplicate_msa(sequences)
        
        # Should remove duplicate "AGHK"
        assert len(deduplicated) < len(sequences)
    
    def test_msa_profile_computation(self, sample_msa):
        """Test MSA profile computation."""
        from novadb.search.msa.processing import compute_msa_profile
        
        sequences = list(sample_msa.values())
        profile = compute_msa_profile(sequences)
        
        # Profile should sum to 1 at each position
        for pos in profile:
            total = sum(pos.values())
            assert abs(total - 1.0) < 0.01


# =============================================================================
# Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    def test_process_structure_end_to_end(
        self,
        sample_mmcif_content,
        temp_dir
    ):
        """Test processing a structure through the full pipeline."""
        from novadb.config import Config
        from novadb.pipeline.pipeline import DataPipeline
        
        # Write test file
        mmcif_path = temp_dir / "test.cif"
        mmcif_path.write_text(sample_mmcif_content)
        
        # Create pipeline with minimal config
        config = Config()
        config.storage.local_path = str(temp_dir / "output")
        config.filtering.min_resolved_residues = 1  # Allow small test structures

        pipeline = DataPipeline(config)
        
        # Process structure (without MSA/templates)
        result = pipeline.process_structure(
            str(mmcif_path),
            run_msa=False,
            run_templates=False,
        )
        
        assert result is not None
        assert result.pdb_id == "TEST"
        assert result.features is not None
        assert result.tokenized is not None
    
    def test_pipeline_filtering(self, sample_structure, temp_dir):
        """Test pipeline filtering logic."""
        from novadb.config import Config, FilteringConfig
        from novadb.processing.curation.filtering import StructureFilter
        
        # Create filter with strict resolution
        filter_config = FilteringConfig(max_resolution=1.0)  # Very strict
        
        structure_filter = StructureFilter(filtering_config=filter_config)
        result = structure_filter.filter(sample_structure)
        
        # Should fail because resolution is 2.0
        assert not result.passed
        assert "resolution" in result.reason.lower()


# =============================================================================
# Conformer Generation Tests
# =============================================================================

class TestConformerGeneration:
    """Tests for conformer generation."""
    
    def test_kabsch_alignment(self):
        """Test Kabsch alignment algorithm."""
        from novadb.processing.conformer import kabsch_align, compute_rmsd
        
        # Create test coordinates
        mobile = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        # Rotate by 90 degrees around Z
        target = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
        ], dtype=np.float32)
        
        aligned, rmsd = kabsch_align(mobile, target)
        
        # RMSD should be near zero after alignment
        assert rmsd < 0.1
    
    def test_conformer_config(self):
        """Test conformer generation configuration."""
        from novadb.processing.conformer import ConformerGenerationConfig
        
        config = ConformerGenerationConfig(
            num_conformers=5,
            force_field="MMFF94",
        )
        
        assert config.num_conformers == 5
        assert config.force_field == "MMFF94"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
