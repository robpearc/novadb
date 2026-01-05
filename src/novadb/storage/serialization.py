"""Data serialization for storage.

Provides serialization/deserialization for:
- Processed features (numpy arrays)
- Tokenized structures
- MSA data
- Metadata
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar
import numpy as np

from novadb.processing.features.features import InputFeatures
from novadb.processing.tokenization.tokenizer import (
    Token,
    TokenType,
    TokenizedStructure,
)
from novadb.search.msa.msa import MSA, MSASequence
from novadb.storage.backends import StorageBackend


T = TypeVar("T")


class DataSerializer:
    """Serialize and deserialize processed data."""

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    def save_features(
        self,
        features: InputFeatures,
        key: str,
        compress: bool = True,
    ) -> None:
        """Save input features to storage.
        
        Args:
            features: InputFeatures to save
            key: Storage key
            compress: Whether to compress numpy arrays
        """
        # Convert to dictionary of arrays and metadata
        data = features.to_dict()

        # Separate numpy arrays from metadata
        arrays = {}
        metadata = {}

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                arrays[k] = v
            else:
                metadata[k] = v

        # Save arrays
        self.storage.put_numpy(f"{key}.npz", arrays)

        # Save metadata
        self.storage.put_json(f"{key}.json", metadata)

    def load_features(self, key: str) -> InputFeatures:
        """Load input features from storage.
        
        Args:
            key: Storage key
            
        Returns:
            InputFeatures instance
        """
        # Load arrays
        arrays = self.storage.get_numpy(f"{key}.npz")

        # Load metadata
        metadata = self.storage.get_json(f"{key}.json")

        # Combine and create InputFeatures
        combined = {**arrays, **metadata}
        return InputFeatures.from_dict(combined)

    def save_tokenized_structure(
        self,
        tokenized: TokenizedStructure,
        key: str,
    ) -> None:
        """Save tokenized structure to storage.
        
        Args:
            tokenized: TokenizedStructure to save
            key: Storage key
        """
        # Convert tokens to serializable format
        token_data = []
        for token in tokenized.tokens:
            atom_data = {}
            for name, atom in token.atoms.items():
                atom_data[name] = {
                    "name": atom.name,
                    "element": atom.element,
                    "coords": atom.coords.tolist(),
                    "b_factor": atom.b_factor,
                    "occupancy": atom.occupancy,
                }

            token_data.append({
                "token_index": token.token_index,
                "residue_index": token.residue_index,
                "chain_id": token.chain_id,
                "chain_index": token.chain_index,
                "entity_id": token.entity_id,
                "token_type": token.token_type.name,
                "residue_name": token.residue_name,
                "center_atom_name": token.center_atom_name,
                "atoms": atom_data,
            })

        data = {
            "pdb_id": tokenized.pdb_id,
            "chain_id_to_index": tokenized.chain_id_to_index,
            "entity_id_map": tokenized.entity_id_map,
            "tokens": token_data,
        }

        self.storage.put_json(f"{key}.json", data)

    def load_tokenized_structure(self, key: str) -> TokenizedStructure:
        """Load tokenized structure from storage.
        
        Args:
            key: Storage key
            
        Returns:
            TokenizedStructure instance
        """
        from novadb.data.parsers.structure import Atom

        data = self.storage.get_json(f"{key}.json")

        # Reconstruct tokens
        tokens = []
        for td in data["tokens"]:
            # Reconstruct atoms
            atoms = {}
            for name, ad in td["atoms"].items():
                atoms[name] = Atom(
                    name=ad["name"],
                    element=ad["element"],
                    coords=np.array(ad["coords"], dtype=np.float32),
                    b_factor=ad.get("b_factor", 0.0),
                    occupancy=ad.get("occupancy", 1.0),
                )

            token = Token(
                token_index=td["token_index"],
                residue_index=td["residue_index"],
                chain_id=td["chain_id"],
                chain_index=td["chain_index"],
                entity_id=td["entity_id"],
                token_type=TokenType[td["token_type"]],
                residue_name=td["residue_name"],
                center_atom_name=td["center_atom_name"],
                atoms=atoms,
            )
            tokens.append(token)

        return TokenizedStructure(
            tokens=tokens,
            pdb_id=data["pdb_id"],
            chain_id_to_index=data["chain_id_to_index"],
            entity_id_map=data["entity_id_map"],
        )

    def save_msa(
        self,
        msa: MSA,
        key: str,
        format: str = "a3m",
    ) -> None:
        """Save MSA to storage.
        
        Args:
            msa: MSA to save
            key: Storage key
            format: Output format (a3m, stockholm, json)
        """
        if format == "a3m":
            content = msa.to_a3m()
            self.storage.put(f"{key}.a3m", content.encode("utf-8"))
        elif format == "stockholm":
            content = msa.to_stockholm()
            self.storage.put(f"{key}.sto", content.encode("utf-8"))
        elif format == "json":
            data = {
                "query_id": msa.query_id,
                "query_sequence": msa.query_sequence,
                "source": msa.source,
                "sequences": [
                    {
                        "sequence_id": seq.sequence_id,
                        "sequence": seq.sequence,
                        "description": seq.description,
                        "species": seq.species,
                        "start": seq.start,
                        "end": seq.end,
                        "deletion_matrix": seq.deletion_matrix,
                    }
                    for seq in msa.sequences
                ],
            }
            self.storage.put_json(f"{key}.json", data)
        else:
            raise ValueError(f"Unknown format: {format}")

    def load_msa(
        self,
        key: str,
        format: str = "a3m",
    ) -> MSA:
        """Load MSA from storage.
        
        Args:
            key: Storage key
            format: Input format (a3m, stockholm, json)
            
        Returns:
            MSA instance
        """
        if format == "a3m":
            content = self.storage.get(f"{key}.a3m").decode("utf-8")
            return MSA.from_a3m(content)
        elif format == "stockholm":
            content = self.storage.get(f"{key}.sto").decode("utf-8")
            return MSA.from_stockholm(content)
        elif format == "json":
            data = self.storage.get_json(f"{key}.json")
            sequences = [
                MSASequence(
                    sequence_id=s["sequence_id"],
                    sequence=s["sequence"],
                    description=s.get("description", ""),
                    species=s.get("species"),
                    start=s.get("start"),
                    end=s.get("end"),
                    deletion_matrix=s.get("deletion_matrix"),
                )
                for s in data["sequences"]
            ]
            return MSA(
                query_id=data["query_id"],
                query_sequence=data["query_sequence"],
                sequences=sequences,
                source=data.get("source", ""),
            )
        else:
            raise ValueError(f"Unknown format: {format}")

    def save_batch(
        self,
        items: List[InputFeatures],
        prefix: str,
    ) -> List[str]:
        """Save a batch of features.
        
        Args:
            items: List of InputFeatures
            prefix: Key prefix
            
        Returns:
            List of keys
        """
        keys = []
        for i, features in enumerate(items):
            key = f"{prefix}/{features.pdb_id}_{i:06d}"
            self.save_features(features, key)
            keys.append(key)

        # Save index
        index = {
            "count": len(keys),
            "keys": keys,
        }
        self.storage.put_json(f"{prefix}/index.json", index)

        return keys

    def load_batch(
        self,
        prefix: str,
        limit: Optional[int] = None,
    ) -> List[InputFeatures]:
        """Load a batch of features.
        
        Args:
            prefix: Key prefix
            limit: Maximum number to load
            
        Returns:
            List of InputFeatures
        """
        # Load index
        index = self.storage.get_json(f"{prefix}/index.json")
        keys = index["keys"]

        if limit is not None:
            keys = keys[:limit]

        return [self.load_features(key) for key in keys]


class DatasetWriter:
    """Write processed datasets for training."""

    def __init__(
        self,
        storage: StorageBackend,
        shard_size: int = 1000,
    ):
        self.storage = storage
        self.serializer = DataSerializer(storage)
        self.shard_size = shard_size

        self._current_shard: List[InputFeatures] = []
        self._shard_idx = 0
        self._total_count = 0

    def add(self, features: InputFeatures) -> None:
        """Add features to dataset.
        
        Args:
            features: InputFeatures to add
        """
        self._current_shard.append(features)
        self._total_count += 1

        if len(self._current_shard) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self) -> None:
        """Flush current shard to storage."""
        if not self._current_shard:
            return

        shard_key = f"shard_{self._shard_idx:06d}"
        self.serializer.save_batch(self._current_shard, shard_key)

        self._current_shard = []
        self._shard_idx += 1

    def finalize(self) -> Dict[str, Any]:
        """Finalize dataset writing.
        
        Returns:
            Dataset metadata
        """
        # Flush remaining
        self._flush_shard()

        # Write dataset metadata
        metadata = {
            "num_shards": self._shard_idx,
            "total_count": self._total_count,
            "shard_size": self.shard_size,
        }
        self.storage.put_json("metadata.json", metadata)

        return metadata


class DatasetReader:
    """Read processed datasets for training."""

    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self.serializer = DataSerializer(storage)

        # Load metadata
        self.metadata = self.storage.get_json("metadata.json")

    @property
    def num_samples(self) -> int:
        """Total number of samples in dataset."""
        return self.metadata["total_count"]

    @property
    def num_shards(self) -> int:
        """Number of shards in dataset."""
        return self.metadata["num_shards"]

    def load_shard(self, shard_idx: int) -> List[InputFeatures]:
        """Load a single shard.
        
        Args:
            shard_idx: Shard index
            
        Returns:
            List of InputFeatures in shard
        """
        shard_key = f"shard_{shard_idx:06d}"
        return self.serializer.load_batch(shard_key)

    def iter_samples(self):
        """Iterate over all samples."""
        for shard_idx in range(self.num_shards):
            for features in self.load_shard(shard_idx):
                yield features
