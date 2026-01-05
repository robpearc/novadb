"""Cropping strategies for training data.

Implements the three cropping strategies from AlphaFold3 Section 2.7:
1. Contiguous cropping: Sample contiguous blocks from chains
2. Spatial cropping: Sample tokens within spatial proximity
3. Spatial interface cropping: Sample tokens near inter-chain interfaces
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from novadb.processing.tokenization.tokenizer import (
    Token,
    TokenizedStructure,
    TokenType,
)


class CroppingStrategy(Enum):
    """Cropping strategy selection."""
    CONTIGUOUS = auto()
    SPATIAL = auto()
    SPATIAL_INTERFACE = auto()


@dataclass
class CropConfig:
    """Configuration for cropping.
    
    From AF3 Table 4 and Section 2.7.
    """
    # Maximum tokens in crop
    max_tokens: int = 384  # Ncrop_tokens from Table 4
    max_atoms: int = 4608  # Ncrop_atoms from Table 4

    # Cropping strategy probabilities
    contiguous_weight: float = 0.2
    spatial_weight: float = 0.4
    spatial_interface_weight: float = 0.4

    # Contiguous cropping parameters
    contiguous_crop_size: int = 128

    # Spatial cropping parameters
    spatial_radius: float = 24.0  # Angstroms

    # Interface parameters
    interface_distance: float = 15.0  # Angstroms


@dataclass
class CropResult:
    """Result of cropping operation."""
    token_indices: np.ndarray  # Indices of selected tokens
    strategy: CroppingStrategy
    center_token_idx: Optional[int] = None
    interface_chains: Optional[Tuple[str, str]] = None

    @property
    def num_tokens(self) -> int:
        return len(self.token_indices)


class BaseCropper(ABC):
    """Base class for cropping strategies."""

    def __init__(self, config: CropConfig):
        self.config = config

    @abstractmethod
    def crop(
        self,
        tokenized: TokenizedStructure,
        rng: np.random.Generator,
    ) -> CropResult:
        """Apply cropping to tokenized structure."""
        pass

    def _get_center_coords(self, tokens: List[Token]) -> np.ndarray:
        """Get center coordinates for all tokens."""
        coords = np.zeros((len(tokens), 3), dtype=np.float32)
        for i, token in enumerate(tokens):
            center = token.center_coords
            if center is not None:
                coords[i] = center
        return coords


class ContiguousCropper(BaseCropper):
    """Contiguous block cropping strategy.
    
    From AF3 Section 2.7: Sample contiguous blocks from each chain,
    concatenate, and keep if total tokens <= max_tokens.
    """

    def crop(
        self,
        tokenized: TokenizedStructure,
        rng: np.random.Generator,
    ) -> CropResult:
        """Crop contiguous blocks from chains."""
        tokens = tokenized.tokens
        if len(tokens) <= self.config.max_tokens:
            return CropResult(
                token_indices=np.arange(len(tokens)),
                strategy=CroppingStrategy.CONTIGUOUS,
            )

        # Group tokens by chain
        chain_tokens: Dict[str, List[Tuple[int, Token]]] = {}
        for i, token in enumerate(tokens):
            if token.chain_id not in chain_tokens:
                chain_tokens[token.chain_id] = []
            chain_tokens[token.chain_id].append((i, token))

        # Sample contiguous blocks from each chain
        selected_indices: List[int] = []

        for chain_id, chain_token_list in chain_tokens.items():
            # Calculate how many tokens to sample from this chain
            chain_fraction = len(chain_token_list) / len(tokens)
            target_tokens = int(self.config.max_tokens * chain_fraction)
            target_tokens = max(1, min(target_tokens, len(chain_token_list)))

            if target_tokens >= len(chain_token_list):
                # Take all tokens from this chain
                selected_indices.extend([idx for idx, _ in chain_token_list])
            else:
                # Sample a contiguous block
                max_start = len(chain_token_list) - target_tokens
                start = rng.integers(0, max_start + 1)
                end = start + target_tokens

                for idx, _ in chain_token_list[start:end]:
                    selected_indices.append(idx)

        # Ensure we don't exceed max_tokens
        if len(selected_indices) > self.config.max_tokens:
            selected_indices = list(
                rng.choice(
                    selected_indices,
                    size=self.config.max_tokens,
                    replace=False,
                )
            )

        return CropResult(
            token_indices=np.array(sorted(selected_indices), dtype=np.int32),
            strategy=CroppingStrategy.CONTIGUOUS,
        )


class SpatialCropper(BaseCropper):
    """Spatial proximity cropping strategy.
    
    From AF3 Section 2.7: Sample a center token, then select
    tokens within spatial_radius Angstroms.
    """

    def crop(
        self,
        tokenized: TokenizedStructure,
        rng: np.random.Generator,
    ) -> CropResult:
        """Crop tokens within spatial proximity of a random center."""
        tokens = tokenized.tokens
        if len(tokens) <= self.config.max_tokens:
            return CropResult(
                token_indices=np.arange(len(tokens)),
                strategy=CroppingStrategy.SPATIAL,
            )

        coords = self._get_center_coords(tokens)

        # Sample center token
        center_idx = rng.integers(0, len(tokens))
        center_coords = coords[center_idx]

        # Compute distances to center
        distances = np.linalg.norm(coords - center_coords, axis=1)

        # Select tokens within radius, prioritizing closer ones
        within_radius = np.where(distances <= self.config.spatial_radius)[0]

        if len(within_radius) <= self.config.max_tokens:
            selected_indices = within_radius
        else:
            # Sort by distance and take closest
            sorted_idx = within_radius[np.argsort(distances[within_radius])]
            selected_indices = sorted_idx[: self.config.max_tokens]

        # If not enough tokens, expand radius
        if len(selected_indices) < self.config.max_tokens:
            # Add more tokens by distance
            sorted_all = np.argsort(distances)
            selected_indices = sorted_all[: self.config.max_tokens]

        return CropResult(
            token_indices=np.array(sorted(selected_indices), dtype=np.int32),
            strategy=CroppingStrategy.SPATIAL,
            center_token_idx=int(center_idx),
        )


class SpatialInterfaceCropper(BaseCropper):
    """Spatial interface cropping strategy.
    
    From AF3 Section 2.7: Find inter-chain interfaces, sample
    a center token from an interface, then select tokens within
    spatial proximity.
    """

    def crop(
        self,
        tokenized: TokenizedStructure,
        rng: np.random.Generator,
    ) -> CropResult:
        """Crop tokens near inter-chain interfaces."""
        tokens = tokenized.tokens
        if len(tokens) <= self.config.max_tokens:
            return CropResult(
                token_indices=np.arange(len(tokens)),
                strategy=CroppingStrategy.SPATIAL_INTERFACE,
            )

        # Find interface tokens
        interface_info = self._find_interface_tokens(tokens)

        if not interface_info:
            # No interfaces found, fall back to spatial cropping
            spatial_cropper = SpatialCropper(self.config)
            result = spatial_cropper.crop(tokenized, rng)
            return CropResult(
                token_indices=result.token_indices,
                strategy=CroppingStrategy.SPATIAL_INTERFACE,
            )

        # Sample an interface and a token from it
        interface_idx = rng.integers(0, len(interface_info))
        interface_tokens, chain_pair = interface_info[interface_idx]

        center_idx = interface_tokens[rng.integers(0, len(interface_tokens))]

        # Now use spatial cropping from this center
        coords = self._get_center_coords(tokens)
        center_coords = coords[center_idx]
        distances = np.linalg.norm(coords - center_coords, axis=1)

        # Select tokens within radius
        sorted_idx = np.argsort(distances)
        selected_indices = sorted_idx[: self.config.max_tokens]

        return CropResult(
            token_indices=np.array(sorted(selected_indices), dtype=np.int32),
            strategy=CroppingStrategy.SPATIAL_INTERFACE,
            center_token_idx=int(center_idx),
            interface_chains=chain_pair,
        )

    def _find_interface_tokens(
        self, tokens: List[Token]
    ) -> List[Tuple[List[int], Tuple[str, str]]]:
        """Find tokens at inter-chain interfaces.
        
        Returns list of (token_indices, (chain1, chain2)) for each interface.
        """
        coords = self._get_center_coords(tokens)

        # Group tokens by chain
        chain_tokens: Dict[str, List[int]] = {}
        for i, token in enumerate(tokens):
            if token.chain_id not in chain_tokens:
                chain_tokens[token.chain_id] = []
            chain_tokens[token.chain_id].append(i)

        chains = list(chain_tokens.keys())
        if len(chains) < 2:
            return []

        interfaces: List[Tuple[List[int], Tuple[str, str]]] = []

        # Find interfaces between each pair of chains
        for i, chain1 in enumerate(chains):
            for chain2 in chains[i + 1 :]:
                interface_tokens = self._find_chain_interface(
                    chain_tokens[chain1],
                    chain_tokens[chain2],
                    coords,
                )
                if interface_tokens:
                    interfaces.append(
                        (interface_tokens, (chain1, chain2))
                    )

        return interfaces

    def _find_chain_interface(
        self,
        chain1_indices: List[int],
        chain2_indices: List[int],
        coords: np.ndarray,
    ) -> List[int]:
        """Find tokens at interface between two chains."""
        interface_tokens = []
        threshold = self.config.interface_distance

        for i in chain1_indices:
            for j in chain2_indices:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= threshold:
                    if i not in interface_tokens:
                        interface_tokens.append(i)
                    if j not in interface_tokens:
                        interface_tokens.append(j)

        return interface_tokens


class Cropper:
    """Main cropper that selects and applies cropping strategies.
    
    From AF3 Section 2.7, cropping strategies are selected with
    probabilities from Table 4.
    """

    def __init__(self, config: Optional[CropConfig] = None):
        self.config = config or CropConfig()

        self.contiguous = ContiguousCropper(self.config)
        self.spatial = SpatialCropper(self.config)
        self.spatial_interface = SpatialInterfaceCropper(self.config)

    def crop(
        self,
        tokenized: TokenizedStructure,
        rng: Optional[np.random.Generator] = None,
        strategy: Optional[CroppingStrategy] = None,
    ) -> CropResult:
        """Crop a tokenized structure.
        
        Args:
            tokenized: Structure to crop
            rng: Random number generator
            strategy: Specific strategy to use, or None to sample
            
        Returns:
            CropResult with selected token indices
        """
        if rng is None:
            rng = np.random.default_rng()

        # Select strategy if not specified
        if strategy is None:
            strategy = self._sample_strategy(rng)

        # Apply selected strategy
        if strategy == CroppingStrategy.CONTIGUOUS:
            return self.contiguous.crop(tokenized, rng)
        elif strategy == CroppingStrategy.SPATIAL:
            return self.spatial.crop(tokenized, rng)
        else:
            return self.spatial_interface.crop(tokenized, rng)

    def _sample_strategy(self, rng: np.random.Generator) -> CroppingStrategy:
        """Sample a cropping strategy based on configured weights."""
        weights = np.array([
            self.config.contiguous_weight,
            self.config.spatial_weight,
            self.config.spatial_interface_weight,
        ])
        weights = weights / weights.sum()

        strategies = [
            CroppingStrategy.CONTIGUOUS,
            CroppingStrategy.SPATIAL,
            CroppingStrategy.SPATIAL_INTERFACE,
        ]

        idx = rng.choice(len(strategies), p=weights)
        return strategies[idx]

    def contiguous_crop(
        self,
        tokenized: TokenizedStructure,
        max_tokens: int,
        rng: Optional[np.random.Generator] = None,
    ) -> TokenizedStructure:
        """Apply contiguous cropping strategy.

        Args:
            tokenized: Structure to crop
            max_tokens: Maximum tokens in result
            rng: Random number generator

        Returns:
            New TokenizedStructure with cropped tokens
        """
        if rng is None:
            rng = np.random.default_rng()

        temp_config = CropConfig(max_tokens=max_tokens)
        temp_cropper = ContiguousCropper(temp_config)
        result = temp_cropper.crop(tokenized, rng)

        return self._apply_crop_result(tokenized, result)

    def spatial_crop(
        self,
        tokenized: TokenizedStructure,
        max_tokens: int,
        rng: Optional[np.random.Generator] = None,
    ) -> TokenizedStructure:
        """Apply spatial cropping strategy.

        Args:
            tokenized: Structure to crop
            max_tokens: Maximum tokens in result
            rng: Random number generator

        Returns:
            New TokenizedStructure with cropped tokens
        """
        if rng is None:
            rng = np.random.default_rng()

        temp_config = CropConfig(max_tokens=max_tokens)
        temp_cropper = SpatialCropper(temp_config)
        result = temp_cropper.crop(tokenized, rng)

        return self._apply_crop_result(tokenized, result)

    def _apply_crop_result(
        self,
        tokenized: TokenizedStructure,
        result: CropResult,
    ) -> TokenizedStructure:
        """Apply a crop result to create new TokenizedStructure."""
        selected = set(result.token_indices)
        new_tokens = []
        for i, token in enumerate(tokenized.tokens):
            if i in selected:
                new_tokens.append(token)

        # Renumber token indices
        for i, token in enumerate(new_tokens):
            token.token_index = i

        return TokenizedStructure(
            tokens=new_tokens,
            pdb_id=tokenized.pdb_id,
            chain_id_to_index=tokenized.chain_id_to_index,
            entity_id_map=tokenized.entity_id_map,
        )

    def crop_to_token_limit(
        self,
        tokenized: TokenizedStructure,
        max_tokens: int,
        rng: Optional[np.random.Generator] = None,
    ) -> TokenizedStructure:
        """Crop and return new tokenized structure.
        
        Args:
            tokenized: Structure to crop
            max_tokens: Maximum tokens in result
            rng: Random number generator
            
        Returns:
            New TokenizedStructure with cropped tokens
        """
        if len(tokenized.tokens) <= max_tokens:
            return tokenized

        if rng is None:
            rng = np.random.default_rng()

        # Create temporary config with specified max
        temp_config = CropConfig(
            max_tokens=max_tokens,
            contiguous_weight=self.config.contiguous_weight,
            spatial_weight=self.config.spatial_weight,
            spatial_interface_weight=self.config.spatial_interface_weight,
        )
        temp_cropper = Cropper(temp_config)
        result = temp_cropper.crop(tokenized, rng)

        # Build new token list
        selected = set(result.token_indices)
        new_tokens = []
        for i, token in enumerate(tokenized.tokens):
            if i in selected:
                new_tokens.append(token)

        # Renumber token indices
        for i, token in enumerate(new_tokens):
            token.token_index = i

        return TokenizedStructure(
            tokens=new_tokens,
            pdb_id=tokenized.pdb_id,
            chain_id_to_index=tokenized.chain_id_to_index,
            entity_id_map=tokenized.entity_id_map,
        )


def check_atom_limit(
    tokens: List[Token],
    selected_indices: np.ndarray,
    max_atoms: int,
) -> np.ndarray:
    """Ensure selected tokens don't exceed atom limit.
    
    From AF3: Both token and atom limits must be respected.
    """
    total_atoms = 0
    valid_indices = []

    for idx in selected_indices:
        token = tokens[idx]
        if total_atoms + token.num_atoms <= max_atoms:
            valid_indices.append(idx)
            total_atoms += token.num_atoms
        else:
            break

    return np.array(valid_indices, dtype=np.int32)
