"""mmCIF file parser for structure data.

This module provides parsing functionality for mmCIF files as described in
AlphaFold3 supplement Section 2.1:
- Parse _atom_site section for coordinates
- Extract metadata (resolution, release date, method)
- Extract non-coordinate information (bioassembly, bonds, chains)
- Structure cleanup (alternative locations, MSE→MET, waters, arginine)
"""

from __future__ import annotations

import gzip
import logging
import re
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO, Tuple, Union

import numpy as np

from novadb.data.parsers.structure import (
    Atom,
    Bond,
    Chain,
    ChainType,
    Residue,
    Structure,
    CRYSTALLIZATION_AIDS,
)
from novadb.processing.curation.bioassembly import (
    AssemblyDefinition,
    BioassemblyExpander,
    BioassemblyExpanderConfig,
    SymmetryOperation,
)


logger = logging.getLogger(__name__)


@dataclass
class BioassemblyOperation:
    """Represents a bioassembly transformation operation."""
    assembly_id: str
    operator_id: str
    chain_ids: List[str]
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3D translation vector


class MMCIFParser:
    """Parser for mmCIF format files.
    
    Implements parsing as described in AF3 supplement Section 2.1:
    - Parse _atom_site section
    - Extract metadata and non-coordinate information
    - Perform structure cleanup
    - Expand bioassembly
    """

    def __init__(
        self,
        remove_hydrogens: bool = True,
        remove_waters: bool = True,
        convert_mse_to_met: bool = True,
        fix_arginine_naming: bool = True,
        expand_bioassembly: bool = True,
        bioassembly_id: str = "1",
    ):
        """Initialize the parser.
        
        Args:
            remove_hydrogens: Remove hydrogen atoms
            remove_waters: Remove water molecules
            convert_mse_to_met: Convert MSE (selenomethionine) to MET
            fix_arginine_naming: Fix arginine NH1/NH2 naming ambiguity
            expand_bioassembly: Expand to first bioassembly
            bioassembly_id: Which bioassembly to expand (default "1")
        """
        self.remove_hydrogens = remove_hydrogens
        self.remove_waters = remove_waters
        self.convert_mse_to_met = convert_mse_to_met
        self.fix_arginine_naming = fix_arginine_naming
        self.expand_bioassembly = expand_bioassembly
        self.bioassembly_id = bioassembly_id

    def parse(
        self,
        file_or_path: Union[str, Path, TextIO],
    ) -> Structure:
        """Parse an mmCIF file.
        
        Args:
            file_or_path: Path to mmCIF file or file-like object
            
        Returns:
            Parsed Structure object
        """
        # Read the file content
        content = self._read_file(file_or_path)

        # Parse into data blocks
        data = self._parse_mmcif_data(content)

        # Extract metadata
        pdb_id = self._extract_pdb_id(data)
        resolution = self._extract_resolution(data)
        method = self._extract_method(data)
        release_date = self._extract_release_date(data)

        # Parse atom records
        chains = self._parse_atom_site(data)

        # Parse bond information
        bonds = self._parse_bonds(data)

        # Apply structure cleanup
        chains = self._cleanup_structure(chains)

        # Convert MSE to MET if requested
        if self.convert_mse_to_met:
            chains = self._convert_mse_to_met(chains)

        # Fix arginine naming if requested
        if self.fix_arginine_naming:
            chains = self._fix_arginine_naming(chains)

        # Remove waters if requested
        if self.remove_waters:
            chains = self._remove_waters(chains)

        # Remove hydrogens if requested
        if self.remove_hydrogens:
            chains = self._remove_hydrogens(chains)

        # Build structure
        structure = Structure(
            pdb_id=pdb_id,
            chains={chain.chain_id: chain for chain in chains},
            resolution=resolution,
            method=method,
            release_date=release_date,
            bonds=bonds,
        )

        # Expand bioassembly if requested
        if self.expand_bioassembly:
            assembly_def = self._get_assembly_definition(data)
            if assembly_def is not None:
                config = BioassemblyExpanderConfig(
                    assembly_id=self.bioassembly_id,
                )
                expander = BioassemblyExpander(config)
                expanded, _ = expander.expand(structure, assembly_def)
                structure = expanded

        return structure

    def _read_file(self, file_or_path: Union[str, Path, TextIO]) -> str:
        """Read file content from path or file object."""
        if isinstance(file_or_path, (str, Path)):
            path = Path(file_or_path)
            if path.suffix == ".gz":
                with gzip.open(path, "rt") as f:
                    return f.read()
            else:
                with open(path) as f:
                    return f.read()
        else:
            return file_or_path.read()

    def _parse_mmcif_data(self, content: str) -> Dict[str, Any]:
        """Parse mmCIF content into a dictionary of data blocks."""
        data = {}
        current_block = None
        current_loop = None
        loop_headers = []
        loop_data = []

        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                i += 1
                continue

            # Data block header
            if line.startswith("data_"):
                current_block = line[5:]
                data["_entry_id"] = current_block
                i += 1
                continue

            # Loop definition
            if line == "loop_":
                # Save previous loop if any
                if current_loop and loop_headers and loop_data:
                    data[current_loop] = {
                        "headers": loop_headers,
                        "data": loop_data,
                    }

                current_loop = None
                loop_headers = []
                loop_data = []
                i += 1
                continue

            # Loop header (only when we just saw 'loop_' and haven't seen data yet)
            # Loop headers are always single tokens without values
            if line.startswith("_") and current_loop is None and loop_headers == []:
                # Check if this is a key-value pair (has whitespace and value)
                parts = line.split(None, 1)
                if len(parts) == 2:
                    # This is a key-value pair, not a loop header
                    key, value = parts
                    data[key] = self._clean_value(value)
                    i += 1
                    continue
                # Otherwise fall through to check if next line has value

            # Loop header collection (we're inside a loop definition)
            if line.startswith("_") and (current_loop is not None or loop_headers):
                # We're collecting loop headers
                if current_loop is None:
                    # First header defines the loop category
                    dot_parts = line.split(".")
                    current_loop = dot_parts[0]
                loop_headers.append(line)
                i += 1
                continue

            # Check for new loop header start (first _ after loop_)
            if line.startswith("_") and current_loop is None:
                parts = line.split(None, 1)
                if len(parts) == 1:
                    # Single token - this is a loop header
                    dot_parts = line.split(".")
                    current_loop = dot_parts[0]
                    loop_headers.append(line)
                    i += 1
                    continue

            # Loop data or single value
            if current_loop and loop_headers:
                # This is loop data
                values = self._parse_loop_line(line, lines, i, len(loop_headers))
                if values:
                    loop_data.append(values)
                    i += 1
                else:
                    # End of loop, save it
                    if loop_headers and loop_data:
                        data[current_loop] = {
                            "headers": loop_headers,
                            "data": loop_data,
                        }
                    current_loop = None
                    loop_headers = []
                    loop_data = []
                continue

            # Single key-value pair
            if line.startswith("_"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    data[key] = self._clean_value(value)
                elif len(parts) == 1 and i + 1 < len(lines):
                    # Value on next line
                    key = parts[0]
                    i += 1
                    value_line = lines[i].strip()
                    if value_line.startswith(";"):
                        # Multi-line value
                        value_lines = [value_line[1:]]
                        i += 1
                        while i < len(lines) and not lines[i].strip().startswith(";"):
                            value_lines.append(lines[i])
                            i += 1
                        data[key] = "\n".join(value_lines)
                    else:
                        data[key] = self._clean_value(value_line)

            i += 1

        # Save final loop
        if current_loop and loop_headers and loop_data:
            data[current_loop] = {
                "headers": loop_headers,
                "data": loop_data,
            }

        return data

    def _parse_loop_line(
        self,
        line: str,
        lines: List[str],
        start_idx: int,
        num_fields: int,
    ) -> Optional[List[str]]:
        """Parse a line of loop data, handling quoted strings."""
        if line.startswith("_") or line == "loop_" or line.startswith("#"):
            return None

        # Fast path: no quotes -> simple split
        if '"' not in line and "'" not in line:
            values = line.split()
            if len(values) != num_fields:
                return None
            return [self._clean_value(v) for v in values]

        values = []
        current = ""
        in_quotes = False
        quote_char = None

        for char in line + " ":
            if char in "\"'" and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                values.append(current)
                current = ""
            elif char.isspace() and not in_quotes:
                if current:
                    values.append(current)
                    current = ""
            else:
                current += char

        if len(values) != num_fields:
            return None

        return [self._clean_value(v) for v in values]

    def _clean_value(self, value: str) -> str:
        """Clean a value by removing quotes and handling special cases."""
        value = value.strip()
        if value in (".", "?"):
            return ""
        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            return value[1:-1]
        return value

    def _extract_pdb_id(self, data: Dict[str, Any]) -> str:
        """Extract PDB ID from parsed data."""
        return data.get("_entry_id", data.get("_entry.id", "UNKNOWN")).upper()

    def _extract_resolution(self, data: Dict[str, Any]) -> Optional[float]:
        """Extract resolution from parsed data."""
        # Try different possible keys
        for key in [
            "_refine.ls_d_res_high",
            "_em_3d_reconstruction.resolution",
            "_reflns.d_resolution_high",
        ]:
            if key in data:
                try:
                    return float(data[key])
                except (ValueError, TypeError):
                    pass
        return None

    def _extract_method(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract experimental method from parsed data."""
        method = data.get("_exptl.method", "")
        if isinstance(method, str):
            return method.upper()

        # Check loop format
        if "_exptl" in data:
            loop_data = data["_exptl"]
            headers = loop_data.get("headers", [])
            records = loop_data.get("data", [])
            if records:
                for i, header in enumerate(headers):
                    if "method" in header.lower():
                        return records[0][i].upper()
        return None

    def _extract_release_date(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract release date from parsed data."""
        return data.get(
            "_pdbx_database_status.recvd_initial_deposition_date",
            data.get("_database_PDB_rev.date_original", None)
        )

    def _parse_atom_site(self, data: Dict[str, Any]) -> List[Chain]:
        """Parse _atom_site records into Chain objects."""
        atom_site = data.get("_atom_site")
        if not atom_site:
            return []

        headers = atom_site.get("headers", [])
        records = atom_site.get("data", [])

        if not headers or not records:
            return []

        # Map header names to indices
        header_idx = {h.split(".")[-1]: i for i, h in enumerate(headers)}

        # Required fields
        required = ["Cartn_x", "Cartn_y", "Cartn_z", "auth_asym_id"]
        for field in required:
            if field not in header_idx:
                raise ValueError(f"Missing required field: {field}")

        # Build chains
        chains_dict: Dict[str, Chain] = {}
        residues_dict: Dict[Tuple[str, int, str], Residue] = {}

        get_idx = header_idx.get

        for record in records:
            rec_len = len(record)

            # Extract fields with defaults
            def get_field(name: str, default: Any = "") -> Any:
                idx = get_idx(name)
                if idx is not None and idx < rec_len:
                    return record[idx]
                return default

            # Parse atom data without raising exceptions on missing values
            x_raw = get_field("Cartn_x")
            y_raw = get_field("Cartn_y")
            z_raw = get_field("Cartn_z")

            if x_raw == "" or y_raw == "" or z_raw == "":
                continue

            try:
                x = float(x_raw)
                y = float(y_raw)
                z = float(z_raw)
            except (TypeError, ValueError):
                continue

            chain_id = get_field("auth_asym_id")
            res_name = get_field("auth_comp_id", get_field("label_comp_id", "UNK"))
            res_seq = int(get_field("auth_seq_id", get_field("label_seq_id", 0)) or 0)
            ins_code = get_field("pdbx_PDB_ins_code", "")
            atom_name = get_field("auth_atom_id", get_field("label_atom_id", ""))
            element = get_field("type_symbol", "")

            occupancy_raw = get_field("occupancy", 1.0)
            b_factor_raw = get_field("B_iso_or_equiv", 0.0)
            occupancy = float(occupancy_raw or 1.0)
            b_factor = float(b_factor_raw or 0.0)

            alt_loc = get_field("label_alt_id", "")
            serial = int(get_field("id", 0) or 0)

            # Parse charge
            charge_str = get_field("pdbx_formal_charge", "0")
            try:
                charge = int(charge_str) if charge_str else 0
            except (TypeError, ValueError):
                charge = 0

            is_hetero = get_field("group_PDB", "ATOM") == "HETATM"

            # Create atom
            atom = Atom(
                name=atom_name,
                element=element,
                coords=np.array([x, y, z], dtype=np.float32),
                occupancy=occupancy,
                b_factor=b_factor,
                charge=charge,
                is_hetero=is_hetero,
                alt_loc=alt_loc,
                serial=serial,
            )

            # Get or create chain
            if chain_id not in chains_dict:
                chains_dict[chain_id] = Chain(chain_id=chain_id)

            # Get or create residue
            res_key = (chain_id, res_seq, ins_code)
            if res_key not in residues_dict:
                residue = Residue(
                    name=res_name,
                    seq_id=res_seq,
                    insertion_code=ins_code,
                )
                residues_dict[res_key] = residue
                chains_dict[chain_id].residues.append(residue)

            residue = residues_dict[res_key]

            # Handle alternative locations - keep highest occupancy
            if atom_name in residue.atoms:
                existing = residue.atoms[atom_name]
                if atom.occupancy > existing.occupancy:
                    residue.atoms[atom_name] = atom
            else:
                residue.atoms[atom_name] = atom

        # Convert to list and sort residues
        chains = list(chains_dict.values())
        for chain in chains:
            chain.residues.sort(key=lambda r: (r.seq_id, r.insertion_code))
            chain.chain_type = chain._infer_chain_type()

        return chains

    def _parse_bonds(self, data: Dict[str, Any]) -> List[Bond]:
        """Parse covalent bond information from struct_conn."""
        bonds = []

        struct_conn = data.get("_struct_conn")
        if not struct_conn:
            return bonds

        headers = struct_conn.get("headers", [])
        records = struct_conn.get("data", [])

        if not headers or not records:
            return bonds

        header_idx = {h.split(".")[-1]: i for i, h in enumerate(headers)}

        get_idx = header_idx.get

        for record in records:
            rec_len = len(record)

            def get_field(name: str, default: Any = "") -> Any:
                idx = get_idx(name)
                if idx is not None and idx < rec_len:
                    return record[idx]
                return default

            conn_type = get_field("conn_type_id", "")
            if "covale" not in conn_type.lower() and "metalc" not in conn_type.lower():
                continue

            try:
                bond = Bond(
                    chain1_id=get_field("ptnr1_auth_asym_id"),
                    res1_seq_id=int(get_field("ptnr1_auth_seq_id", 0)),
                    atom1_name=get_field("ptnr1_label_atom_id"),
                    chain2_id=get_field("ptnr2_auth_asym_id"),
                    res2_seq_id=int(get_field("ptnr2_auth_seq_id", 0)),
                    atom2_name=get_field("ptnr2_label_atom_id"),
                    bond_order=1,
                )
                bonds.append(bond)
            except (ValueError, TypeError):
                continue

        return bonds

    def _parse_bioassembly(self, data: Dict[str, Any]) -> List[BioassemblyOperation]:
        """Parse bioassembly transformation operations."""
        operations = []

        # Parse assembly generation info
        gen_data = data.get("_pdbx_struct_assembly_gen")
        if not gen_data:
            return operations

        # Parse transformation matrices
        oper_data = data.get("_pdbx_struct_oper_list")
        if not oper_data:
            return operations

        # Build operator dictionary
        oper_headers = oper_data.get("headers", [])
        oper_records = oper_data.get("data", [])
        oper_idx = {h.split(".")[-1]: i for i, h in enumerate(oper_headers)}

        operators = {}
        for record in oper_records:
            def get_field(name: str, default: Any = "") -> Any:
                idx = oper_idx.get(name)
                if idx is not None and idx < len(record):
                    return record[idx]
                return default

            oper_id = get_field("id")

            try:
                rotation = np.array([
                    [float(get_field(f"matrix[1][{i}]", 0)) for i in range(1, 4)],
                    [float(get_field(f"matrix[2][{i}]", 0)) for i in range(1, 4)],
                    [float(get_field(f"matrix[3][{i}]", 0)) for i in range(1, 4)],
                ], dtype=np.float32)

                translation = np.array([
                    float(get_field("vector[1]", 0)),
                    float(get_field("vector[2]", 0)),
                    float(get_field("vector[3]", 0)),
                ], dtype=np.float32)

                operators[oper_id] = (rotation, translation)
            except (ValueError, TypeError):
                continue

        # Parse assembly generation
        gen_headers = gen_data.get("headers", [])
        gen_records = gen_data.get("data", [])
        gen_idx = {h.split(".")[-1]: i for i, h in enumerate(gen_headers)}

        for record in gen_records:
            def get_field(name: str, default: Any = "") -> Any:
                idx = gen_idx.get(name)
                if idx is not None and idx < len(record):
                    return record[idx]
                return default

            assembly_id = get_field("assembly_id")
            if assembly_id != self.bioassembly_id:
                continue

            oper_expression = get_field("oper_expression")
            chain_ids_str = get_field("asym_id_list")

            chain_ids = [c.strip() for c in chain_ids_str.split(",")]

            # Parse operator expression (can be complex like "1,2,3" or "(1-3)")
            oper_ids = self._parse_oper_expression(oper_expression)

            for oper_id in oper_ids:
                if oper_id in operators:
                    rotation, translation = operators[oper_id]
                    operations.append(BioassemblyOperation(
                        assembly_id=assembly_id,
                        operator_id=oper_id,
                        chain_ids=chain_ids,
                        rotation=rotation,
                        translation=translation,
                    ))

        return operations

    def _parse_all_assemblies(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Parse all bioassembly definitions from mmCIF data.
        
        This method extracts complete assembly information including metadata
        that can be used with the bioassembly module for advanced processing.
        
        Args:
            data: Parsed mmCIF data dictionary
            
        Returns:
            Dictionary mapping assembly_id to assembly definition containing:
            - operations: List of (operator_id, rotation, translation, chain_ids) tuples
            - details: Assembly description
            - method: How assembly was determined
            - oligomeric_count: Number of molecules in assembly
            - oligomeric_details: Description of oligomeric state
        """
        assemblies: Dict[str, Dict[str, Any]] = {}
        
        # Parse assembly metadata
        assembly_data = data.get("_pdbx_struct_assembly")
        if assembly_data:
            headers = assembly_data.get("headers", [])
            records = assembly_data.get("data", [])
            idx = {h.split(".")[-1]: i for i, h in enumerate(headers)}
            
            for record in records:
                def get_field(name: str, default: Any = None) -> Any:
                    i = idx.get(name)
                    if i is not None and i < len(record):
                        val = record[i]
                        return val if val not in ("?", ".") else default
                    return default
                
                assembly_id = get_field("id")
                if assembly_id:
                    assemblies[assembly_id] = {
                        "operations": [],
                        "chain_ids": set(),
                        "details": get_field("details"),
                        "method": get_field("method_details"),
                        "oligomeric_count": get_field("oligomeric_count"),
                        "oligomeric_details": get_field("oligomeric_details"),
                    }
        
        # Parse transformation matrices
        oper_data = data.get("_pdbx_struct_oper_list")
        if not oper_data:
            return assemblies
        
        oper_headers = oper_data.get("headers", [])
        oper_records = oper_data.get("data", [])
        oper_idx = {h.split(".")[-1]: i for i, h in enumerate(oper_headers)}
        
        operators: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {}
        for record in oper_records:
            def get_field(name: str, default: Any = "") -> Any:
                i = oper_idx.get(name)
                if i is not None and i < len(record):
                    return record[i]
                return default
            
            oper_id = get_field("id")
            oper_name = get_field("name", "")
            oper_type = get_field("type", "")
            
            try:
                rotation = np.array([
                    [float(get_field(f"matrix[1][{i}]", 0)) for i in range(1, 4)],
                    [float(get_field(f"matrix[2][{i}]", 0)) for i in range(1, 4)],
                    [float(get_field(f"matrix[3][{i}]", 0)) for i in range(1, 4)],
                ], dtype=np.float64)
                
                translation = np.array([
                    float(get_field("vector[1]", 0)),
                    float(get_field("vector[2]", 0)),
                    float(get_field("vector[3]", 0)),
                ], dtype=np.float64)
                
                operators[oper_id] = (rotation, translation, oper_name or oper_type)
            except (ValueError, TypeError):
                continue
        
        # Parse assembly generation records
        gen_data = data.get("_pdbx_struct_assembly_gen")
        if not gen_data:
            return assemblies
        
        gen_headers = gen_data.get("headers", [])
        gen_records = gen_data.get("data", [])
        gen_idx = {h.split(".")[-1]: i for i, h in enumerate(gen_headers)}
        
        for record in gen_records:
            def get_field(name: str, default: Any = "") -> Any:
                i = gen_idx.get(name)
                if i is not None and i < len(record):
                    return record[i]
                return default
            
            assembly_id = get_field("assembly_id")
            if not assembly_id:
                continue
            
            if assembly_id not in assemblies:
                assemblies[assembly_id] = {
                    "operations": [],
                    "chain_ids": set(),
                    "details": None,
                    "method": None,
                    "oligomeric_count": None,
                    "oligomeric_details": None,
                }
            
            oper_expression = get_field("oper_expression")
            chain_ids_str = get_field("asym_id_list")
            chain_ids = [c.strip() for c in chain_ids_str.split(",") if c.strip()]
            
            assemblies[assembly_id]["chain_ids"].update(chain_ids)
            
            # Parse operator expression
            oper_ids = self._parse_oper_expression(oper_expression)
            
            for oper_id in oper_ids:
                if oper_id in operators:
                    rotation, translation, name = operators[oper_id]
                    assemblies[assembly_id]["operations"].append({
                        "operator_id": oper_id,
                        "rotation": rotation,
                        "translation": translation,
                        "chain_ids": chain_ids,
                        "name": name,
                    })
        
        return assemblies

    def get_assembly_definitions(
        self,
        file_or_path: Union[str, Path, TextIO],
    ) -> Dict[str, Dict[str, Any]]:
        """Parse and return all assembly definitions from an mmCIF file.
        
        This is a convenience method for accessing assembly information
        without parsing the full structure.
        
        Args:
            file_or_path: Path to mmCIF file or file-like object
            
        Returns:
            Dictionary mapping assembly_id to assembly definition
        """
        content = self._read_file(file_or_path)
        data = self._parse_mmcif_data(content)
        return self._parse_all_assemblies(data)

    def _get_assembly_definition(
        self,
        data: Dict[str, Any],
    ) -> Optional[AssemblyDefinition]:
        """Return AssemblyDefinition for configured assembly_id if available.

        Falls back to legacy _parse_bioassembly when richer metadata is
        unavailable, preserving prior behavior.
        """
        assemblies = self._parse_all_assemblies(data)
        if assemblies:
            assembly = assemblies.get(self.bioassembly_id)
            if assembly:
                return self._build_definition_from_dict(
                    assembly_id=self.bioassembly_id,
                    assembly=assembly,
                )

        # Fallback to legacy parser
        legacy_ops = self._parse_bioassembly(data)
        if legacy_ops:
            return self._build_definition_from_legacy_ops(legacy_ops)

        return None

    def _build_definition_from_dict(
        self,
        assembly_id: str,
        assembly: Dict[str, Any],
    ) -> AssemblyDefinition:
        operations: List[SymmetryOperation] = []
        for op in assembly.get("operations", []):
            operations.append(SymmetryOperation(
                operator_id=str(op.get("operator_id", "1")),
                rotation=np.array(op.get("rotation", np.eye(3)), dtype=np.float64),
                translation=np.array(op.get("translation", np.zeros(3)), dtype=np.float64),
                name=op.get("name"),
            ))

        chain_ids = set(assembly.get("chain_ids", []))

        return AssemblyDefinition(
            assembly_id=assembly_id,
            operations=operations,
            chain_ids=chain_ids,
            details=assembly.get("details"),
            method=assembly.get("method"),
            oligomeric_count=assembly.get("oligomeric_count"),
        )

    def _build_definition_from_legacy_ops(
        self,
        operations: List[BioassemblyOperation],
    ) -> AssemblyDefinition:
        sym_ops: List[SymmetryOperation] = []
        chain_ids: set[str] = set()

        for op in operations:
            chain_ids.update(op.chain_ids)
            sym_ops.append(SymmetryOperation(
                operator_id=op.operator_id,
                rotation=np.array(op.rotation, dtype=np.float64),
                translation=np.array(op.translation, dtype=np.float64),
            ))

        return AssemblyDefinition(
            assembly_id=self.bioassembly_id,
            operations=sym_ops,
            chain_ids=chain_ids,
        )

    def _parse_oper_expression(self, expression: str) -> List[str]:
        """Parse operator expression like '1,2,3' or '(1-5)' or '(1-3)(4-6)'.
        
        Handles PDB operator expressions including:
        - Simple: "1"
        - List: "1,2,3"
        - Range: "(1-5)"
        - Cartesian product: "(1-3)(4-6)" for combined rotations
        """
        expression = expression.strip()
        
        # Handle Cartesian product notation: (1-3)(4-6)
        if ")(" in expression:
            parts = expression.split(")(")
            parts[0] = parts[0].lstrip("(")
            parts[-1] = parts[-1].rstrip(")")
            
            # Parse each part
            part_ids = [self._parse_simple_oper_expression(p) for p in parts]
            
            # Generate Cartesian product
            result = [""]
            for ids in part_ids:
                new_result = []
                for prefix in result:
                    for op_id in ids:
                        sep = "_" if prefix else ""
                        new_result.append(f"{prefix}{sep}{op_id}")
                result = new_result
            
            return result
        
        return self._parse_simple_oper_expression(expression)
    
    def _parse_simple_oper_expression(self, expression: str) -> List[str]:
        """Parse a simple operator expression (no Cartesian product)."""
        oper_ids = []
        expression = expression.strip("()")

        for part in expression.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part and not part.startswith("-"):
                # Range like "1-5"
                try:
                    start, end = part.split("-", 1)
                    for i in range(int(start), int(end) + 1):
                        oper_ids.append(str(i))
                except ValueError:
                    oper_ids.append(part)
            else:
                oper_ids.append(part)

        return oper_ids

    def _expand_bioassembly(
        self,
        structure: Structure,
        operations: List[BioassemblyOperation],
    ) -> Structure:
        """Expand structure to bioassembly using transformation operations."""
        if not operations:
            return structure

        # Check if only identity operation
        if len(operations) == 1:
            op = operations[0]
            if np.allclose(op.rotation, np.eye(3)) and np.allclose(op.translation, np.zeros(3)):
                return structure

        new_chains: Dict[str, Chain] = {}
        new_bonds: List[Bond] = []

        for op in operations:
            suffix = f"_{op.operator_id}" if op.operator_id != "1" else ""

            for chain_id in op.chain_ids:
                if chain_id not in structure.chains:
                    continue

                old_chain = structure.chains[chain_id]
                new_chain_id = f"{chain_id}{suffix}"

                # Transform coordinates
                new_residues = []
                for res in old_chain.residues:
                    new_atoms = {}
                    for atom_name, atom in res.atoms.items():
                        new_coords = op.rotation @ atom.coords + op.translation
                        new_atoms[atom_name] = Atom(
                            name=atom.name,
                            element=atom.element,
                            coords=new_coords,
                            occupancy=atom.occupancy,
                            b_factor=atom.b_factor,
                            charge=atom.charge,
                            is_hetero=atom.is_hetero,
                            alt_loc=atom.alt_loc,
                            serial=atom.serial,
                        )

                    new_residues.append(Residue(
                        name=res.name,
                        seq_id=res.seq_id,
                        atoms=new_atoms,
                        insertion_code=res.insertion_code,
                        is_standard=res.is_standard,
                    ))

                new_chains[new_chain_id] = Chain(
                    chain_id=new_chain_id,
                    residues=new_residues,
                    entity_id=old_chain.entity_id,
                    chain_type=old_chain.chain_type,
                )

        return Structure(
            pdb_id=structure.pdb_id,
            chains=new_chains if new_chains else structure.chains,
            resolution=structure.resolution,
            method=structure.method,
            release_date=structure.release_date,
            bonds=new_bonds if new_bonds else structure.bonds,
            title=structure.title,
            authors=structure.authors,
        )

    def _cleanup_structure(self, chains: List[Chain]) -> List[Chain]:
        """Apply structure cleanup as per AF3 Section 2.1."""
        # Already handled: alternative locations (in _parse_atom_site)
        return chains

    def _convert_mse_to_met(self, chains: List[Chain]) -> List[Chain]:
        """Convert MSE (selenomethionine) residues to MET.
        
        From AF3 supplement Section 2.1.
        """
        for chain in chains:
            for residue in chain.residues:
                if residue.name == "MSE":
                    residue.name = "MET"
                    # Convert SE to S
                    if "SE" in residue.atoms:
                        atom = residue.atoms.pop("SE")
                        atom.name = "SD"
                        atom.element = "S"
                        residue.atoms["SD"] = atom
        return chains

    def _fix_arginine_naming(self, chains: List[Chain]) -> List[Chain]:
        """Fix arginine NH1/NH2 naming ambiguity.
        
        From AF3 supplement Section 2.1:
        Ensure NH1 is always closer to CD than NH2.
        """
        for chain in chains:
            for residue in chain.residues:
                if residue.name != "ARG":
                    continue

                cd = residue.get_atom("CD")
                nh1 = residue.get_atom("NH1")
                nh2 = residue.get_atom("NH2")

                if cd is None or nh1 is None or nh2 is None:
                    continue

                dist_nh1 = cd.distance_to(nh1)
                dist_nh2 = cd.distance_to(nh2)

                # Swap if NH2 is closer to CD than NH1
                if dist_nh2 < dist_nh1:
                    residue.atoms["NH1"] = nh2
                    residue.atoms["NH2"] = nh1
                    nh2.name = "NH1"
                    nh1.name = "NH2"

        return chains

    def _remove_waters(self, chains: List[Chain]) -> List[Chain]:
        """Remove water molecules from chains."""
        for chain in chains:
            chain.residues = [res for res in chain.residues if res.name != "HOH"]
        return [chain for chain in chains if chain.residues]

    def _remove_hydrogens(self, chains: List[Chain]) -> List[Chain]:
        """Remove hydrogen atoms from all residues."""
        for chain in chains:
            for residue in chain.residues:
                residue.atoms = {
                    name: atom
                    for name, atom in residue.atoms.items()
                    if not atom.is_hydrogen
                }
        return chains


def parse_mmcif(path: Union[str, Path]) -> Structure:
    """Convenience function to parse an mmCIF file with default settings."""
    parser = MMCIFParser()
    return parser.parse(path)


def get_assembly_definitions(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Convenience function to read assembly definitions from an mmCIF file."""
    parser = MMCIFParser(expand_bioassembly=False)
    return parser.get_assembly_definitions(path)
