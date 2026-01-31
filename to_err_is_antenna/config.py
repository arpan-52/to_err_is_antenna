"""
Configuration handling for visibility corruption.

Parses YAML config files and validates parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Literal
import yaml


@dataclass
class CorruptionConfig:
    """Configuration for visibility corruption."""
    
    # Required
    input_ms: Path
    affect: Literal["phase_error", "amp_error", "both_error"]
    quantify_error: float  # percentage, e.g., 10 means 10%
    
    # Selection parameters (optional - None means all)
    antennas: Optional[str] = None  # e.g., "C04,C06" or "C04&C05"
    spw: Optional[str] = None       # e.g., "0:120~150" or "0:120~124,1:124~200"
    scan: Optional[str] = None      # e.g., "1,2,3" or "1~5"
    time: Optional[str] = None      # UTC, local, or relative like "30m~40m"
    
    # Column handling
    input_column: str = "DATA"
    output_column: str = "CORRECTED_DATA"
    
    # Processing options
    chunk_size_mb: float = 512.0  # RAM limit per chunk in MB
    seed: Optional[int] = None    # Random seed for reproducibility
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "CorruptionConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "CorruptionConfig":
        """Create config from dictionary."""
        
        # Validate required fields
        required = ['input_ms', 'affect', 'quantify_error']
        missing = [k for k in required if k not in config_dict]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
        
        # Validate affect type
        valid_affects = ["phase_error", "amp_error", "both_error"]
        if config_dict['affect'] not in valid_affects:
            raise ValueError(
                f"Invalid 'affect' value: {config_dict['affect']}. "
                f"Must be one of: {valid_affects}"
            )
        
        # Validate quantify_error
        error_pct = float(config_dict['quantify_error'])
        if error_pct < 0 or error_pct > 100:
            raise ValueError(
                f"quantify_error must be between 0 and 100, got: {error_pct}"
            )
        
        # Convert input_ms to Path
        config_dict['input_ms'] = Path(config_dict['input_ms'])
        
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def validate_ms(self) -> None:
        """Validate that the MS exists and is accessible."""
        if not self.input_ms.exists():
            raise FileNotFoundError(f"Measurement Set not found: {self.input_ms}")
        
        # Check it's a directory (MS is a directory)
        if not self.input_ms.is_dir():
            raise ValueError(
                f"Input path is not a directory (MS should be a directory): {self.input_ms}"
            )
        
        # Check for table.dat (casacore table marker)
        table_dat = self.input_ms / "table.dat"
        if not table_dat.exists():
            raise ValueError(
                f"Path does not appear to be a valid Measurement Set: {self.input_ms}"
            )


def generate_example_config(output_path: str | Path = "config.yaml") -> None:
    """Generate an example configuration file."""
    
    example = """\
# to_err_is_antenna Configuration File
# =====================================

# Required: Path to input Measurement Set
input_ms: /path/to/your/data.ms

# Required: Type of error to introduce
# Options: phase_error, amp_error, both_error
affect: phase_error

# Required: Error magnitude as percentage
# For phase_error: adds random phase offset scaled to this % of 360 degrees
# For amp_error: multiplies amplitude by (1 + random * quantify_error/100)
# For both_error: applies both phase and amplitude errors
quantify_error: 10

# Optional: Column selection
# input_column: DATA           # Default: DATA
# output_column: CORRECTED_DATA  # Default: CORRECTED_DATA

# Optional: Selection parameters (leave commented to apply to all)

# Antenna selection:
# - Single antennas: "C04,C06" - corrupts all baselines involving these antennas
# - Antenna pairs: "C04&C05" - corrupts only this specific baseline
# - Mixed: "C04,C05&C06" - C04 all baselines + C05-C06 baseline only
# antennas: C04,C06

# Spectral window and channel selection (CASA syntax):
# - Single SPW: "0"
# - SPW with channels: "0:120~150"
# - Multiple: "0:120~124,1:124~200"
# spw: 0:120~150

# Scan selection:
# - Single: "1"
# - Multiple: "1,2,3"
# - Range: "1~5"
# scan: 1,2,3

# Time selection:
# - UTC: "2024-01-15T10:30:00~2024-01-15T11:00:00"
# - Relative to scan start: "30m~40m" (30th to 40th minute)
# - Relative: "0.5h~1h" (30min to 1hour from start)
# time: 30m~40m

# Processing options
# chunk_size_mb: 512   # RAM limit per chunk in MB
# seed: 42             # Random seed for reproducibility
"""
    
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        f.write(example)
    
    print(f"Example config written to: {output_path}")
