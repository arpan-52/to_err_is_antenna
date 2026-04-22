"""
Configuration handling for visibility corruption.

Parses YAML config files and validates parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class CorruptionRule:
    """A single corruption rule with its own selection and error values."""

    antennas: Optional[str] = None  # e.g., "C04,C06" or "C04&C05"
    spw: Optional[str] = None       # e.g., "0:120~150" or "0:120~124,1:124~200"
    scan: Optional[str] = None      # e.g., "1,2,3" or "1~5"
    time: Optional[str] = None      # UTC or relative like "30m~40m"
    phase_error_deg: Optional[float] = None
    amp_error_pct: Optional[float] = None

    @classmethod
    def from_dict(cls, rule_dict: dict[str, Any]) -> "CorruptionRule":
        """Create a rule from a dictionary with validation."""
        if not isinstance(rule_dict, dict):
            raise ValueError("Each rule must be a mapping")
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown_fields = sorted(set(rule_dict) - valid_fields)
        if unknown_fields:
            raise ValueError(f"Unknown rule field(s): {unknown_fields}")
        filtered_dict = {k: v for k, v in rule_dict.items() if k in valid_fields}
        rule = cls(**filtered_dict)
        rule.validate()
        return rule

    def validate(self) -> None:
        """Validate rule contents."""
        if self.phase_error_deg is None and self.amp_error_pct is None:
            raise ValueError(
                "Each rule must define at least one of 'phase_error_deg' or 'amp_error_pct'"
            )

        if self.phase_error_deg is not None:
            self.phase_error_deg = float(self.phase_error_deg)

        if self.amp_error_pct is not None:
            self.amp_error_pct = float(self.amp_error_pct)
            if self.amp_error_pct <= -100:
                raise ValueError(
                    f"amp_error_pct must be greater than -100, got: {self.amp_error_pct}"
                )

    def describe(self) -> str:
        """Return a short human-readable description of the rule."""
        parts: list[str] = []
        if self.antennas:
            parts.append(f"antennas={self.antennas}")
        if self.phase_error_deg is not None:
            parts.append(f"phase={self.phase_error_deg}deg")
        if self.amp_error_pct is not None:
            parts.append(f"amp={self.amp_error_pct}%")
        return ", ".join(parts) if parts else "all visibilities"


@dataclass
class CorruptionConfig:
    """Configuration for visibility corruption."""

    input_ms: Path
    rules: list[CorruptionRule] = field(default_factory=list)

    # Column handling
    input_column: str = "DATA"
    output_column: str = "CORRECTED_DATA"

    # Processing options
    chunk_size_mb: float = 512.0
    seed: Optional[int] = None

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "CorruptionConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "CorruptionConfig":
        """Create config from dictionary."""
        if not isinstance(config_dict, dict):
            raise ValueError("Configuration must be a mapping")

        valid_fields = {
            "input_ms",
            "rules",
            "input_column",
            "output_column",
            "chunk_size_mb",
            "seed",
        }
        unknown_fields = sorted(set(config_dict) - valid_fields)
        if unknown_fields:
            raise ValueError(f"Unknown config field(s): {unknown_fields}")

        if "input_ms" not in config_dict:
            raise ValueError("Missing required config field: 'input_ms'")

        rules = cls._parse_rules(config_dict)
        input_ms = Path(config_dict["input_ms"])

        return cls(
            input_ms=input_ms,
            rules=rules,
            input_column=config_dict.get("input_column", "DATA"),
            output_column=config_dict.get("output_column", "CORRECTED_DATA"),
            chunk_size_mb=float(config_dict.get("chunk_size_mb", 512.0)),
            seed=config_dict.get("seed"),
        )

    @staticmethod
    def _parse_rules(config_dict: dict[str, Any]) -> list[CorruptionRule]:
        """Parse the required rules list."""
        if "rules" not in config_dict:
            raise ValueError("Missing required config field: 'rules'")

        raw_rules = config_dict["rules"]
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError("'rules' must be a non-empty list")

        return [CorruptionRule.from_dict(rule_dict) for rule_dict in raw_rules]

    def validate_ms(self) -> None:
        """Validate that the MS exists and is accessible."""
        if not self.input_ms.exists():
            raise FileNotFoundError(f"Measurement Set not found: {self.input_ms}")

        if not self.input_ms.is_dir():
            raise ValueError(
                f"Input path is not a directory (MS should be a directory): {self.input_ms}"
            )

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

# Required: Corruption rules
# Each rule bundles selection with the exact errors to apply.
rules:
  # Add +10 degrees to the C04-C05 baseline phase
  - antennas: C04&C05
    phase_error_deg: 10

  # Increase amplitude by 5% on every baseline involving C06
  - antennas: C06
    amp_error_pct: 5

  # Apply both errors on one specific baseline and channel range
  - antennas: C07&C08
    spw: "0:120~150"
    phase_error_deg: 3
    amp_error_pct: 2

# Optional: Column selection
# input_column: DATA
# output_column: CORRECTED_DATA

# Optional: Processing options
# chunk_size_mb: 512
# seed: 42
"""

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        f.write(example)

    print(f"Example config written to: {output_path}")
