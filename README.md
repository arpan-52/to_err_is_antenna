# to_err_is_antenna 

> *"To err is human, same with antennas"*

**Developed by Arpan Pal** 

A visibility error simulator for radio interferometry Measurement Sets (MS). Inject controlled phase and/or amplitude errors into your data for visualization and testing.

## Breaking Change

The configuration API is now rule-based only.

- `affect` is no longer supported
- `quantify_error` is no longer supported
- every config must define a non-empty `rules:` list
- every rule must define at least one of `phase_error_deg` or `amp_error_pct`

## Installation

```bash
# From source
git clone https://github.com/yourusername/to_err_is_antenna.git
cd to_err_is_antenna
pip install -e .
```

### Requirements

- Python 3.9+
- python-casacore
- NumPy
- PyYAML
- tqdm

## Quick Start

### 1. Generate a config file

```bash
corruptms --generate-config
```

This creates `config.yaml` with all available options documented.

### 2. Edit the config

```yaml
input_ms: /path/to/your/data.ms
rules:
  - antennas: C04&C05
    phase_error_deg: 10

  - antennas: C06
    amp_error_pct: 5
    spw: "0:120~150"
    scan: "1,2,3"
```

### 3. Run corruption

```bash
corruptms config.yaml
```

## Error Models

The tool introduces deterministic errors by modifying visibility data according to the configured rules.

### Phase Error (`phase_error_deg`)

Adds a fixed phase rotation to each selected visibility:

```
V_corrupted = V_original × exp(i × φ)
```

Where:
- `φ` is a fixed phase offset in degrees from the config

**Example**: With `phase_error_deg: 10`, the selected visibilities get `phase = phase + 10°`.

**Physical interpretation**: Simulates phase errors from atmospheric fluctuations, instrumental delays, or clock offsets.

### Amplitude Error (`amp_error_pct`)

Scales each selected visibility's amplitude by a fixed percentage:

```
V_corrupted = V_original × (1 + ε)
```

Where:
- `ε` is the configured fractional scaling

**Example**: With `amp_error_pct: 10`, amplitudes are scaled by **1.10×**.

**Physical interpretation**: Simulates gain errors from receiver instabilities, pointing errors, or bandpass variations.

### Combined Error

Applies both phase and amplitude errors sequentially when both are present in the same rule:

```
V_corrupted = V_original × (1 + ε) × exp(i × φ)
```

### Important Notes

1. **Per-rule behavior**: Each rule applies its own fixed phase/amplitude settings to its matching visibilities
2. **Rule stacking**: If multiple rules match the same visibility, they are applied in config order
3. **Complete output**: The output column contains ALL data - corrupted visibilities where selection matches, unchanged visibilities elsewhere

## Configuration Reference

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `input_ms` | Path to Measurement Set | `/data/obs.ms` |
| `rules` | List of corruption rules | See below |

### Rule Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `antennas` | Antenna/baseline selection | `C04,C06` or `C04&C05` |
| `spw` | Spectral window + channels | `0:120~150` |
| `scan` | Scan numbers | `1,2,3` or `1~5` |
| `time` | Time range | `30m~40m` or UTC range |
| `phase_error_deg` | Additive phase offset in degrees | `10` |
| `amp_error_pct` | Amplitude scale in percent | `5` |

### Column Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_column` | `DATA` | Source data column |
| `output_column` | `CORRECTED_DATA` | Target column for output data |

### Processing Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size_mb` | `512` | RAM limit per chunk in MB |
| `seed` | None | Random seed stored in config; currently not used by deterministic rule application |

## Selection Syntax

### Antenna Selection

```yaml
# All baselines involving C04 and C06
rules:
  - antennas: C04,C06
    phase_error_deg: 10

# Only the C04-C05 baseline
  - antennas: C04&C05
    phase_error_deg: 10

# Mixed: C04 all baselines + C05-C06 specific baseline
  - antennas: C04,C05&C06
    amp_error_pct: 5
```

### SPW/Channel Selection

```yaml
# All channels in SPW 0
spw: "0"

# Channels 120-150 in SPW 0
spw: "0:120~150"

# Multiple SPWs with different channel ranges
spw: "0:120~124,1:124~200"
```

**Note**: If `spw` is not specified, ALL SPWs and ALL channels are affected.

### Time Selection

```yaml
# Relative to scan start (30th to 40th minute)
time: "30m~40m"

# Relative in hours
time: "0.5h~1h"

# Absolute UTC
time: "2024-01-15T10:30:00~2024-01-15T11:00:00"
```

### Scan Selection

```yaml
# Specific scans
scan: "1,2,3"

# Range
scan: "1~5"

# Mixed
scan: "1,3~5,7"
```

## Data Handling

### Output Column Behavior

The tool writes **complete data** to the output column:

- **Selected visibilities**: Corrupted according to error model
- **Non-selected visibilities**: Copied unchanged from input column

## Python API

```python
from pathlib import Path
from to_err_is_antenna import VisibilityCorruptor, CorruptionConfig, CorruptionRule

# From YAML
config = CorruptionConfig.from_yaml("config.yaml")

# Or build programmatically
config = CorruptionConfig(
    input_ms=Path("/data/obs.ms"),
    rules=[
        CorruptionRule(
            antennas="C04&C05",
            phase_error_deg=10,
        ),
        CorruptionRule(
            antennas="C06",
            amp_error_pct=5,
            spw="0:120~150",
        ),
    ],
    seed=42
)

# Run corruption
corruptor = VisibilityCorruptor(config)
stats = corruptor.run()

print(f"Corrupted {stats['visibilities_corrupted']} of {stats['total_visibilities']} visibilities")
```

### Corrupt specific baseline pair only

```yaml
input_ms: /data/vla_obs.ms
rules:
  - antennas: ea01&ea02
    phase_error_deg: 5
```

### Simulate bandpass error on specific channels

```yaml
input_ms: /data/alma_obs.ms
rules:
  - amp_error_pct: 8
    spw: "0:500~600,1:500~600"
    scan: "1~10"
```

### Time-variable corruption

```yaml
input_ms: /data/long_obs.ms
rules:
  - antennas: C04,C07,C12
    time: "30m~45m"  # Only corrupt 30-45 min after scan start
    phase_error_deg: 10
    amp_error_pct: 10
```

## License

MIT License - see LICENSE file.
