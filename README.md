# to_err_is_antenna 📡

> *"To err is human, to corrupt visibilities is antenna"*

A visibility error simulator for radio interferometry Measurement Sets (MS). Inject controlled phase and amplitude errors into your data for testing calibration pipelines.

## Features

- **Controlled error injection**: Phase errors, amplitude errors, or both
- **Flexible selection**: Target specific antennas, baselines, SPWs, channels, scans, and time ranges
- **Complete data output**: Writes ALL visibilities (corrupted + unchanged) to output column
- **Memory efficient**: Chunked processing for large datasets
- **CASA-compatible**: Uses standard CASA selection syntax
- **Progress bar**: Visual feedback for long-running operations

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
affect: phase_error
quantify_error: 10  # 10% error

# Optional selections (omit to apply to ALL data)
antennas: C04,C06        # Corrupt baselines involving these antennas
spw: "0:120~150"         # SPW 0, channels 120-150
scan: "1,2,3"            # Only these scans
```

### 3. Run corruption

```bash
corruptms config.yaml
```

## Error Models

The tool introduces errors by modifying visibility data. The `quantify_error` parameter controls the error magnitude as a percentage.

### Phase Error (`affect: phase_error`)

Adds a random phase rotation to each selected visibility:

```
V_corrupted = V_original × exp(i × φ)
```

Where:
- `φ` is a random phase offset
- `φ` is uniformly distributed in the range `[-π × (quantify_error/100), +π × (quantify_error/100)]`

**Example**: With `quantify_error: 10`, phase offsets range from **-18° to +18°** (i.e., ±10% of 180°).

**Physical interpretation**: Simulates phase errors from atmospheric fluctuations, instrumental delays, or clock offsets.

### Amplitude Error (`affect: amp_error`)

Scales each selected visibility's amplitude by a random factor:

```
V_corrupted = V_original × (1 + ε)
```

Where:
- `ε` is a random scaling factor
- `ε` is uniformly distributed in the range `[-(quantify_error/100), +(quantify_error/100)]`

**Example**: With `quantify_error: 10`, amplitudes are scaled by factors between **0.9× and 1.1×**.

**Physical interpretation**: Simulates gain errors from receiver instabilities, pointing errors, or bandpass variations.

### Combined Error (`affect: both_error`)

Applies both phase and amplitude errors sequentially:

```
V_corrupted = V_original × (1 + ε) × exp(i × φ)
```

Both `ε` and `φ` are independent random values drawn from their respective distributions.

### Important Notes

1. **Random per visibility**: Each visibility gets an independent random error value
2. **Reproducibility**: Set `seed` in config for reproducible results
3. **Complete output**: The output column contains ALL data - corrupted visibilities where selection matches, unchanged visibilities elsewhere

## Configuration Reference

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `input_ms` | Path to Measurement Set | `/data/obs.ms` |
| `affect` | Error type: `phase_error`, `amp_error`, or `both_error` | `phase_error` |
| `quantify_error` | Error magnitude as percentage (0-100) | `10` |

### Selection Parameters (Optional)

Leave these commented out to apply corruption to ALL data.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `antennas` | Antenna/baseline selection | `C04,C06` or `C04&C05` |
| `spw` | Spectral window + channels | `0:120~150` |
| `scan` | Scan numbers | `1,2,3` or `1~5` |
| `time` | Time range | `30m~40m` or UTC range |

### Column Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_column` | `DATA` | Source data column |
| `output_column` | `CORRECTED_DATA` | Target column for output data |

### Processing Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size_mb` | `512` | RAM limit per chunk in MB |
| `seed` | None | Random seed for reproducibility |

## Selection Syntax

### Antenna Selection

```yaml
# All baselines involving C04 and C06
antennas: C04,C06

# Only the C04-C05 baseline
antennas: C04&C05

# Mixed: C04 all baselines + C05-C06 specific baseline
antennas: C04,C05&C06
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

This ensures the output column contains valid, complete data that can be used directly for further processing.

### Example Workflow

```yaml
# Config: corrupt only antenna C04 baselines with 10% phase error
input_ms: /data/observation.ms
affect: phase_error
quantify_error: 10
antennas: C04
input_column: DATA
output_column: CORRECTED_DATA
```

Result in `CORRECTED_DATA`:
- Baselines with C04: Original data + random phase errors (±18°)
- All other baselines: Exact copy of original data

## CLI Usage

```bash
corruptms config.yaml              # Run corruption
corruptms config.yaml -v           # Verbose output  
corruptms config.yaml -q           # Quiet mode (no progress bar)
corruptms --generate-config        # Create example config.yaml
corruptms --version                # Show version
```

All corruption parameters must be specified in the config file.

## Python API

```python
from pathlib import Path
from to_err_is_antenna import VisibilityCorruptor, CorruptionConfig

# From YAML
config = CorruptionConfig.from_yaml("config.yaml")

# Or build programmatically
config = CorruptionConfig(
    input_ms=Path("/data/obs.ms"),
    affect="phase_error",
    quantify_error=10.0,
    antennas="C04,C06",
    spw="0:120~150",
    seed=42
)

# Run corruption
corruptor = VisibilityCorruptor(config)
stats = corruptor.run()

print(f"Corrupted {stats['visibilities_corrupted']} of {stats['total_visibilities']} visibilities")
```

## Memory Management

The tool processes data in chunks to handle large MS files:

```yaml
chunk_size_mb: 512  # Use ~512 MB RAM per chunk
```

For a 100 GB MS with 1024 channels and 4 correlations:
- Each row ≈ 32 KB (complex64)
- 512 MB chunk ≈ 16,000 rows
- Entire MS processed with minimal memory footprint

## Examples

### Corrupt all data with phase errors

```yaml
input_ms: /data/vla_obs.ms
affect: phase_error
quantify_error: 5
# No selections = corrupt everything
```

### Corrupt specific baseline pair only

```yaml
input_ms: /data/vla_obs.ms
affect: phase_error
quantify_error: 5
antennas: ea01&ea02
```

### Simulate bandpass error on specific channels

```yaml
input_ms: /data/alma_obs.ms
affect: amp_error
quantify_error: 8
spw: "0:500~600,1:500~600"
scan: "1~10"
```

### Time-variable corruption

```yaml
input_ms: /data/long_obs.ms
affect: both_error
quantify_error: 10
time: "30m~45m"  # Only corrupt 30-45 min after scan start
antennas: C04,C07,C12
```

### Reproducible corruption

```yaml
input_ms: /data/test.ms
affect: phase_error
quantify_error: 10
seed: 42  # Same seed = same random errors
```

## License

MIT License - see LICENSE file.

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.
