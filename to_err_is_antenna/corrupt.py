"""
Core visibility corruption engine.

Handles reading MS data in chunks, applying errors, and writing back.
Memory-efficient processing for large datasets.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Set, Tuple, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from tqdm import tqdm

from .config import CorruptionConfig
from .selectors import SelectionParser, SPWSelection, TimeSelection

# Try to import casacore
try:
    from casacore.tables import table, taql
    CASACORE_AVAILABLE = True
except ImportError:
    CASACORE_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Information about a processing chunk."""
    start_row: int
    end_row: int  # exclusive
    n_rows: int


class VisibilityCorruptor:
    """
    Main class for corrupting visibility data in Measurement Sets.
    
    Handles:
    - Memory-efficient chunked processing
    - Selection filtering (antenna, SPW, time, scan)
    - Phase and amplitude error injection
    - Progress reporting
    """
    
    # Bytes per complex visibility (complex64 = 8 bytes, complex128 = 16 bytes)
    BYTES_PER_VIS = 8  # complex64
    
    def __init__(self, config: CorruptionConfig):
        """
        Initialize the corruptor.
        
        Args:
            config: CorruptionConfig object with all parameters
        """
        if not CASACORE_AVAILABLE:
            raise ImportError(
                "python-casacore is required but not installed. "
                "Install with: pip install python-casacore"
            )
        
        self.config = config
        self.config.validate_ms()
        
        # Initialize random state
        self.rng = np.random.default_rng(config.seed)
        
        # Will be populated when MS is opened
        self._ms: Optional[table] = None
        self._antenna_names: List[str] = []
        self._n_rows: int = 0
        self._n_channels: int = 0
        self._n_correlations: int = 0
        self._data_shape: Tuple[int, ...] = ()
        
        # Parsed selections
        self._single_antennas: Set[int] = set()
        self._baseline_pairs: Set[Tuple[int, int]] = set()
        self._spw_selections: List[SPWSelection] = []
        self._scan_selection: Set[int] = set()
        self._time_selection: Optional[TimeSelection] = None
        
        # Statistics
        self.stats = {
            'rows_processed': 0,
            'rows_corrupted': 0,
            'visibilities_corrupted': 0,
            'total_visibilities': 0,
        }
    
    def _open_ms(self) -> None:
        """Open the Measurement Set and read metadata."""
        logger.info(f"Opening Measurement Set: {self.config.input_ms}")
        
        self._ms = table(str(self.config.input_ms), readonly=False)
        self._n_rows = self._ms.nrows()
        
        # Get antenna names from ANTENNA subtable
        ant_table = table(str(self.config.input_ms / "ANTENNA"), readonly=True)
        self._antenna_names = list(ant_table.getcol("NAME"))
        ant_table.close()
        
        # Get data shape from first row
        if self._n_rows > 0:
            sample_data = self._ms.getcol(self.config.input_column, startrow=0, nrow=1)
            self._data_shape = sample_data.shape[1:]  # (n_channels, n_correlations)
            self._n_channels = self._data_shape[0]
            self._n_correlations = self._data_shape[1]
        
        logger.info(f"MS has {self._n_rows} rows, {len(self._antenna_names)} antennas")
        logger.info(f"Data shape per row: {self._data_shape}")
    
    def _close_ms(self) -> None:
        """Close the Measurement Set."""
        if self._ms is not None:
            self._ms.close()
            self._ms = None
    
    def _parse_selections(self) -> None:
        """Parse all selection parameters."""
        # Antenna selection
        self._single_antennas, self._baseline_pairs = SelectionParser.parse_antennas(
            self.config.antennas, self._antenna_names
        )
        
        if self._single_antennas or self._baseline_pairs:
            logger.info(f"Antenna selection: single={self._single_antennas}, pairs={self._baseline_pairs}")
        
        # SPW selection
        self._spw_selections = SelectionParser.parse_spw(self.config.spw)
        if self._spw_selections:
            logger.info(f"SPW selection: {self._spw_selections}")
        
        # Scan selection
        self._scan_selection = SelectionParser.parse_scan(self.config.scan)
        if self._scan_selection:
            logger.info(f"Scan selection: {self._scan_selection}")
        
        # Time selection
        self._time_selection = SelectionParser.parse_time(self.config.time)
        if self._time_selection:
            logger.info(f"Time selection: {self._time_selection}")
    
    def _calculate_chunk_size(self) -> int:
        """Calculate optimal chunk size based on RAM limit."""
        # Memory per row = n_channels * n_correlations * bytes_per_complex * 2 (read + write buffer)
        bytes_per_row = self._n_channels * self._n_correlations * self.BYTES_PER_VIS * 2
        
        # Add overhead for other columns (antenna, time, scan, etc.) - estimate 100 bytes per row
        bytes_per_row += 100
        
        # Target memory usage
        target_bytes = self.config.chunk_size_mb * 1024 * 1024
        
        # Calculate chunk size
        chunk_size = max(1, int(target_bytes / bytes_per_row))
        
        # Cap at total rows
        chunk_size = min(chunk_size, self._n_rows)
        
        logger.info(f"Chunk size: {chunk_size} rows (using ~{chunk_size * bytes_per_row / 1024 / 1024:.1f} MB)")
        
        return chunk_size
    
    def _generate_chunks(self, chunk_size: int) -> List[ChunkInfo]:
        """Generate chunk information for processing."""
        chunks = []
        start = 0
        
        while start < self._n_rows:
            end = min(start + chunk_size, self._n_rows)
            chunks.append(ChunkInfo(
                start_row=start,
                end_row=end,
                n_rows=end - start
            ))
            start = end
        
        return chunks
    
    def _apply_phase_error(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply phase error to visibility data.
        
        The error is a random phase offset scaled by quantify_error percentage.
        E.g., 10% means phase offset uniformly distributed in [-36, +36] degrees.
        """
        # Generate random phase errors for masked visibilities
        error_scale = self.config.quantify_error / 100.0 * 2 * np.pi
        phase_errors = self.rng.uniform(-error_scale, error_scale, size=data.shape)
        
        # Apply only to masked elements
        phase_errors = np.where(mask, phase_errors, 0)
        
        # Apply phase rotation: data * exp(i * phase_error)
        corrupted = data * np.exp(1j * phase_errors)
        
        return corrupted
    
    def _apply_amp_error(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply amplitude error to visibility data.
        
        The error multiplies amplitude by (1 + random * quantify_error/100).
        E.g., 10% means amplitude scaled by factor in [0.9, 1.1].
        """
        error_scale = self.config.quantify_error / 100.0
        amp_factors = 1.0 + self.rng.uniform(-error_scale, error_scale, size=data.shape)
        
        # Apply only to masked elements
        amp_factors = np.where(mask, amp_factors, 1.0)
        
        corrupted = data * amp_factors
        
        return corrupted
    
    def _apply_errors(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply configured error type(s) to data."""
        if self.config.affect == "phase_error":
            return self._apply_phase_error(data, mask)
        elif self.config.affect == "amp_error":
            return self._apply_amp_error(data, mask)
        elif self.config.affect == "both_error":
            data = self._apply_phase_error(data, mask)
            data = self._apply_amp_error(data, mask)
            return data
        else:
            raise ValueError(f"Unknown affect type: {self.config.affect}")
    
    def _get_channel_mask(self, spw_id: int) -> np.ndarray:
        """Get channel mask for a given SPW ID.
        
        Returns a boolean mask indicating which channels should be corrupted.
        If no SPW selection is specified, ALL channels in ALL SPWs are corrupted.
        If SPW selection is specified but this spw_id is not in the selection,
        returns all False (no corruption for this SPW).
        """
        mask = np.zeros(self._n_channels, dtype=bool)
        
        if not self._spw_selections:
            # No SPW selection = corrupt all channels in all SPWs
            mask[:] = True
            return mask
        
        # Check if this SPW is in our selection
        spw_found = False
        for sel in self._spw_selections:
            if sel.spw_id == spw_id:
                spw_found = True
                sel_mask = sel.channel_mask(self._n_channels)
                mask |= sel_mask
        
        # If this SPW wasn't in selection, mask stays all False (no corruption)
        return mask
    
    def _check_time_match(self, time_mjd: float, scan_start_mjd: Optional[float] = None) -> bool:
        """Check if a time matches the time selection."""
        if self._time_selection is None:
            return True
        
        if self._time_selection.is_relative:
            if scan_start_mjd is None:
                return True  # Can't check relative without scan start
            
            # Convert MJD seconds to offset
            offset_sec = (time_mjd - scan_start_mjd) * 86400.0
            
            return (self._time_selection.start_relative_sec <= offset_sec <= 
                    self._time_selection.end_relative_sec)
        else:
            # UTC comparison
            # MJD to datetime conversion
            time_dt = self._mjd_to_datetime(time_mjd)
            return (self._time_selection.start_utc <= time_dt <= 
                    self._time_selection.end_utc)
    
    @staticmethod
    def _mjd_to_datetime(mjd: float) -> datetime:
        """Convert Modified Julian Date (in seconds) to datetime."""
        # MJD epoch is 1858-11-17 00:00:00
        mjd_epoch = datetime(1858, 11, 17)
        return mjd_epoch + timedelta(seconds=mjd)
    
    def _ensure_output_column_exists(self) -> None:
        """Ensure the output column exists in the MS."""
        if self.config.output_column not in self._ms.colnames():
            logger.info(f"Creating output column: {self.config.output_column}")
            
            # Get column description from input column
            col_desc = self._ms.getcoldesc(self.config.input_column)
            col_desc['name'] = self.config.output_column
            
            # Add the column
            self._ms.addcols(col_desc)
    
    def _process_chunk(self, chunk: ChunkInfo, scan_start_times: Dict[int, float]) -> int:
        """
        Process a single chunk of data.
        
        IMPORTANT: We always write ALL data back to the output column.
        - Data matching selection criteria: corrupted then written
        - Data NOT matching selection: written unchanged (copied from input)
        
        This ensures the output column contains complete, valid data.
        
        Returns:
            Number of rows with corrupted data in this chunk
        """
        # Read data and metadata
        data = self._ms.getcol(
            self.config.input_column, 
            startrow=chunk.start_row, 
            nrow=chunk.n_rows
        ).astype(np.complex64)
        
        ant1 = self._ms.getcol("ANTENNA1", startrow=chunk.start_row, nrow=chunk.n_rows)
        ant2 = self._ms.getcol("ANTENNA2", startrow=chunk.start_row, nrow=chunk.n_rows)
        times = self._ms.getcol("TIME", startrow=chunk.start_row, nrow=chunk.n_rows)
        scans = self._ms.getcol("SCAN_NUMBER", startrow=chunk.start_row, nrow=chunk.n_rows)
        data_desc_ids = self._ms.getcol("DATA_DESC_ID", startrow=chunk.start_row, nrow=chunk.n_rows)
        
        # Initialize corruption mask (rows x channels x correlations)
        # True = this visibility will be corrupted, False = copied unchanged
        corruption_mask = np.zeros(data.shape, dtype=bool)
        rows_corrupted = 0
        
        for i in range(chunk.n_rows):
            row_ant1 = ant1[i]
            row_ant2 = ant2[i]
            row_scan = scans[i]
            row_time = times[i]
            row_ddid = data_desc_ids[i]
            
            # Check antenna selection
            if not SelectionParser.baseline_matches(
                row_ant1, row_ant2, self._single_antennas, self._baseline_pairs
            ):
                continue
            
            # Check scan selection
            if self._scan_selection and row_scan not in self._scan_selection:
                continue
            
            # Check time selection
            scan_start = scan_start_times.get(row_scan)
            if not self._check_time_match(row_time, scan_start):
                continue
            
            # Get channel mask for this SPW
            # Note: DATA_DESC_ID maps to SPW, for simplicity we use it directly
            # In a real implementation, you'd query the DATA_DESCRIPTION subtable
            channel_mask = self._get_channel_mask(row_ddid)
            
            # If no channels selected for this SPW, skip this row
            if not np.any(channel_mask):
                continue
            
            # Mark this row's selected channels for corruption
            corruption_mask[i, channel_mask, :] = True
            rows_corrupted += 1
        
        # Apply errors ONLY where mask is True
        # Where mask is False, data remains unchanged
        corrupted_data = self._apply_errors(data, corruption_mask)
        
        # Write ALL data to output column (corrupted + unchanged)
        self._ms.putcol(
            self.config.output_column,
            corrupted_data,
            startrow=chunk.start_row,
            nrow=chunk.n_rows
        )
        
        # Update statistics
        self.stats['visibilities_corrupted'] += int(np.sum(corruption_mask))
        self.stats['total_visibilities'] += data.size
        
        return rows_corrupted
    
    def _get_scan_start_times(self) -> Dict[int, float]:
        """Get the start time for each scan."""
        scan_starts: Dict[int, float] = {}
        
        # Query unique scans and their minimum times
        scans = self._ms.getcol("SCAN_NUMBER")
        times = self._ms.getcol("TIME")
        
        for scan in np.unique(scans):
            scan_mask = scans == scan
            scan_starts[int(scan)] = float(times[scan_mask].min())
        
        return scan_starts
    
    def run(self, show_progress: bool = True) -> Dict[str, Any]:
        """
        Run the corruption process.
        
        Args:
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting visibility corruption")
        logger.info(f"Error type: {self.config.affect}, magnitude: {self.config.quantify_error}%")
        
        try:
            self._open_ms()
            self._parse_selections()
            self._ensure_output_column_exists()
            
            # Get scan start times for relative time selection
            scan_start_times = self._get_scan_start_times()
            
            # Calculate chunk size and generate chunks
            chunk_size = self._calculate_chunk_size()
            chunks = self._generate_chunks(chunk_size)
            
            logger.info(f"Processing {self._n_rows} rows in {len(chunks)} chunks")
            
            # Process chunks with progress bar
            iterator = tqdm(chunks, desc="Corrupting", unit="chunk", 
                           disable=not show_progress)
            
            for chunk in iterator:
                rows_corrupted = self._process_chunk(chunk, scan_start_times)
                self.stats['rows_processed'] += chunk.n_rows
                self.stats['rows_corrupted'] += rows_corrupted
                
                # Update progress bar description
                pct = 100 * self.stats['rows_corrupted'] / max(1, self.stats['rows_processed'])
                iterator.set_postfix({
                    'corrupted': f"{pct:.1f}%"
                })
            
            # Flush changes
            self._ms.flush()
            
        finally:
            self._close_ms()
        
        # Log summary
        logger.info("Corruption complete!")
        logger.info(f"Rows processed: {self.stats['rows_processed']}")
        logger.info(f"Rows corrupted: {self.stats['rows_corrupted']}")
        logger.info(f"Visibilities corrupted: {self.stats['visibilities_corrupted']}")
        
        return self.stats


def corrupt_ms(config_path: str | Path, show_progress: bool = True) -> Dict[str, Any]:
    """
    Convenience function to corrupt an MS from a config file.
    
    Args:
        config_path: Path to YAML config file
        show_progress: Whether to show progress bar
        
    Returns:
        Processing statistics
    """
    config = CorruptionConfig.from_yaml(config_path)
    corruptor = VisibilityCorruptor(config)
    return corruptor.run(show_progress=show_progress)
