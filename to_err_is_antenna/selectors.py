"""
Selection parsing for MS data.

Handles CASA-style selection syntax for antennas, SPW, channels, time, and scans.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Set, Tuple, List, Dict
import numpy as np


@dataclass
class SPWSelection:
    """Represents a spectral window and channel selection."""
    spw_id: int
    channels: Optional[Tuple[int, int]] = None  # (start, end) inclusive, None = all
    
    def channel_mask(self, n_channels: int) -> np.ndarray:
        """Return boolean mask for selected channels."""
        mask = np.zeros(n_channels, dtype=bool)
        if self.channels is None:
            mask[:] = True
        else:
            start, end = self.channels
            # Clamp to valid range
            start = max(0, start)
            end = min(n_channels - 1, end)
            mask[start:end + 1] = True
        return mask


@dataclass  
class TimeSelection:
    """Represents a time selection."""
    # For UTC times
    start_utc: Optional[datetime] = None
    end_utc: Optional[datetime] = None
    
    # For relative times (from scan start)
    start_relative_sec: Optional[float] = None
    end_relative_sec: Optional[float] = None
    
    is_relative: bool = False


class SelectionParser:
    """Parse CASA-style selection strings."""
    
    @staticmethod
    def parse_antennas(antenna_str: Optional[str], 
                       antenna_names: List[str]) -> Tuple[Set[int], Set[Tuple[int, int]]]:
        """
        Parse antenna selection string.
        
        Args:
            antenna_str: Selection like "C04,C06" or "C04&C05" or "C04,C05&C06"
            antenna_names: List of antenna names from MS (index = antenna ID)
            
        Returns:
            (single_antennas, baseline_pairs)
            - single_antennas: Set of antenna IDs where all baselines are affected
            - baseline_pairs: Set of (ant1, ant2) tuples for specific baselines
        """
        if antenna_str is None:
            return set(), set()
        
        # Build name to ID mapping
        name_to_id = {name: idx for idx, name in enumerate(antenna_names)}
        
        single_antennas: Set[int] = set()
        baseline_pairs: Set[Tuple[int, int]] = set()
        
        # Split by comma for multiple selections
        parts = [p.strip() for p in antenna_str.split(',')]
        
        for part in parts:
            if '&' in part:
                # Specific baseline: C04&C05
                ant_names = part.split('&')
                if len(ant_names) != 2:
                    raise ValueError(f"Invalid baseline specification: {part}")
                
                name1, name2 = ant_names[0].strip(), ant_names[1].strip()
                
                if name1 not in name_to_id:
                    raise ValueError(f"Unknown antenna: {name1}")
                if name2 not in name_to_id:
                    raise ValueError(f"Unknown antenna: {name2}")
                
                id1, id2 = name_to_id[name1], name_to_id[name2]
                # Store in canonical order (lower ID first)
                baseline_pairs.add((min(id1, id2), max(id1, id2)))
            else:
                # Single antenna: all baselines involving it
                name = part.strip()
                if name not in name_to_id:
                    raise ValueError(f"Unknown antenna: {name}")
                single_antennas.add(name_to_id[name])
        
        return single_antennas, baseline_pairs
    
    @staticmethod
    def parse_spw(spw_str: Optional[str]) -> List[SPWSelection]:
        """
        Parse SPW selection string.
        
        Args:
            spw_str: Selection like "0:120~150" or "0:120~124,1:124~200"
            
        Returns:
            List of SPWSelection objects
        """
        if spw_str is None:
            return []
        
        selections = []
        parts = [p.strip() for p in spw_str.split(',')]
        
        for part in parts:
            if ':' in part:
                # SPW with channel range: "0:120~150"
                spw_part, chan_part = part.split(':', 1)
                spw_id = int(spw_part)
                
                if '~' in chan_part:
                    chan_start, chan_end = chan_part.split('~')
                    channels = (int(chan_start), int(chan_end))
                else:
                    # Single channel
                    chan = int(chan_part)
                    channels = (chan, chan)
                
                selections.append(SPWSelection(spw_id=spw_id, channels=channels))
            else:
                # Just SPW ID: "0"
                spw_id = int(part)
                selections.append(SPWSelection(spw_id=spw_id, channels=None))
        
        return selections
    
    @staticmethod
    def parse_scan(scan_str: Optional[str]) -> Set[int]:
        """
        Parse scan selection string.
        
        Args:
            scan_str: Selection like "1,2,3" or "1~5"
            
        Returns:
            Set of scan numbers
        """
        if scan_str is None:
            return set()
        
        scans: Set[int] = set()
        parts = [p.strip() for p in scan_str.split(',')]
        
        for part in parts:
            if '~' in part:
                # Range: "1~5"
                start, end = part.split('~')
                scans.update(range(int(start), int(end) + 1))
            else:
                # Single scan
                scans.add(int(part))
        
        return scans
    
    @staticmethod
    def parse_time(time_str: Optional[str], 
                   scan_start_time: Optional[datetime] = None) -> Optional[TimeSelection]:
        """
        Parse time selection string.
        
        Args:
            time_str: Selection like "2024-01-15T10:30:00~2024-01-15T11:00:00" 
                      or "30m~40m" or "0.5h~1h"
            scan_start_time: Start time of the scan (for relative times)
            
        Returns:
            TimeSelection object or None
        """
        if time_str is None:
            return None
        
        # Check for relative time format (e.g., "30m~40m", "0.5h~1h")
        relative_pattern = r'^([\d.]+)([mhs])~([\d.]+)([mhs])$'
        match = re.match(relative_pattern, time_str.strip())
        
        if match:
            # Relative time
            start_val, start_unit = float(match.group(1)), match.group(2)
            end_val, end_unit = float(match.group(3)), match.group(4)
            
            # Convert to seconds
            unit_to_sec = {'s': 1, 'm': 60, 'h': 3600}
            start_sec = start_val * unit_to_sec[start_unit]
            end_sec = end_val * unit_to_sec[end_unit]
            
            return TimeSelection(
                start_relative_sec=start_sec,
                end_relative_sec=end_sec,
                is_relative=True
            )
        
        # Try UTC format
        if '~' in time_str:
            start_str, end_str = time_str.split('~')
            
            # Try various datetime formats
            formats = [
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
            ]
            
            start_dt = None
            end_dt = None
            
            for fmt in formats:
                try:
                    start_dt = datetime.strptime(start_str.strip(), fmt)
                    end_dt = datetime.strptime(end_str.strip(), fmt)
                    break
                except ValueError:
                    continue
            
            if start_dt is None or end_dt is None:
                raise ValueError(f"Could not parse time string: {time_str}")
            
            return TimeSelection(
                start_utc=start_dt,
                end_utc=end_dt,
                is_relative=False
            )
        
        raise ValueError(f"Invalid time format: {time_str}")
    
    @staticmethod
    def baseline_matches(ant1: int, ant2: int,
                        single_antennas: Set[int],
                        baseline_pairs: Set[Tuple[int, int]]) -> bool:
        """
        Check if a baseline matches the antenna selection.
        
        Args:
            ant1, ant2: Antenna IDs for the baseline
            single_antennas: Set of antenna IDs where all baselines are affected
            baseline_pairs: Set of specific (ant1, ant2) baseline pairs
            
        Returns:
            True if this baseline should be corrupted
        """
        # If no selection, match everything
        if not single_antennas and not baseline_pairs:
            return True
        
        # Check single antenna match (any baseline involving these antennas)
        if ant1 in single_antennas or ant2 in single_antennas:
            return True
        
        # Check specific baseline pair
        canonical = (min(ant1, ant2), max(ant1, ant2))
        if canonical in baseline_pairs:
            return True
        
        return False
