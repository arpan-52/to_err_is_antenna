"""
Tests for to_err_is_antenna.

Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import yaml

from to_err_is_antenna.config import CorruptionConfig, generate_example_config
from to_err_is_antenna.selectors import SelectionParser, SPWSelection, TimeSelection


class TestCorruptionConfig:
    """Tests for configuration handling."""
    
    def test_from_dict_minimal(self):
        """Test creating config with minimal required fields."""
        config_dict = {
            'input_ms': '/path/to/data.ms',
            'affect': 'phase_error',
            'quantify_error': 10,
        }
        
        config = CorruptionConfig.from_dict(config_dict)
        
        assert config.input_ms == Path('/path/to/data.ms')
        assert config.affect == 'phase_error'
        assert config.quantify_error == 10
        assert config.input_column == 'DATA'  # default
        assert config.output_column == 'CORRECTED_DATA'  # default
    
    def test_from_dict_full(self):
        """Test creating config with all fields."""
        config_dict = {
            'input_ms': '/path/to/data.ms',
            'affect': 'both_error',
            'quantify_error': 15.5,
            'antennas': 'C04,C06',
            'spw': '0:120~150',
            'scan': '1,2,3',
            'time': '30m~40m',
            'input_column': 'DATA',
            'output_column': 'CORRECTED_DATA',
            'chunk_size_mb': 1024,
            'seed': 42,
        }
        
        config = CorruptionConfig.from_dict(config_dict)
        
        assert config.antennas == 'C04,C06'
        assert config.spw == '0:120~150'
        assert config.seed == 42
    
    def test_from_dict_missing_required(self):
        """Test that missing required fields raise error."""
        config_dict = {
            'input_ms': '/path/to/data.ms',
            # missing 'affect' and 'quantify_error'
        }
        
        with pytest.raises(ValueError, match="Missing required"):
            CorruptionConfig.from_dict(config_dict)
    
    def test_from_dict_invalid_affect(self):
        """Test that invalid affect value raises error."""
        config_dict = {
            'input_ms': '/path/to/data.ms',
            'affect': 'invalid_error',
            'quantify_error': 10,
        }
        
        with pytest.raises(ValueError, match="Invalid 'affect'"):
            CorruptionConfig.from_dict(config_dict)
    
    def test_from_dict_invalid_quantify_error(self):
        """Test that out-of-range quantify_error raises error."""
        config_dict = {
            'input_ms': '/path/to/data.ms',
            'affect': 'phase_error',
            'quantify_error': 150,  # > 100
        }
        
        with pytest.raises(ValueError, match="quantify_error"):
            CorruptionConfig.from_dict(config_dict)
    
    def test_from_yaml(self):
        """Test loading config from YAML file."""
        config_content = """
input_ms: /path/to/data.ms
affect: amp_error
quantify_error: 10
antennas: C04&C05
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            config = CorruptionConfig.from_yaml(temp_path)
            assert config.affect == 'amp_error'
            assert config.antennas == 'C04&C05'
        finally:
            Path(temp_path).unlink()
    
    def test_generate_example_config(self):
        """Test generating example config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_config.yaml'
            generate_example_config(output_path)
            
            assert output_path.exists()
            
            with open(output_path) as f:
                content = f.read()
            
            assert 'input_ms' in content
            assert 'affect' in content
            assert 'quantify_error' in content


class TestSelectionParser:
    """Tests for selection parsing."""
    
    def test_parse_antennas_single(self):
        """Test parsing single antenna selection."""
        antenna_names = ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06']
        
        single, pairs = SelectionParser.parse_antennas('C04,C06', antenna_names)
        
        assert single == {4, 6}
        assert pairs == set()
    
    def test_parse_antennas_pair(self):
        """Test parsing antenna pair selection."""
        antenna_names = ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06']
        
        single, pairs = SelectionParser.parse_antennas('C04&C05', antenna_names)
        
        assert single == set()
        assert pairs == {(4, 5)}
    
    def test_parse_antennas_mixed(self):
        """Test parsing mixed antenna selection."""
        antenna_names = ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06']
        
        single, pairs = SelectionParser.parse_antennas('C02,C04&C05', antenna_names)
        
        assert single == {2}
        assert pairs == {(4, 5)}
    
    def test_parse_antennas_unknown(self):
        """Test that unknown antenna raises error."""
        antenna_names = ['C00', 'C01', 'C02']
        
        with pytest.raises(ValueError, match="Unknown antenna"):
            SelectionParser.parse_antennas('C99', antenna_names)
    
    def test_parse_antennas_none(self):
        """Test that None returns empty sets."""
        single, pairs = SelectionParser.parse_antennas(None, ['C00', 'C01'])
        
        assert single == set()
        assert pairs == set()
    
    def test_parse_spw_single(self):
        """Test parsing single SPW."""
        selections = SelectionParser.parse_spw('0')
        
        assert len(selections) == 1
        assert selections[0].spw_id == 0
        assert selections[0].channels is None
    
    def test_parse_spw_with_channels(self):
        """Test parsing SPW with channel range."""
        selections = SelectionParser.parse_spw('0:120~150')
        
        assert len(selections) == 1
        assert selections[0].spw_id == 0
        assert selections[0].channels == (120, 150)
    
    def test_parse_spw_multiple(self):
        """Test parsing multiple SPW selections."""
        selections = SelectionParser.parse_spw('0:120~124,1:200~250')
        
        assert len(selections) == 2
        assert selections[0].spw_id == 0
        assert selections[0].channels == (120, 124)
        assert selections[1].spw_id == 1
        assert selections[1].channels == (200, 250)
    
    def test_parse_spw_none(self):
        """Test that None returns empty list."""
        selections = SelectionParser.parse_spw(None)
        assert selections == []
    
    def test_parse_scan_single(self):
        """Test parsing single scan."""
        scans = SelectionParser.parse_scan('5')
        assert scans == {5}
    
    def test_parse_scan_multiple(self):
        """Test parsing multiple scans."""
        scans = SelectionParser.parse_scan('1,3,5')
        assert scans == {1, 3, 5}
    
    def test_parse_scan_range(self):
        """Test parsing scan range."""
        scans = SelectionParser.parse_scan('1~5')
        assert scans == {1, 2, 3, 4, 5}
    
    def test_parse_scan_mixed(self):
        """Test parsing mixed scan selection."""
        scans = SelectionParser.parse_scan('1,3~5,7')
        assert scans == {1, 3, 4, 5, 7}
    
    def test_parse_scan_none(self):
        """Test that None returns empty set."""
        scans = SelectionParser.parse_scan(None)
        assert scans == set()
    
    def test_parse_time_relative_minutes(self):
        """Test parsing relative time in minutes."""
        selection = SelectionParser.parse_time('30m~40m')
        
        assert selection is not None
        assert selection.is_relative
        assert selection.start_relative_sec == 30 * 60
        assert selection.end_relative_sec == 40 * 60
    
    def test_parse_time_relative_hours(self):
        """Test parsing relative time in hours."""
        selection = SelectionParser.parse_time('0.5h~1.5h')
        
        assert selection is not None
        assert selection.is_relative
        assert selection.start_relative_sec == 0.5 * 3600
        assert selection.end_relative_sec == 1.5 * 3600
    
    def test_parse_time_utc(self):
        """Test parsing UTC time range."""
        selection = SelectionParser.parse_time(
            '2024-01-15T10:30:00~2024-01-15T11:00:00'
        )
        
        assert selection is not None
        assert not selection.is_relative
        assert selection.start_utc.hour == 10
        assert selection.start_utc.minute == 30
        assert selection.end_utc.hour == 11
    
    def test_parse_time_none(self):
        """Test that None returns None."""
        selection = SelectionParser.parse_time(None)
        assert selection is None
    
    def test_baseline_matches_no_selection(self):
        """Test baseline matching with no selection (match all)."""
        assert SelectionParser.baseline_matches(0, 1, set(), set())
        assert SelectionParser.baseline_matches(5, 10, set(), set())
    
    def test_baseline_matches_single_antenna(self):
        """Test baseline matching with single antenna selection."""
        single = {3, 5}
        pairs = set()
        
        # Baseline involving antenna 3 - should match
        assert SelectionParser.baseline_matches(3, 7, single, pairs)
        assert SelectionParser.baseline_matches(1, 3, single, pairs)
        
        # Baseline not involving 3 or 5 - should not match
        assert not SelectionParser.baseline_matches(1, 2, single, pairs)
    
    def test_baseline_matches_pair(self):
        """Test baseline matching with pair selection."""
        single = set()
        pairs = {(3, 5)}
        
        # Exact pair - should match
        assert SelectionParser.baseline_matches(3, 5, single, pairs)
        assert SelectionParser.baseline_matches(5, 3, single, pairs)  # reverse order
        
        # Different pair - should not match
        assert not SelectionParser.baseline_matches(3, 6, single, pairs)


class TestSPWSelection:
    """Tests for SPW selection channel masking."""
    
    def test_channel_mask_all(self):
        """Test channel mask with no channel selection."""
        sel = SPWSelection(spw_id=0, channels=None)
        mask = sel.channel_mask(100)
        
        assert mask.shape == (100,)
        assert np.all(mask)
    
    def test_channel_mask_range(self):
        """Test channel mask with channel range."""
        sel = SPWSelection(spw_id=0, channels=(20, 30))
        mask = sel.channel_mask(100)
        
        assert mask.shape == (100,)
        assert not np.any(mask[:20])  # Before range
        assert np.all(mask[20:31])    # In range (inclusive)
        assert not np.any(mask[31:])  # After range
    
    def test_channel_mask_clamped(self):
        """Test channel mask is clamped to valid range."""
        sel = SPWSelection(spw_id=0, channels=(90, 200))  # 200 > 100
        mask = sel.channel_mask(100)
        
        assert mask.shape == (100,)
        assert np.all(mask[90:])  # Should include 90-99


class TestIntegration:
    """Integration tests (require more setup)."""
    
    def test_config_to_yaml_roundtrip(self):
        """Test that config survives YAML serialization."""
        original = {
            'input_ms': '/path/to/data.ms',
            'affect': 'both_error',
            'quantify_error': 12.5,
            'antennas': 'C04&C05,C06',
            'spw': '0:100~200,1:50~100',
            'scan': '1~3,5',
            'time': '10m~20m',
            'seed': 42,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(original, f)
            temp_path = f.name
        
        try:
            config = CorruptionConfig.from_yaml(temp_path)
            
            assert str(config.input_ms) == '/path/to/data.ms'
            assert config.affect == 'both_error'
            assert config.quantify_error == 12.5
            assert config.antennas == 'C04&C05,C06'
            assert config.seed == 42
        finally:
            Path(temp_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
