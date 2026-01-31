"""
Command-line interface for to_err_is_antenna.

Usage:
    corruptms config.yaml
    corruptms --help
    corruptms --generate-config
"""

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import CorruptionConfig, generate_example_config
from .corrupt import VisibilityCorruptor


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog='corruptms',
        description='Introduce controlled errors into Measurement Set visibility data.',
        epilog='Example: corruptms config.yaml'
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress bar'
    )
    
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate an example configuration file'
    )
    
    parser.add_argument(
        '--output-config',
        default='config.yaml',
        help='Output path for generated config (default: config.yaml)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    args = parser.parse_args()
    
    # Handle generate-config mode
    if args.generate_config:
        generate_example_config(args.output_config)
        return 0
    
    # Require config file if not generating
    if args.config is None:
        parser.error("config file is required (or use --generate-config)")
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load config
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1
        
        config = CorruptionConfig.from_yaml(config_path)
        
        # Run corruption
        logger.info(f"to_err_is_antenna v{__version__}")
        logger.info(f"Config: {config_path}")
        
        corruptor = VisibilityCorruptor(config)
        stats = corruptor.run(show_progress=not args.quiet)
        
        # Print summary
        print("\n" + "=" * 50)
        print("CORRUPTION COMPLETE")
        print("=" * 50)
        print(f"Input MS:           {config.input_ms}")
        print(f"Error type:         {config.affect}")
        print(f"Error magnitude:    {config.quantify_error}%")
        print(f"Input column:       {config.input_column}")
        print(f"Output column:      {config.output_column}")
        print("-" * 50)
        print(f"Rows processed:     {stats['rows_processed']:,}")
        print(f"Rows corrupted:     {stats['rows_corrupted']:,}")
        print(f"Vis. corrupted:     {stats['visibilities_corrupted']:,}")
        print(f"Total visibilities: {stats['total_visibilities']:,}")
        corruption_pct = 100 * stats['visibilities_corrupted'] / max(1, stats['total_visibilities'])
        print(f"Corruption rate:    {corruption_pct:.2f}%")
        print("=" * 50)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except ImportError as e:
        logger.error(str(e))
        logger.error("Please install python-casacore: pip install python-casacore")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
