"""
to_err_is_antenna - Visibility Error Simulator for Measurement Sets

A tool to introduce controlled errors (phase, amplitude, or both) into 
radio interferometry visibility data for testing calibration pipelines.
"""

__version__ = "0.1.0"
__author__ = "to_err_is_antenna contributors"

from .corrupt import VisibilityCorruptor
from .config import CorruptionConfig
from .selectors import SelectionParser

__all__ = ["VisibilityCorruptor", "CorruptionConfig", "SelectionParser"]
