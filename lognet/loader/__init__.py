"""
:mod: `lognet.loader` is a package implementing various data loaders for different methods.
"""

from .structured_data_loader import LazyCSVDataset, StructuredDataset

__all__ = ['LazyCSVDataset', 'StructuredDataset']

del structured_data_loader 
