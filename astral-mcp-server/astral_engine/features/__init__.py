"""
Dynamic Feature Generation and CSV Enrichment Module.
Allows users and AI to dynamically specify, compose, and populate celestial features.
"""
from .generator import FeatureGenerator, FeatureSpec
from .csv_processor import enrich_csv_dataset, describe_feature_options

__all__ = [
    "FeatureGenerator",
    "FeatureSpec",
    "enrich_csv_dataset",
    "describe_feature_options",
]
