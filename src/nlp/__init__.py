"""NLP package: document classification and entity extraction.

Heavy dependencies (torch, transformers) are imported lazily inside
each class so the package is importable without a GPU/ML environment.
"""

from .entity_extractor import EntityExtractor
from .preprocessor import TextPreprocessor

# DocumentClassifier is NOT imported here — it requires torch + transformers.
# Import it directly when needed: from src.nlp.classifier import DocumentClassifier

__all__ = ["TextPreprocessor", "EntityExtractor"]
