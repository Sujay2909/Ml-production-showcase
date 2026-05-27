"""
Text preprocessing utilities shared by the classifier and entity extractor.

Handles tokenisation, cleaning, and spaCy sentence splitting.
Designed for compliance documents: contracts, policy PDFs, incident reports.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

import spacy
from spacy.language import Language

from src.logger import get_logger
from src.settings import get_settings

logger = get_logger(__name__)

# Compile once
_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


class TextPreprocessor:
    """
    Lightweight text cleaner + spaCy pipeline loader.

    Usage
    -----
    prep = TextPreprocessor()
    clean_text = prep.clean("Hello   World!  <b>tag</b>")
    sentences   = prep.sentencize(long_doc)
    """

    def __init__(self, spacy_model: Optional[str] = None) -> None:
        settings = get_settings()
        self.spacy_model_name = spacy_model or settings.spacy_model
        self.max_length = settings.nlp_max_length
        self._nlp: Optional[Language] = None

    @property
    def nlp(self) -> Language:
        if self._nlp is None:
            logger.info("loading_spacy_model", model=self.spacy_model_name)
            self._nlp = spacy.load(self.spacy_model_name)
        return self._nlp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(
        self,
        text: str,
        remove_emails: bool = True,
        remove_urls: bool = True,
        lowercase: bool = False,
    ) -> str:
        """Strip HTML, normalise whitespace, optionally redact PII tokens."""
        text = unicodedata.normalize("NFKD", text)
        text = _HTML_TAG_RE.sub(" ", text)
        if remove_emails:
            text = _EMAIL_RE.sub("[EMAIL]", text)
        if remove_urls:
            text = _URL_RE.sub("[URL]", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        if lowercase:
            text = text.lower()
        return text

    def sentencize(self, text: str) -> List[str]:
        """Split document into sentences using spaCy's dependency parser."""
        doc = self.nlp(text[: self.max_length * 4])  # avoid very long docs
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Return a list of lemmatized, lower-cased tokens."""
        doc = self.nlp(text[: self.max_length * 4])
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_punct
            and not token.is_space
            and (not remove_stopwords or not token.is_stop)
        ]
        return tokens

    def truncate_for_bert(self, text: str, max_tokens: int = 512) -> str:
        """
        Truncate text to approximately max_tokens sub-words.
        Simple word-count heuristic (1 word ≈ 1.3 sub-words).
        """
        words = text.split()
        limit = int(max_tokens / 1.3)
        return " ".join(words[:limit])
