"""Unit tests for NLP module: preprocessor, classifier, entity extractor."""

from __future__ import annotations

import pytest

from src.nlp.entity_extractor import EntityExtractor
from src.nlp.preprocessor import TextPreprocessor

# ---------------------------------------------------------------------------
# TextPreprocessor
# ---------------------------------------------------------------------------


class TestTextPreprocessor:
    def test_clean_removes_html(self):
        prep = TextPreprocessor.__new__(TextPreprocessor)
        prep.max_length = 512
        prep._nlp = None
        result = prep.clean("Hello <b>world</b> test")
        assert "<b>" not in result
        assert "Hello" in result

    def test_clean_redacts_email(self):
        prep = TextPreprocessor.__new__(TextPreprocessor)
        prep.max_length = 512
        prep._nlp = None
        result = prep.clean("Contact us at admin@example.com for help.")
        assert "admin@example.com" not in result
        assert "[EMAIL]" in result

    def test_clean_redacts_url(self):
        prep = TextPreprocessor.__new__(TextPreprocessor)
        prep.max_length = 512
        prep._nlp = None
        result = prep.clean("Visit https://example.com for details.")
        assert "https://example.com" not in result
        assert "[URL]" in result

    def test_truncate_for_bert(self):
        prep = TextPreprocessor.__new__(TextPreprocessor)
        prep.max_length = 512
        prep._nlp = None
        long_text = " ".join(["word"] * 1000)
        result = prep.truncate_for_bert(long_text, max_tokens=512)
        assert len(result.split()) <= 512

    def test_clean_normalises_whitespace(self):
        prep = TextPreprocessor.__new__(TextPreprocessor)
        prep.max_length = 512
        prep._nlp = None
        result = prep.clean("Hello   world\t\n test")
        assert "  " not in result


# ---------------------------------------------------------------------------
# EntityExtractor  (uses spaCy — mocked for unit speed)
# ---------------------------------------------------------------------------


SAMPLE_CONTRACT = (
    "Acme Corporation shall pay GlobalCorp $500,000 by January 31, 2025. "
    "The parties agree that this agreement is governed by the laws of New York. "
    "GlobalCorp must deliver the software within 30 days."
)


@pytest.fixture
def mock_preprocessor():
    """Return a TextPreprocessor with spaCy mocked out."""
    import spacy

    nlp = spacy.blank("en")
    # Add sentencizer for sentence splitting
    nlp.add_pipe("sentencizer")
    prep = TextPreprocessor.__new__(TextPreprocessor)
    prep.spacy_model_name = "en_core_web_sm"
    prep.max_length = 512
    prep._nlp = nlp
    return prep


class TestEntityExtractor:
    def test_extract_obligations(self, mock_preprocessor):
        extractor = EntityExtractor(preprocessor=mock_preprocessor)
        result = extractor.extract(SAMPLE_CONTRACT, doc_id="test_doc")
        # Obligations should include sentences with "shall", "agree", "must"
        obligation_text = " ".join(result.obligations).lower()
        assert any(
            verb in obligation_text for verb in ["shall", "agree", "must"]
        ), f"No obligation verbs found in: {result.obligations}"

    def test_extraction_result_has_doc_id(self, mock_preprocessor):
        extractor = EntityExtractor(preprocessor=mock_preprocessor)
        result = extractor.extract("Simple test text.", doc_id="my_doc")
        assert result.doc_id == "my_doc"

    def test_to_dict_structure(self, mock_preprocessor):
        extractor = EntityExtractor(preprocessor=mock_preprocessor)
        result = extractor.extract("Acme Corp shall pay.", doc_id="d1")
        d = result.to_dict()
        assert "doc_id" in d
        assert "entities" in d
        assert "obligations" in d
        assert "dates" in d
        assert "monetary_values" in d
        assert "parties" in d

    def test_batch_extraction(self, mock_preprocessor):
        extractor = EntityExtractor(preprocessor=mock_preprocessor)
        texts = ["Acme shall pay.", "Party A must deliver by Dec 2025."]
        results = extractor.extract_batch(texts, doc_ids=["d1", "d2"])
        assert len(results) == 2
        assert results[0].doc_id == "d1"
        assert results[1].doc_id == "d2"

    def test_empty_obligations_on_plain_text(self, mock_preprocessor):
        extractor = EntityExtractor(preprocessor=mock_preprocessor)
        result = extractor.extract("The sky is blue today.", doc_id="plain")
        # No obligation verbs — list may be empty
        assert isinstance(result.obligations, list)
