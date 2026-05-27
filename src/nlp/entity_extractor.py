"""
Named-Entity Recognition and custom entity extraction using spaCy.

Extracts key entities from compliance and legal documents:
- Standard NER: ORG, PERSON, DATE, MONEY, GPE, LAW
- Custom rules: contract clauses, policy IDs, obligation verbs

Used to power downstream structured data extraction and
reduce manual document annotation burden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span

from src.logger import get_logger
from .preprocessor import TextPreprocessor

logger = get_logger(__name__)


@dataclass
class ExtractedEntity:
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0  # spaCy NER doesn't expose scores; 1.0 for rule-based


@dataclass
class ExtractionResult:
    doc_id: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    obligations: List[str] = field(default_factory=list)    # "shall", "must" sentences
    dates: List[str] = field(default_factory=list)
    monetary_values: List[str] = field(default_factory=list)
    parties: List[str] = field(default_factory=list)

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "entity_count": self.entity_count,
            "entities": [
                {"text": e.text, "label": e.label}
                for e in self.entities
            ],
            "obligations": self.obligations,
            "dates": self.dates,
            "monetary_values": self.monetary_values,
            "parties": self.parties,
        }


class EntityExtractor:
    """
    spaCy-based entity extractor for compliance documents.

    Supports:
    - Standard NER (18 entity types)
    - Rule-based obligation detection (shall / must / will + VP)
    - Structured extraction (dates, money, parties)

    Usage
    -----
    extractor = EntityExtractor()
    result = extractor.extract("Acme Corp. shall pay $500,000 by Jan 2025.")
    print(result.obligations)  # ["Acme Corp. shall pay $500,000 by Jan 2025."]
    """

    OBLIGATION_VERBS = {"shall", "must", "will", "agree", "agrees", "agreed", "require", "requires"}
    TARGET_ENTITY_TYPES = {"ORG", "PERSON", "DATE", "MONEY", "GPE", "LAW", "PRODUCT", "EVENT"}

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None) -> None:
        self.preprocessor = preprocessor or TextPreprocessor()
        self._matcher: Optional[Matcher] = None

    @property
    def matcher(self) -> Matcher:
        if self._matcher is None:
            self._matcher = Matcher(self.preprocessor.nlp.vocab)
            # Rule: obligation pattern — NOUN/PROPN + obligation_verb
            self._matcher.add(
                "OBLIGATION",
                [
                    [
                        {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"},
                        {"LOWER": {"IN": list(self.OBLIGATION_VERBS)}},
                    ]
                ],
            )
        return self._matcher

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str, doc_id: str = "doc") -> ExtractionResult:
        """Extract entities and obligations from a single document."""
        clean_text = self.preprocessor.clean(text)
        doc: Doc = self.preprocessor.nlp(clean_text)
        result = ExtractionResult(doc_id=doc_id)
        self._extract_ner(doc, result)
        self._extract_obligations(doc, result)
        logger.info(
            "entity_extraction_complete",
            doc_id=doc_id,
            entities=result.entity_count,
            obligations=len(result.obligations),
        )
        return result

    def extract_batch(
        self, texts: List[str], doc_ids: Optional[List[str]] = None
    ) -> List[ExtractionResult]:
        """Batch extraction using spaCy's pipe() for throughput."""
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(texts))]
        cleaned = [self.preprocessor.clean(t) for t in texts]
        results = []
        for doc, doc_id in zip(self.preprocessor.nlp.pipe(cleaned, batch_size=64), doc_ids):
            res = ExtractionResult(doc_id=doc_id)
            self._extract_ner(doc, res)
            self._extract_obligations(doc, res)
            results.append(res)
        logger.info("batch_extraction_complete", total=len(results))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_ner(self, doc: Doc, result: ExtractionResult) -> None:
        for ent in doc.ents:
            if ent.label_ in self.TARGET_ENTITY_TYPES:
                result.entities.append(
                    ExtractedEntity(
                        text=ent.text,
                        label=ent.label_,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                    )
                )
                if ent.label_ == "DATE":
                    result.dates.append(ent.text)
                elif ent.label_ == "MONEY":
                    result.monetary_values.append(ent.text)
                elif ent.label_ in ("ORG", "PERSON"):
                    result.parties.append(ent.text)

    def _extract_obligations(self, doc: Doc, result: ExtractionResult) -> None:
        """Find sentences containing obligation verbs."""
        for sent in doc.sents:
            tokens_lower = {t.lower_ for t in sent}
            if tokens_lower & self.OBLIGATION_VERBS:
                result.obligations.append(sent.text.strip())
