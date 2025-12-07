"""
NER layer for contextual anonymization.

Uses a HerBERT token-classification model and maps entity labels to
anonymization tags (e.g., PER -> {name}, LOC -> {city}).
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)


@dataclass
class NEREntity:
    """Represents a detected entity."""

    text: str
    label: str  # Entity type (PER, LOC, ORG, etc.)
    start: int
    end: int
    tag: str  # Tag such as {name}, {city}, etc.


class NERAnonymizer:
    """
    Contextual anonymization with NLP (HerBERT).

    Defaults to `Nikod3m/hacknation-2025-NASK-herbert-ner-v2`, but a custom
    model path can be provided.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_brackets: bool = False,
        device: Optional[int] = None,
    ):
        """
        Initialize NER anonymizer.

        Args:
            model_path: HF path/name to HerBERT token-class model
            use_brackets: Use [tag] if True, else {tag}
            device: GPU index or -1 (CPU); auto-detect if None
        """
        self.model_path = model_path or "Nikod3m/hacknation-2025-NASK-herbert-ner-v2"
        self.use_brackets = use_brackets
        self.device = (
            device
            if device is not None
            else (0 if torch.cuda.is_available() else -1)
        )

        self._pipeline = None
        self._tokenizer = None
        self._model = None

        # Map NER labels to our tags
        # Legacy labels
        self._label_to_tag = {
            "PER": "name",
            "PERSON": "name",
            "LOC": "city",
            "LOCATION": "city",
            "GPE": "city",
            "ORG": "company",
            "ORGANIZATION": "company",
            "MISC": None,
            # Nowe etykiety z modelu herbert_ner_v2
            "NAME": "name",
            "SURNAME": "surname",
            "CITY": "city",
            "ADDRESS": "address",
            "AGE": "age",
            "COMPANY": "company",
            "DOCUMENT-NUMBER": "document-number",
            "JOB-TITLE": "job-title",
            "SCHOOL-NAME": "school-name",
            "SEX": "sex",
            "RELIGION": "religion",
            "POLITICAL-VIEW": "political-view",
            "ETHNICITY": "ethnicity",
            "SEXUAL-ORIENTATION": "sexual-orientation",
            "HEALTH": "health",
            "RELATIVE": "relative",
            "DATE": "date",
            "DATE-OF-BIRTH": "date-of-birth",
            "USERNAME": "username",
            "SECRET": "secret",
        }

    def _format_tag(self, tag: str) -> str:
        """Format tag according to bracket style."""
        if self.use_brackets:
            return f"[{tag}]"
        return f"{{{tag}}}"

    def _init_pipeline(self):
        """Lazy-load HerBERT pipeline (downloads from HF if not cached)."""
        if self._pipeline is not None:
            return

        try:
            # First try to load locally, if fails - download from HuggingFace
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                )
                self._model = AutoModelForTokenClassification.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                )
            except OSError:
                # Model not in cache - download it
                print(f"\t[NER] Downloading model from HuggingFace: {self.model_path}")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=False,
                )
                self._model = AutoModelForTokenClassification.from_pretrained(
                    self.model_path,
                    local_files_only=False,
                )
            self._pipeline = pipeline(
                "token-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                aggregation_strategy="simple",
                device=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load NER model from '{self.model_path}': {exc}. "
                "Check your internet connection or provide a local model path."
            ) from exc

    def _map_entity_group(self, entity_group: str) -> Optional[str]:
        """Map model label to anonymization tag."""
        if not entity_group:
            return None
        normalized = entity_group.upper()
        tag = self._label_to_tag.get(normalized)
        if tag:
            return tag

        # Try without prefixes (e.g., B-PER/I-PER)
        normalized = normalized.split("-")[-1]
        return self._label_to_tag.get(normalized)

    def extract_entities(self, text: str, debug: bool = False) -> List[NEREntity]:
        """Extract entities from text and map to anonymization tags."""
        if not text:
            return []

        self._init_pipeline()
        predictions = self._pipeline(text)
        if debug:
            print(f"[DEBUG][NER] Surowe predykcje dla: {text!r}")
            for pred in predictions:
                print(f"  - {pred}")

        entities: List[NEREntity] = []
        for pred in predictions:
            tag = self._map_entity_group(pred.get("entity_group"))
            if not tag:
                continue

            entities.append(
                NEREntity(
                    text=pred.get("word", ""),
                    label=pred.get("entity_group", ""),
                    start=int(pred["start"]),
                    end=int(pred["end"]),
                    tag=self._format_tag(tag),
                )
            )
        if debug:
            if entities:
                print("[DEBUG][NER] Zmapowane encje -> tagi:")
                for ent in entities:
                    print(
                        f"  - {ent.label} ({ent.start}:{ent.end}) '{ent.text}' -> {ent.tag}"
                    )
            else:
                print("[DEBUG][NER] Brak encji po mapowaniu.")
        return entities

    def anonymize(self, text: str) -> str:
        """Replace detected NER entities with anonymization tags."""
        entities = sorted(self.extract_entities(text), key=lambda e: e.start)
        if not entities:
            return text

        result = text
        offset = 0

        for ent in entities:
            start = ent.start + offset
            end = ent.end + offset
            result = result[:start] + ent.tag + result[end:]
            offset += len(ent.tag) - (ent.end - ent.start)

        return result


# Singleton dla szybkiego dostępu
_default_anonymizer: Optional[NERAnonymizer] = None


def get_ner_anonymizer() -> NERAnonymizer:
    """Return default NER anonymizer singleton."""
    global _default_anonymizer
    if _default_anonymizer is None:
        _default_anonymizer = NERAnonymizer()
    return _default_anonymizer
