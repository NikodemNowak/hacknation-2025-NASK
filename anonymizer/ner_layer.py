"""
Warstwa NER (Named Entity Recognition) do anonimizacji danych kontekstowych.

Używa modelu HerBERT (token-classification) z transformers i mapuje
etykiety encji na tagi anonimizacji (np. PER -> {name}, LOC -> {city}).
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


@dataclass
class NEREntity:
    """Reprezentuje wykrytą encję."""

    text: str
    label: str  # Typ encji (PER, LOC, ORG, etc.)
    start: int
    end: int
    tag: str  # Nasz tag anonimizacji ({name}, {city}, etc.)


class NERAnonymizer:
    """
    Klasa do anonimizacji danych kontekstowych za pomocą modeli NLP (HerBERT).

    Domyślnie używa modelu `allegro/herbert-base-cased`, ale można
    przekazać własną ścieżkę do dotrenowanego modelu.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_brackets: bool = False,
        local_files_only: bool = True,
        device: Optional[int] = None,
    ):
        """
        Inicjalizacja anonimizera NER.

        Args:
            model_path: Ścieżka/nazwa modelu HerBERT (token-classification).
                        Jeśli None, użyje `allegro/herbert-base-cased`.
            use_brackets: Jeśli True, używa [tag], jeśli False, {tag}
            local_files_only: Jeśli True, ładuje wyłącznie lokalne pliki HF
            device: Wymuszony index GPU lub -1 (CPU). Domyślnie autodetekcja.
        """
        self.model_path = model_path or "allegro/herbert-base-cased"
        self.use_brackets = use_brackets
        self.local_files_only = local_files_only
        self.device = (
            device if device is not None else (0 if torch.cuda.is_available() else -1)
        )

        self._pipeline = None
        self._tokenizer = None
        self._model = None

        # Mapowanie etykiet NER na nasze tagi
        # Stare etykiety (standardowe NER)
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
        """Formatuje tag zgodnie z preferencją nawiasów."""
        if self.use_brackets:
            return f"[{tag}]"
        return f"{{{tag}}}"

    def _init_pipeline(self):
        """Leniwe ładowanie modelu/pipe'a HerBERT."""
        if self._pipeline is not None:
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
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
                f"Nie można załadować modelu NER z '{self.model_path}': {exc}. "
                "Uruchom: python download_models.py lub wskaż lokalną ścieżkę."
            ) from exc

    def _map_entity_group(self, entity_group: str) -> Optional[str]:
        """Mapuje etykietę modelu na tag anonimizacji."""
        if not entity_group:
            return None
        normalized = entity_group.upper()
        tag = self._label_to_tag.get(normalized)
        if tag:
            return tag

        # Spróbuj bez prefiksów (np. B-PER/I-PER)
        normalized = normalized.split("-")[-1]
        return self._label_to_tag.get(normalized)

    def extract_entities(self, text: str, debug: bool = False) -> List[NEREntity]:
        """
        Wyodrębnia encje z tekstu i mapuje je na tagi anonimizacji.
        """
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
        """
        Anonimizuje tekst zastępując encje NER odpowiednimi tagami.
        """
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
    """Zwraca domyślny anonymizer NER (singleton)."""
    global _default_anonymizer
    if _default_anonymizer is None:
        _default_anonymizer = NERAnonymizer()
    return _default_anonymizer
