"""
Warstwa NER (Named Entity Recognition) do anonimizacji danych kontekstowych.

Używa modeli NLP do wykrywania:
- Imiona i nazwiska -> {name}, {surname}
- Miasta -> {city}
- Adresy -> {address}
- Nazwy firm -> {company}
- Daty -> {date}
- I inne encje kontekstowe

UWAGA: To jest placeholder dla późniejszej implementacji.
Aktualnie zwraca tekst bez zmian.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NEREntity:
    """Reprezentuje wykrytą encję."""
    text: str
    label: str  # Typ encji (PER, LOC, ORG, etc.)
    start: int
    end: int
    tag: str    # Nasz tag anonimizacji ({name}, {city}, etc.)


class NERAnonymizer:
    """
    Klasa do anonimizacji danych kontekstowych za pomocą modeli NLP.
    
    Obsługuje (po implementacji):
    - Imiona i nazwiska (PER -> {name}, {surname})
    - Miasta i lokalizacje (LOC -> {city}, {address})
    - Organizacje (ORG -> {company})
    - Daty (DATE -> {date})
    
    Domyślnie używa SpaCy, ale można przełączyć na HerBERT dla lepszej
    dokładności na polskim tekście.
    """
    
    def __init__(
        self, 
        model_name: str = "pl_core_news_lg",
        use_herbert: bool = False,
        use_brackets: bool = False,
        offline: bool = True
    ):
        """
        Inicjalizacja anonimizera NER.
        
        Args:
            model_name: Nazwa modelu SpaCy (domyślnie pl_core_news_lg)
            use_herbert: Czy używać modelu HerBERT zamiast SpaCy
            use_brackets: Jeśli True, używa [tag], jeśli False, {tag}
            offline: Wymuś tryb offline (nie pobieraj modeli)
        """
        self.model_name = model_name
        self.use_herbert = use_herbert
        self.use_brackets = use_brackets
        self.offline = offline
        
        self._nlp = None
        self._herbert_model = None
        self._herbert_tokenizer = None
        
        # Mapowanie etykiet NER na nasze tagi
        self._label_to_tag = {
            # SpaCy labels
            "persName": "name",      # Imię
            "persName_surname": "surname",  # Nazwisko
            "geogName": "city",      # Lokalizacja geograficzna
            "placeName": "address",  # Miejsca
            "orgName": "company",    # Organizacje
            "date": "date",          # Daty
            "time": "date",          # Czas (mapujemy na date)
            
            # Alternatywne etykiety (różne modele)
            "PER": "name",
            "LOC": "city",
            "ORG": "company",
            "MISC": "other",
        }
    
    def _format_tag(self, tag: str) -> str:
        """Formatuje tag."""
        if self.use_brackets:
            return f"[{tag}]"
        return f"{{{tag}}}"
    
    def _load_spacy(self):
        """Ładuje model SpaCy."""
        if self._nlp is not None:
            return
        
        try:
            import spacy
            self._nlp = spacy.load(self.model_name)
        except OSError:
            # Model nie jest zainstalowany
            if self.offline:
                raise RuntimeError(
                    f"Model SpaCy '{self.model_name}' nie jest zainstalowany. "
                    f"Uruchom: python download_models.py"
                )
            else:
                import spacy.cli
                spacy.cli.download(self.model_name)
                import spacy
                self._nlp = spacy.load(self.model_name)
    
    def _load_herbert(self):
        """Ładuje model HerBERT."""
        if self._herbert_model is not None:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            
            model_name = "allegro/herbert-base-cased"
            self._herbert_tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                local_files_only=self.offline
            )
            self._herbert_model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                local_files_only=self.offline
            )
        except Exception as e:
            raise RuntimeError(
                f"Nie można załadować modelu HerBERT: {e}. "
                f"Uruchom: python download_models.py"
            )
    
    def anonymize(self, text: str) -> str:
        """
        Anonimizuje tekst zastępując encje NER odpowiednimi tagami.
        
        UWAGA: Aktualnie to jest placeholder - zwraca tekst bez zmian.
        
        Args:
            text: Tekst do anonimizacji
            
        Returns:
            Zanonimizowany tekst
        """
        # TODO: Implementacja NER
        # Na razie zwracamy tekst bez zmian - RegEx layer zajmuje się
        # danymi o stałym formacie, NER będzie dodany później
        return text
    
    def extract_entities(self, text: str) -> List[NEREntity]:
        """
        Wyodrębnia encje z tekstu.
        
        UWAGA: Aktualnie to jest placeholder - zwraca pustą listę.
        
        Args:
            text: Tekst do analizy
            
        Returns:
            Lista wykrytych encji
        """
        # TODO: Implementacja ekstrakcji encji
        return []
    
    def anonymize_with_context(
        self, 
        text: str, 
        context_words: int = 3
    ) -> str:
        """
        Anonimizuje z uwzględnieniem kontekstu.
        
        Pozwala np. odróżnić miasto od adresu:
        - "Jadę do Warszawy" -> {city}
        - "Mieszkam w Warszawie przy ul. Długiej" -> {address}
        
        UWAGA: Aktualnie to jest placeholder.
        
        Args:
            text: Tekst do anonimizacji
            context_words: Liczba słów kontekstu do analizy
            
        Returns:
            Zanonimizowany tekst
        """
        # TODO: Implementacja kontekstowa
        return text


# Singleton dla szybkiego dostępu
_default_anonymizer: Optional[NERAnonymizer] = None


def get_ner_anonymizer() -> NERAnonymizer:
    """Zwraca domyślny anonymizer NER (singleton)."""
    global _default_anonymizer
    if _default_anonymizer is None:
        _default_anonymizer = NERAnonymizer()
    return _default_anonymizer
