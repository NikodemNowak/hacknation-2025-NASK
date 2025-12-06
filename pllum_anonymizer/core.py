"""
Główna klasa Anonymizer - hybrydowe rozwiązanie do anonimizacji tekstu.

Łączy warstwę RegEx (dla danych o stałym formacie) z warstwą NER
(dla danych kontekstowych) w jedną spójną klasę "plug & play".

Użycie:
    from pllum_anonymizer import Anonymizer
    
    model = Anonymizer()
    text = "Nazywam się Jan Kowalski, PESEL 90010112345."
    result = model.anonymize(text)
    # Wynik: "Nazywam się Jan Kowalski, PESEL {pesel}."
    
    # Z generacją danych syntetycznych:
    synthetic = model.synthesize(result)
    # Wynik: "Nazywam się Jan Kowalski, PESEL 85032112345."
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .regex_layer import RegexAnonymizer
from .ner_layer import NERAnonymizer
from .synthetic import SyntheticGenerator


@dataclass
class AnonymizationStats:
    """Statystyki anonimizacji."""
    total_replacements: int
    regex_replacements: int
    ner_replacements: int
    tags_used: Dict[str, int]


class Anonymizer:
    """
    Główna klasa do anonimizacji tekstu.
    
    Łączy dwa podejścia:
    1. RegEx - dla danych o stałym formacie (PESEL, email, telefon, etc.)
    2. NER - dla danych kontekstowych (imiona, miasta, adresy, etc.)
    
    Atrybuty:
        use_regex: Czy używać warstwy RegEx (domyślnie True)
        use_ner: Czy używać warstwy NER (domyślnie True)
        use_brackets: Czy używać nawiasów kwadratowych [tag] zamiast {tag}
        offline: Tryb offline (nie pobiera modeli z internetu)
    """
    
    def __init__(
        self,
        use_regex: bool = True,
        use_ner: bool = True,
        use_brackets: bool = False,
        offline: bool = True,
        ner_model: str = "pl_core_news_lg"
    ):
        """
        Inicjalizacja Anonymizera.
        
        Args:
            use_regex: Czy używać warstwy RegEx (domyślnie True)
            use_ner: Czy używać warstwy NER (domyślnie True)
            use_brackets: Używaj [tag] zamiast {tag}
            offline: Tryb offline (True = nie pobiera modeli)
            ner_model: Nazwa modelu NER do użycia
        """
        self.use_regex = use_regex
        self.use_ner = use_ner
        self.use_brackets = use_brackets
        self.offline = offline
        self.ner_model = ner_model
        
        # Inicjalizacja warstw
        self._regex_layer: Optional[RegexAnonymizer] = None
        self._ner_layer: Optional[NERAnonymizer] = None
        self._synthetic_generator: Optional[SyntheticGenerator] = None
        
        # Lazy loading - warstwy są ładowane przy pierwszym użyciu
        if use_regex:
            self._init_regex_layer()
    
    def _init_regex_layer(self):
        """Inicjalizuje warstwę RegEx."""
        if self._regex_layer is None:
            self._regex_layer = RegexAnonymizer(use_brackets=self.use_brackets)
    
    def _init_ner_layer(self):
        """Inicjalizuje warstwę NER (lazy loading)."""
        if self._ner_layer is None:
            self._ner_layer = NERAnonymizer(
                model_name=self.ner_model,
                use_brackets=self.use_brackets,
                offline=self.offline
            )
    
    def _init_synthetic_generator(self):
        """Inicjalizuje generator danych syntetycznych."""
        if self._synthetic_generator is None:
            self._synthetic_generator = SyntheticGenerator()
    
    def anonymize(self, text: str) -> str:
        """
        Anonimizuje tekst zastępując dane wrażliwe tokenami.
        
        Kolejność przetwarzania:
        1. Warstwa RegEx (szybka, dla danych o stałym formacie)
        2. Warstwa NER (wolniejsza, dla danych kontekstowych)
        
        Args:
            text: Tekst do anonimizacji
            
        Returns:
            Zanonimizowany tekst z tokenami {tag} lub [tag]
        
        Przykład:
            >>> model = Anonymizer()
            >>> model.anonymize("Mój PESEL: 90010112345")
            'Mój PESEL: {pesel}'
        """
        result = text
        
        # 1. Warstwa RegEx
        if self.use_regex and self._regex_layer:
            result = self._regex_layer.anonymize(result)
        
        # 2. Warstwa NER
        if self.use_ner:
            self._init_ner_layer()
            if self._ner_layer:
                result = self._ner_layer.anonymize(result)
        
        return result
    
    def anonymize_batch(self, texts: List[str]) -> List[str]:
        """
        Anonimizuje wiele tekstów (batch processing).
        
        Args:
            texts: Lista tekstów do anonimizacji
            
        Returns:
            Lista zanonimizowanych tekstów
            
        Przykład:
            >>> model = Anonymizer()
            >>> texts = ["PESEL: 90010112345", "Email: jan@test.pl"]
            >>> model.anonymize_batch(texts)
            ['PESEL: {pesel}', 'Email: {email}']
        """
        return [self.anonymize(text) for text in texts]
    
    def synthesize(self, anonymized_text: str) -> str:
        """
        Generuje dane syntetyczne w miejsce tokenów anonimizacji.
        
        Zamienia tokeny ({name}, {city}, etc.) na realistyczne,
        ale fikcyjne dane.
        
        Args:
            anonymized_text: Tekst z tokenami anonimizacji
            
        Returns:
            Tekst z podstawionymi danymi syntetycznymi
            
        Przykład:
            >>> model = Anonymizer()
            >>> model.synthesize("Mieszkam w {city}")
            'Mieszkam w Krakowie'
        """
        self._init_synthetic_generator()
        return self._synthetic_generator.synthesize(anonymized_text)
    
    def synthesize_batch(self, texts: List[str]) -> List[str]:
        """
        Syntetyzuje wiele tekstów.
        
        Args:
            texts: Lista tekstów z tokenami
            
        Returns:
            Lista tekstów z danymi syntetycznymi
        """
        self._init_synthetic_generator()
        return [self._synthetic_generator.synthesize(text) for text in texts]
    
    def process(self, text: str, with_synthesis: bool = False) -> str:
        """
        Przetwarza tekst: anonimizacja + opcjonalna synteza.
        
        Args:
            text: Tekst wejściowy
            with_synthesis: Czy generować dane syntetyczne
            
        Returns:
            Przetworzony tekst
        """
        result = self.anonymize(text)
        if with_synthesis:
            result = self.synthesize(result)
        return result
    
    def process_batch(
        self, 
        texts: List[str], 
        with_synthesis: bool = False
    ) -> List[str]:
        """
        Przetwarza wiele tekstów.
        
        Args:
            texts: Lista tekstów wejściowych
            with_synthesis: Czy generować dane syntetyczne
            
        Returns:
            Lista przetworzonych tekstów
        """
        return [self.process(text, with_synthesis) for text in texts]
    
    def get_supported_tags(self) -> List[str]:
        """
        Zwraca listę obsługiwanych tagów.
        
        Returns:
            Lista nazw tagów (bez nawiasów)
        """
        from .utils import ALL_TAGS
        return sorted(list(ALL_TAGS))
    
    def __repr__(self) -> str:
        return (
            f"Anonymizer(use_regex={self.use_regex}, "
            f"use_ner={self.use_ner}, "
            f"use_brackets={self.use_brackets})"
        )
