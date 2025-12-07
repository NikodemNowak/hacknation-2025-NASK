"""
Główna klasa Anonymizer - hybrydowe rozwiązanie do anonimizacji tekstu.

Łączy warstwę RegEx (dla danych o stałym formacie), warstwę NER
(dla danych kontekstowych) oraz opcjonalną syntezę PLLuM w jedną
spójną klasę "plug & play".

Użycie:
    from anonymizer import Anonymizer

    model = Anonymizer()
    text = "Nazywam się Jan Kowalski, PESEL 90010112345."
    result = model.anonymize(text)
    # Wynik: "Nazywam się Jan Kowalski, PESEL {pesel}."

    # Z generacją danych syntetycznych:
    synthetic = model.synthesize(result)
    # Wynik: "Nazywam się Jan Kowalski, PESEL 85032112345."
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .ner_layer import NERAnonymizer
from .regex_layer import RegexAnonymizer
from .synthetic import SyntheticGenerator
from .pllum_client import PLLUMClient
from .utils import ALL_TAGS


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

    Łączy trzy podejścia:
    1. RegEx - dla danych o stałym formacie (PESEL, email, telefon, etc.)
    2. NER (HerBERT) - dla danych kontekstowych (imiona, miasta, adresy, etc.)
    3. Synteza PLLuM - opcjonalnie zamienia tagi na realistyczne dane

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
        ner_model_path: Optional[str] = None,
        use_synthetic: bool = False,
        pllum_api_key: Optional[str] = None,
        pllum_base_url: Optional[str] = None,
        pllum_model_name: Optional[str] = None,
    ):
        """
        Inicjalizacja Anonymizera.

        Args:
            use_regex: Czy używać warstwy RegEx (domyślnie True)
            use_ner: Czy używać warstwy NER (domyślnie True)
            use_brackets: Używaj [tag] zamiast {tag}
            offline: Tryb offline (True = nie pobiera modeli z internetu)
            ner_model_path: Ścieżka do modelu HerBERT (token-classification)
            use_synthetic: Czy na końcu uruchamiać warstwę PLLuM
            pllum_api_key: Klucz API do PLLuM (fallback do .env/ENV)
            pllum_base_url: URL API PLLuM (fallback do ENV lub domyślnego)
            pllum_model_name: Nazwa modelu PLLuM
        """
        self.use_regex = use_regex
        self.use_ner = use_ner
        self.use_brackets = use_brackets
        self.offline = offline
        self.ner_model_path = ner_model_path or os.environ.get("NER_MODEL_PATH")
        self.use_synthetic = use_synthetic
        self.pllum_api_key = pllum_api_key
        self.pllum_base_url = pllum_base_url
        self.pllum_model_name = pllum_model_name

        # Inicjalizacja warstw
        self._regex_layer: Optional[RegexAnonymizer] = None
        self._ner_layer: Optional[NERAnonymizer] = None
        self._synthetic_generator: Optional[SyntheticGenerator] = None
        self._pllum_client: Optional[PLLUMClient] = None

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
                model_path=self.ner_model_path,
                use_brackets=self.use_brackets,
                local_files_only=self.offline
            )

    def _init_synthetic_generator(self, use_llm: Optional[bool] = None):
        """Inicjalizuje generator danych syntetycznych."""
        desired_use_llm = self.use_synthetic if use_llm is None else use_llm
        if (
            self._synthetic_generator is None
            or self._synthetic_generator.use_llm != desired_use_llm
        ):
            self._synthetic_generator = SyntheticGenerator(
                use_llm=desired_use_llm,
                api_key=self.pllum_api_key,
                base_url=self.pllum_base_url,
                model_name=self.pllum_model_name,
            )

    def anonymize(self, text: str, with_synthetic: Optional[bool] = None) -> str:
        """
        Anonimizuje tekst zastępując dane wrażliwe tokenami.

        Kolejność przetwarzania:
        1. Warstwa RegEx (szybka, dla danych o stałym formacie)
        2. Warstwa NER (wolniejsza, dla danych kontekstowych)
        3. Warstwa PLLuM (opcjonalnie zamiana tagów na dane syntetyczne)

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
        if self.use_regex:
            self._init_regex_layer()
            if self._regex_layer:
                result = self._regex_layer.anonymize(result)

        # 2. Warstwa NER
        if self.use_ner:
            self._init_ner_layer()
            if self._ner_layer:
                result = self._ner_layer.anonymize(result)

        # 3. Warstwa syntetyczna (PLLuM) - tylko jeśli włączona
        apply_synthetic = (
            self.use_synthetic if with_synthetic is None else with_synthetic
        )
        if apply_synthetic:
            self._init_synthetic_generator(use_llm=apply_synthetic)
            if self._synthetic_generator:
                result = self._synthetic_generator.synthesize(result)

        # 4. Łączenie zduplikowanych sąsiadujących tagów
        result = self._merge_duplicate_tags(result)

        # 5. Opcjonalne domknięcie LLM (walidacja + uzupełnianie tagów)
        if apply_synthetic and self._ensure_pllum_client():
            result = self._llm_refine(text, result)

        return result

    def _ensure_pllum_client(self) -> bool:
        """Leniewe tworzenie klienta PLLuM do walidacji/uzupełniania tagów."""
        if self._pllum_client is not None:
            return True
        try:
            if self.pllum_api_key or os.environ.get("PLLLUM_API_KEY") or os.environ.get("PLUM_API_KEY"):
                self._pllum_client = PLLUMClient(
                    api_key=self.pllum_api_key,
                    base_url=self.pllum_base_url,
                    model_name=self.pllum_model_name,
                )
                return True
        except Exception:
            return False
        return False

    def _merge_duplicate_tags(self, text: str) -> str:
        """
        Łączy zduplikowane sąsiadujące tagi w jeden.
        
        Np. '[name] [name]' -> '[name]'
            '{city} {city} {city}' -> '{city}'
        """
        if self.use_brackets:
            # Pattern dla [tag] [tag] -> [tag]
            pattern = r'\[([^\]]+)\](?:\s*\[\1\])+'
            return re.sub(pattern, r'[\1]', text)
        else:
            # Pattern dla {tag} {tag} -> {tag}
            pattern = r'\{([^}]+)\}(?:\s*\{\1\})+'
            return re.sub(pattern, r'{\1}', text)

    def _llm_refine(self, original_text: str, anonymized_text: str) -> str:
        """
        Używa PLLuM do weryfikacji / domknięcia tagowania:
        - sprawdza czy wszystkie dane wrażliwe są otagowane
        - zachowuje istniejące tagi, dodaje brakujące
        """
        if not self._pllum_client:
            return anonymized_text

        tags_list = ", ".join(sorted(ALL_TAGS))
        prompt = (
            "Jesteś asystentem anonimizacji. "
            "Masz tekst źródłowy i jego zanonimizowaną wersję z tagami. "
            "Zachowaj wszystkie istniejące tagi i dodaj brakujące, "
            "używając wyłącznie tagów: {tags}. "
            "Nie wstawiaj żadnych danych osobowych ani syntetycznych – tylko tagi. "
            "Zwróć sam zanonimizowany tekst.\n\n"
            "TAGI: {tags}\n\n"
            "ORYGINAŁ:\n{orig}\n\n"
            "ZANONIMIZOWANE:\n{anon}\n\n"
            "WYNIK:"
        ).format(tags=tags_list, orig=original_text, anon=anonymized_text)

        try:
            response = self._pllum_client.generate(prompt)
            if response:
                return response.strip()
        except Exception:
            pass
        return anonymized_text

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
        return self.anonymize(text, with_synthetic=with_synthesis)

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
            f"use_brackets={self.use_brackets}, "
            f"use_synthetic={self.use_synthetic})"
        )
