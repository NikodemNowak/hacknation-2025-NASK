"""
Core - connects all layers into a single Anonymizer class (RegEx, NER
and optional)

Usage:
    from anonymizer import Anonymizer

    model = Anonymizer()
    text = "Nazywam się Jan Kowalski, PESEL 90010112345."
    result = model.anonymize(text)
    # Wynik: "Nazywam się [name] [surname], PESEL [pesel]."
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .ner_layer import NERAnonymizer
from .pllum_client import PLLUMClient
from .regex_layer import RegexAnonymizer
from .synthetic import SyntheticGenerator
from .utils import ALL_TAGS


@dataclass
class AnonymizationStats:
    total_replacements: int
    regex_replacements: int
    ner_replacements: int
    tags_used: Dict[str, int]


class Anonymizer:
    """
    Main class for text anonymization.
    """

    def __init__(
        self,
        use_regex: bool = True,
        use_ner: bool = True,
        use_brackets: bool = True,
        ner_model_path: Optional[str] = None,
        use_synthetic: bool = False,
        pllum_api_key: Optional[str] = None,
        pllum_base_url: Optional[str] = None,
        pllum_model_name: Optional[str] = None,
        pllum_offline: Optional[bool] = None,
    ):
        """
        Initializer for Anonymizer.

        Args:
            use_regex: Whether to use RegEx layer (default True)
            use_ner: Whether to use NER layer (default True)
            use_brackets: Use [tag] instead of {tag} (default True)
            ner_model_path: Path to HerBERT model (token-classification)
            use_synthetic: Whether to run PLLuM layer at the end
            pllum_api_key: API key for PLLuM (fallback to .env/ENV)
            pllum_base_url: PLLuM API URL (fallback to ENV or default)
            pllum_model_name: PLLuM model name
        """
        self.use_regex = use_regex
        self.use_ner = use_ner
        self.use_brackets = use_brackets
        self.ner_model_path = ner_model_path or os.environ.get(
            "NER_MODEL_PATH"
        )
        self.use_synthetic = use_synthetic
        self.pllum_api_key = pllum_api_key
        self.pllum_base_url = pllum_base_url
        self.pllum_model_name = pllum_model_name
        self.pllum_offline = pllum_offline

        self._regex_layer: Optional[RegexAnonymizer] = None
        self._ner_layer: Optional[NERAnonymizer] = None
        self._synthetic_generator: Optional[SyntheticGenerator] = None
        self._pllum_client: Optional[PLLUMClient] = None

        if use_regex:
            self._init_regex_layer()

    def _init_regex_layer(self):
        if self._regex_layer is None:
            self._regex_layer = RegexAnonymizer(use_brackets=self.use_brackets)

    def _init_ner_layer(self):
        if self._ner_layer is None:
            self._ner_layer = NERAnonymizer(
                model_path=self.ner_model_path,
                use_brackets=self.use_brackets,
            )

    def _init_synthetic_generator(self, use_llm: Optional[bool] = None):
        desired_use_llm = self.use_synthetic if use_llm is None else use_llm
        # If no API key is available, force offline fallback to avoid raising.
        offline = (
            self.pllum_offline
            if self.pllum_offline is not None
            else not (
                self.pllum_api_key
                or os.environ.get("PLLLUM_API_KEY")
                or os.environ.get("PLUM_API_KEY")
                or os.environ.get("PLLUM_API_KEY")
                or os.environ.get("API_KEY")
            )
        )
        if (
            self._synthetic_generator is None
            or self._synthetic_generator.use_llm != desired_use_llm
            or self._synthetic_generator.use_brackets != self.use_brackets
            or self._synthetic_generator.offline != offline
        ):
            self._synthetic_generator = SyntheticGenerator(
                use_llm=desired_use_llm,
                api_key=self.pllum_api_key,
                base_url=self.pllum_base_url,
                model_name=self.pllum_model_name,
                use_brackets=self.use_brackets,
                offline=offline,
            )

    def anonymize(
        self, text: str, with_synthetic: Optional[bool] = None
    ) -> str:
        """
        Anonimizes the input text - swaps sensitive words for tokens.

        Order:
            1. RegEx layer
            2. NER layer
            3. (optional) Synthetic layer (PLLuM)

        Args:
            text: Text to anonymize
            with_synthetic: Override for using synthetic layer

        Returns:
            Anonymized text with tokens

        Example:
            >>> model = Anonymizer()
            >>> model.anonymize("Mój PESEL: 90010112345")
            'Mój PESEL: [pesel]'
        """
        result = text

        # 1. RegEx
        if self.use_regex:
            self._init_regex_layer()
            if self._regex_layer:
                result = self._regex_layer.anonymize(result)

        # 2. NER
        if self.use_ner:
            self._init_ner_layer()
            if self._ner_layer:
                result = self._ner_layer.anonymize(result)

        # 3. Synthetic layer (PLLuM) - only if on
        apply_synthetic = (
            self.use_synthetic if with_synthetic is None else with_synthetic
        )
        if apply_synthetic:
            self._init_synthetic_generator(use_llm=apply_synthetic)
            if self._synthetic_generator:
                result = self._synthetic_generator.synthesize(result)

        # 4. Merge duplicate tags
        result = self._merge_duplicate_tags(result)

        # 5. Optional LLM refinement
        if apply_synthetic and self._ensure_pllum_client():
            result = self._llm_refine(text, result)

        return result

    def _ensure_pllum_client(self) -> bool:
        """
        Creates PLLuM client if not already created.
        """
        if self._pllum_client is not None:
            return True
        try:
            api_key = (
                self.pllum_api_key
                or os.environ.get("PLLLUM_API_KEY")
                or os.environ.get("PLUM_API_KEY")
                or os.environ.get("PLLUM_API_KEY")
                or os.environ.get("API_KEY")
            )
            offline_mode = api_key is None

            self._pllum_client = PLLUMClient(
                api_key=api_key,
                base_url=self.pllum_base_url,
                model_name=self.pllum_model_name,
                offline=offline_mode,
            )
            return True
        except Exception:
            return False
        return False

    def _merge_duplicate_tags(self, text: str) -> str:
        """
        Connects duplicate adjacent tags into a single tag.
        """
        if self.use_brackets:
            # [tag] [tag] -> [tag]
            pattern = r'\[([^\]]+)\](?:\s*\[\1\])+'
            return re.sub(pattern, r'[\1]', text)
        else:
            # {tag} {tag} -> {tag}
            pattern = r'\{([^}]+)\}(?:\s*\{\1\})+'
            return re.sub(pattern, r'{\1}', text)

    def _llm_refine(self, original_text: str, anonymized_text: str) -> str:
        """
        LLM refinement:
        - checks whether all sensitive data have been tagged
        - keeps existing tags and adds missing ones
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
        Batch processing for anonymization.

        Args:
            texts: List of texts to anonymize

        Returns:
            List of anonymized texts

        Przykład:
            >>> model = Anonymizer()
            >>> texts = ["PESEL: 90010112345", "Email: jan@test.pl"]
            >>> model.anonymize_batch(texts)
            ['PESEL: [pesel]', 'Email: [email]']
        """
        return [self.anonymize(text) for text in texts]

    def synthesize(self, anonymized_text: str) -> str:
        """
        Generates synthetic data for anonymized text.

        Changes tokens back to synthetic data (fake but realistically
        looking).

        Args:
            anonymized_text: Text with tokens

        Returns:
            Text with synthetic data

        Przykład:
            >>> model = Anonymizer()
            >>> model.synthesize("Mieszkam w {city}")
            'Mieszkam w Krakowie'
        """
        self._init_synthetic_generator(use_llm=True)
        return self._synthetic_generator.synthesize(anonymized_text)

    def synthesize_batch(self, texts: List[str]) -> List[str]:
        """
        Synthesizes multiple texts.

        Args:
            texts: List of texts with tokens

        Returns:
            List of texts with synthetic data
        """
        self._init_synthetic_generator()
        return [self._synthetic_generator.synthesize(text) for text in texts]

    def process(self, text: str, with_synthesis: bool = False) -> str:
        """
        Processes a single text (anonymization + optional synthesis).

        Args:
            text: Input text
            with_synthesis: Whether to generate synthetic data

        Returns:
            Processed text
        """
        return self.anonymize(text, with_synthetic=with_synthesis)

    def process_batch(
        self, texts: List[str], with_synthesis: bool = False
    ) -> List[str]:
        """
        Processes multiple texts (anonymization + optional synthesis).

        Args:
            texts: List of input texts
            with_synthesis: Whether to generate synthetic data

        Returns:
            List of processed texts
        """
        return [self.process(text, with_synthesis) for text in texts]

    def get_supported_tags(self) -> List[str]:
        """
        Returns the list of supported tags.

        Returns:
            List of supported tags
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
