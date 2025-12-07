"""
Moduł do integracji z modelem PLLUM.

Obsługuje dwa tryby:
1. API mode - używa hostowanego modelu przez API organizatora
2. Offline mode - używa lokalnie pobranego modelu

Użycie (API mode):
    from anonymizer.pllum_client import PLLUMClient

    client = PLLUMClient(api_key="TWOJ_KLUCZ")
    response = client.generate("Przerób tekst...")

Użycie (Offline mode):
    client = PLLUMClient(offline=True)
    response = client.generate("Przerób tekst...")
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List


class PLLUMClient:
    """
    Klient do modelu PLLUM.

    Obsługuje zarówno API organizatora jak i tryb offline z lokalnym modelem.
    """

    # Konfiguracja API organizatora
    DEFAULT_BASE_URL = "https://apim-pllum-tst-pcn.azure-api.net/vllm/v1"
    DEFAULT_MODEL_NAME = "CYFRAGOVPL/pllum-12b-nc-chat-250715"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        offline: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 300,
    ):
        """
        Inicjalizacja klienta PLLUM.

        Args:
            api_key: Klucz API (Ocp-Apim-Subscription-Key).
                     Może być ustawiony w .env (PLLUM_API_KEY / API_KEY).
            base_url: URL bazowy API (domyślnie: API organizatora)
            model_name: Nazwa modelu (domyślnie: pllum-12b-nc-chat)
            offline: Czy używać trybu offline (lokalny model)
            temperature: Temperatura generacji (0.0-1.0)
            max_tokens: Maksymalna liczba tokenów odpowiedzi
        """
        self._load_env()
        self.offline = offline
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Konfiguracja
        self.api_key = (
            api_key
            or os.environ.get("PLLUM_API_KEY")
            or os.environ.get("API_KEY")
        )
        self.base_url = (
            base_url
            or os.environ.get("PLLUM_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        self.model_name = (
            model_name
            or os.environ.get("PLLUM_MODEL_NAME")
            or self.DEFAULT_MODEL_NAME
        )

        # Inicjalizacja klienta
        self._llm = None
        self._local_model = None
        self._local_tokenizer = None

        if not offline:
            self._init_api_client()

    @staticmethod
    def _load_env(env_path: str = ".env") -> None:
        """
        Wczytuje zmienne z pliku .env, jeśli istnieje (bez dodatkowych zależności).
        """
        path = Path(env_path)
        if not path.exists():
            return

        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    def _init_api_client(self):
        """Inicjalizuje klienta API (LangChain + OpenAI)."""
        if self._llm is not None:
            return

        if not self.api_key:
            raise ValueError(
                "Brak klucza API! Ustaw api_key lub zmienne PLLUM_API_KEY / API_KEY. "
                "Alternatywnie użyj offline=True dla trybu lokalnego."
            )

        try:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key="EMPTY",  # Wymagane przez LangChain
                openai_api_base=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                default_headers={'Ocp-Apim-Subscription-Key': self.api_key},
            )
        except ImportError:
            raise ImportError(
                "Brak langchain_openai! Zainstaluj: pip install langchain-openai"
            )

    def _init_local_model(self):
        """Inicjalizuje lokalny model (tryb offline)."""
        if self._local_model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            print(f"⏳ Ładowanie modelu {self.model_name}...")

            self._local_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )

            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            print("✅ Model załadowany!")

        except Exception as e:
            raise RuntimeError(
                f"Nie można załadować modelu lokalnie: {e}. "
                f"Uruchom: python download_models.py --pllum"
            )

    def generate(self, prompt: str) -> str:
        """
        Generuje odpowiedź na prompt.

        Args:
            prompt: Tekst promptu

        Returns:
            Wygenerowana odpowiedź
        """
        if self.offline:
            return self._generate_local(prompt)
        else:
            return self._generate_api(prompt)

    def _generate_api(self, prompt: str) -> str:
        """Generuje przez API."""
        self._init_api_client()

        response = self._llm.invoke(prompt)

        # LangChain zwraca obiekt AIMessage
        if hasattr(response, 'content'):
            return response.content
        else:
            # Fallback dla starszych wersji
            return str(response)

    def _generate_local(self, prompt: str) -> str:
        """Generuje lokalnie."""
        self._init_local_model()

        import torch

        inputs = self._local_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._local_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._local_tokenizer.eos_token_id,
            )

        response = self._local_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def anonymize_with_llm(self, text: str) -> str:
        """
        Anonimizuje tekst używając modelu LLM.

        UWAGA: To jest wolniejsze niż RegEx/NER, ale może być
        dokładniejsze dla trudnych przypadków.

        Args:
            text: Tekst do anonimizacji

        Returns:
            Zanonimizowany tekst
        """
        prompt = f"""Jesteś ekspertem od anonimizacji danych osobowych.
Zamień wszystkie dane osobowe w poniższym tekście na odpowiednie tokeny.

Użyj następujących tokenów:
- {{name}} dla imion
- {{surname}} dla nazwisk
- {{pesel}} dla numeru PESEL
- {{email}} dla adresów email
- {{phone}} dla numerów telefonu
- {{city}} dla miast
- {{address}} dla adresów
- {{date}} dla dat
- {{company}} dla nazw firm

Tekst do anonimizacji:
{text}

Zanonimizowany tekst:"""

        response = self.generate(prompt)
        return response.strip()

    def synthesize_with_llm(self, anonymized_text: str) -> str:
        """
        Generuje dane syntetyczne używając LLM.

        Args:
            anonymized_text: Tekst z tokenami anonimizacji

        Returns:
            Tekst z podstawionymi danymi syntetycznymi
        """
        prompt = f"""Zastąp tokeny anonimizacji (np. {{name}}, {{city}}) realistycznymi,
ale fikcyjnymi danymi po polsku. Zachowaj poprawną gramatykę (fleksję).

Tekst z tokenami:
{anonymized_text}

Tekst z danymi syntetycznymi:"""

        response = self.generate(prompt)
        return response.strip()


def download_pllum_model(
    model_name: str = "CYFRAGOVPL/pllum-12b-nc-chat-250715",
) -> bool:
    """
    Pobiera model PLLUM do użytku offline.

    UWAGA: Model jest duży (~24GB), pobieranie może zająć dużo czasu!

    Args:
        model_name: Nazwa modelu na Hugging Face

    Returns:
        True jeśli sukces
    """
    print(f"\n{'='*60}")
    print(f"📦 Pobieranie modelu PLLUM: {model_name}")
    print("⚠️  UWAGA: Ten model jest bardzo duży (~24GB)!")
    print('=' * 60)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("⏳ Pobieranie tokenizera...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer pobrany!")

        print("⏳ Pobieranie modelu (to może potrwać bardzo długo)...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✅ Model PLLUM pobrany pomyślnie!")

        return True

    except Exception as e:
        print(f"❌ Błąd: {e}")
        return False
