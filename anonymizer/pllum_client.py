"""
PLLUM model integration module.

Supports two modes:
1. API mode - uses hosted model via organizer's API
2. Offline mode - uses locally downloaded model

Usage (API mode):
    from anonymizer.pllum_client import PLLUMClient

    client = PLLUMClient(api_key="YOUR_KEY")
    response = client.generate("Transform text...")

Usage (Offline mode):
    client = PLLUMClient(offline=True)
    response = client.generate("Transform text...")
"""

import os
from pathlib import Path
from typing import Optional


class PLLUMClient:
    """
    Client for PLLUM model.

    Supports both organizer's API and offline mode with local model.
    """

    # Organizer API configuration
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
        Initialize PLLUM client.

        Args:
            api_key: API key (Ocp-Apim-Subscription-Key).
                     Can be set in .env (PLLUM_API_KEY / API_KEY).
            base_url: API base URL (default: organizer's API)
            model_name: Model name (default: pllum-12b-nc-chat)
            offline: Use offline mode (local model)
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum response tokens
        """
        self._load_env()
        self.offline = offline
        self.temperature = temperature
        self.max_tokens = max_tokens

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

        self._llm = None
        self._local_model = None
        self._local_tokenizer = None

        if not offline:
            self._init_api_client()

    @staticmethod
    def _load_env(env_path: str = ".env") -> None:
        """
        Load variables from .env file if it exists (no extra dependencies).
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
        """Initialize API client (LangChain + OpenAI)."""
        if self._llm is not None:
            return

        if not self.api_key:
            raise ValueError(
                "Missing API key! Set api_key or env vars PLLUM_API_KEY / API_KEY. "
                "Alternatively use offline=True for local mode."
            )

        try:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key="EMPTY",
                openai_api_base=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                default_headers={'Ocp-Apim-Subscription-Key': self.api_key},
            )
        except ImportError:
            raise ImportError(
                "Missing langchain_openai! Install: pip install langchain-openai"
            )

    def _init_local_model(self):
        """Initialize local model (offline mode)."""
        if self._local_model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"\tLoading model {self.model_name}...")

            self._local_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )

            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            print("\tModel loaded!")

        except Exception as e:
            raise RuntimeError(
                f"Cannot load model locally: {e}. "
                f"Run: python download_models.py --pllum"
            )

    def generate(self, prompt: str) -> str:
        """
        Generate response to prompt.

        Args:
            prompt: Prompt text

        Returns:
            Generated response
        """
        if self.offline:
            return self._generate_local(prompt)
        else:
            return self._generate_api(prompt)

    def _generate_api(self, prompt: str) -> str:
        """Generate via API."""
        self._init_api_client()

        response = self._llm.invoke(prompt)

        # LangChain returns AIMessage object
        if hasattr(response, 'content'):
            return response.content
        else:
            # Fallback for older versions
            return str(response)

    def _generate_local(self, prompt: str) -> str:
        """Generate locally."""
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
        Anonymize text using LLM model.

        NOTE: Slower than RegEx/NER but may be more accurate for edge cases.

        Args:
            text: Text to anonymize

        Returns:
            Anonymized text
        """
        prompt = f"""You are an expert in personal data anonymization.
Replace all personal data in the text below with appropriate tokens.

Use these tokens:
- {{name}} for first names
- {{surname}} for surnames
- {{pesel}} for PESEL numbers
- {{email}} for email addresses
- {{phone}} for phone numbers
- {{city}} for cities
- {{address}} for addresses
- {{date}} for dates
- {{company}} for company names

Text to anonymize:
{text}

Anonymized text:"""

        response = self.generate(prompt)
        return response.strip()

    def synthesize_with_llm(self, anonymized_text: str) -> str:
        """
        Generate synthetic data using LLM.

        Args:
            anonymized_text: Text with anonymization tokens

        Returns:
            Text with synthetic data substituted
        """
        prompt = f"""Replace anonymization tokens (e.g., {{name}}, {{city}}) with realistic
but fake Polish data. Preserve correct grammar (inflection).

Text with tokens:
{anonymized_text}

Text with synthetic data:"""

        response = self.generate(prompt)
        return response.strip()


def download_pllum_model(
    model_name: str = "CYFRAGOVPL/pllum-12b-nc-chat-250715",
) -> bool:
    """
    Download PLLUM model for offline use.

    WARNING: Model is large (~24GB), download may take a long time!

    Args:
        model_name: Model name on Hugging Face

    Returns:
        True if successful
    """
    print(f"\n{'='*60}")
    print(f"\tDownloading PLLUM model: {model_name}")
    print("\t\tWARNING: This model is very large (~24GB)!")
    print('=' * 60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("\tTokenizer downloaded!")

        print("Downloading model (this may take a very long time)...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("\tPLLUM model downloaded successfully!")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
