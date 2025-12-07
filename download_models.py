#!/usr/bin/env python3
"""
Script to download PLLUM model before running in offline mode.

Downloads:
- PLLUM model: CYFRAGOVPL/pllum-12b-nc-chat-250715 (~24GB!)

Usage:
    python download_models.py           # Downloads PLLUM model
    python download_models.py --verify  # Verification only

WARNING: Downloading PLLUM model (~24GB) may take a long time!
For API mode (hosted model) you don't need to download locally.

After running this script, the library can work in offline mode.
"""

import argparse
import sys
from pathlib import Path


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

        # Show model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Parametry: {num_params:,}")

        return True

    except ImportError:
        print("❌ Błąd: Transformers nie jest zainstalowany.")
        print("   Uruchom: pip install transformers torch")
        return False
    except Exception as e:
        print(f"❌ Błąd: {e}")
        return False


def verify_offline_mode() -> bool:
    """Verify if library can work offline."""
    print(f"\n{'='*60}")
    print("🔍 Weryfikacja trybu offline")
    print('=' * 60)

    try:
        # Import library
        from anonymizer import Anonymizer

        # Create anonymizer
        anonymizer = Anonymizer(offline=True)

        # Test on example
        test_text = "Mój PESEL to 90010112345, email: jan@test.pl"
        result = anonymizer.anonymize(test_text)

        print(f"✅ Biblioteka działa poprawnie!")
        print(f"\n   Test:")
        print(f"   Input:  '{test_text}'")
        print(f"   Output: '{result}'")

        return True

    except ImportError as e:
        print(f"❌ Błąd importu: {e}")
        print(
            "   Upewnij się, że biblioteka jest zainstalowana: pip install -e ."
        )
        return False
    except Exception as e:
        print(f"❌ Błąd weryfikacji: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download PLLUM model for anonymizer library"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify offline mode (no download)",
    )
    parser.add_argument(
        "--model",
        default="CYFRAGOVPL/pllum-12b-nc-chat-250715",
        help="PLLUM model name (default: CYFRAGOVPL/pllum-12b-nc-chat-250715)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 anonymizer - Pobieranie modelu PLLUM")
    print("=" * 60)
    print()
    print(
        "Ten skrypt pobiera model PLLUM potrzebny do działania w trybie offline."
    )
    print("UWAGA: Model ma ~24GB, pobieranie może zająć dużo czasu!")
    print()
    print(
        "💡 Jeśli masz klucz API, możesz używać modelu hostowanego bez pobierania:"
    )
    print("   from anonymizer import PLLUMClient")
    print("   client = PLLUMClient(api_key='TWOJ_KLUCZ')")

    success = True

    # If --verify, only verification
    if args.verify:
        success = verify_offline_mode()
        sys.exit(0 if success else 1)

    # Download PLLUM model
    if not download_pllum_model(args.model):
        success = False

    # Verification
    if success:
        verify_offline_mode()

    print()
    print("=" * 60)
    if success:
        print("✅ Gotowe! Teraz możesz pracować offline.")
    else:
        print("⚠️  Model nie został pobrany poprawnie.")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
