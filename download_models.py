#!/usr/bin/env python3
"""
Skrypt do pobrania modelu PLLUM przed uruchomieniem w trybie offline.

Pobiera:
- Model PLLUM: CYFRAGOVPL/pllum-12b-nc-chat-250715 (~24GB!)

Użycie:
    python download_models.py           # Pobiera model PLLUM
    python download_models.py --verify  # Tylko weryfikacja

UWAGA: Pobieranie modelu PLLUM (~24GB) może zająć dużo czasu!
Dla trybu API (hostowany model) nie musisz pobierać modelu lokalnie.

Po uruchomieniu tego skryptu biblioteka może działać w trybie offline.
"""

import argparse
import sys
from pathlib import Path


def download_pllum_model(model_name: str = "CYFRAGOVPL/pllum-12b-nc-chat-250715") -> bool:
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
    print('='*60)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("⏳ Pobieranie tokenizera...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer pobrany!")
        
        print("⏳ Pobieranie modelu (to może potrwać bardzo długo)...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✅ Model PLLUM pobrany pomyślnie!")
        
        # Pokaż informacje o modelu
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
    """Weryfikuje czy biblioteka może działać offline."""
    print(f"\n{'='*60}")
    print("🔍 Weryfikacja trybu offline")
    print('='*60)
    
    try:
        # Importuj bibliotekę
        from pllum_anonymizer import Anonymizer
        
        # Stwórz anonymizer
        anonymizer = Anonymizer(offline=True)
        
        # Przetestuj na przykładzie
        test_text = "Mój PESEL to 90010112345, email: jan@test.pl"
        result = anonymizer.anonymize(test_text)
        
        print(f"✅ Biblioteka działa poprawnie!")
        print(f"\n   Test:")
        print(f"   Input:  '{test_text}'")
        print(f"   Output: '{result}'")
        
        return True
        
    except ImportError as e:
        print(f"❌ Błąd importu: {e}")
        print("   Upewnij się, że biblioteka jest zainstalowana: pip install -e .")
        return False
    except Exception as e:
        print(f"❌ Błąd weryfikacji: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pobiera model PLLUM dla biblioteki pllum_anonymizer"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Tylko zweryfikuj tryb offline (bez pobierania)"
    )
    parser.add_argument(
        "--model",
        default="CYFRAGOVPL/pllum-12b-nc-chat-250715",
        help="Nazwa modelu PLLUM (domyślnie: CYFRAGOVPL/pllum-12b-nc-chat-250715)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 pllum_anonymizer - Pobieranie modelu PLLUM")
    print("="*60)
    print()
    print("Ten skrypt pobiera model PLLUM potrzebny do działania w trybie offline.")
    print("UWAGA: Model ma ~24GB, pobieranie może zająć dużo czasu!")
    print()
    print("💡 Jeśli masz klucz API, możesz używać modelu hostowanego bez pobierania:")
    print("   from pllum_anonymizer import PLLUMClient")
    print("   client = PLLUMClient(api_key='TWOJ_KLUCZ')")
    
    success = True
    
    # Jeśli --verify, to tylko weryfikacja
    if args.verify:
        success = verify_offline_mode()
        sys.exit(0 if success else 1)
    
    # Pobierz model PLLUM
    if not download_pllum_model(args.model):
        success = False
    
    # Weryfikacja
    if success:
        verify_offline_mode()
    
    print()
    print("="*60)
    if success:
        print("✅ Gotowe! Teraz możesz pracować offline.")
    else:
        print("⚠️  Model nie został pobrany poprawnie.")
    print("="*60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()