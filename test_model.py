from anonymizer.core import Anonymizer

# Nowy model v2
MODEL_PATH = "models/herbert_ner_v2"
TRAIN_DATA_PATH = "nask_train/original.txt"


def load_all_lines(file_path: str) -> list[str]:
    """Wczytuje wszystkie linie z pliku."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_test():
    print(f"🔄 Ładowanie modelu z: {MODEL_PATH}...")

    # 1. Inicjalizacja z nowym modelem (use_brackets=True jak w Colab)
    try:
        anonymizer = Anonymizer(ner_model_path=MODEL_PATH, use_brackets=True)
        print("✅ Model załadowany pomyślnie!")
        # Włącz NER od razu
        anonymizer._init_ner_layer()
    except Exception as e:
        print(f"❌ Błąd ładowania modelu: {e}")
        print("Czy folder models/herbert_ner_v2 zawiera plik config.json?")
        return

    # 2. Wczytaj WSZYSTKIE linie z danych treningowych
    print(f"\n📂 Wczytywanie wszystkich linii z: {TRAIN_DATA_PATH}...")
    try:
        test_cases = load_all_lines(TRAIN_DATA_PATH)
        print(f"✅ Wczytano {len(test_cases)} linii\n")
    except Exception as e:
        print(f"❌ Błąd wczytywania danych: {e}")
        return

    print("=" * 80)
    print("           TESTY ANONIMIZACJI (RegEx + NER herbert_ner_v2)")
    print("=" * 80)

    for i, text in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"📝 PRZYKŁAD {i}/{len(test_cases)}")
        print(f"{'─' * 80}")
        
        # Wyświetl oryginał (skrócony jeśli za długi)
        display_text = text if len(text) <= 500 else text[:500] + "..."
        print(f"\n🔵 ORYGINAŁ:\n{display_text}")

        # Uruchomienie anonimizacji (Regex najpierw, potem NER)
        result = anonymizer.anonymize(text)
        
        # Wyświetl wynik (skrócony jeśli za długi)
        display_result = result if len(result) <= 500 else result[:500] + "..."
        print(f"\n🟢 ZANONIMIZOWANE:\n{display_result}")

    print(f"\n{'=' * 80}")
    print("                              KONIEC TESTÓW")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    run_test()
