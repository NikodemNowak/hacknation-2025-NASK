import time
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
        total = len(test_cases)
        print(f"✅ Wczytano {total} linii\n")
    except Exception as e:
        print(f"❌ Błąd wczytywania danych: {e}")
        return

    print("=" * 80)
    print("           TESTY ANONIMIZACJI (RegEx + NER herbert_ner_v2)")
    print("=" * 80)

    start_time = time.time()
    results = []

    for i, text in enumerate(test_cases, 1):
        # Anonimizacja
        result = anonymizer.anonymize(text)
        results.append((i, text, result))

        # Progress bar co 50 linii
        if i % 50 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"\r⏳ Postęp: {i}/{total} ({100*i/total:.1f}%) | "
                f"Prędkość: {rate:.1f} linii/s | ETA: {eta:.0f}s",
                end="",
                flush=True,
            )

    print()  # Nowa linia po progress bar

    elapsed_total = time.time() - start_time
    print(
        f"\n✅ Przetworzono {total} linii w {elapsed_total:.1f}s ({total/elapsed_total:.1f} linii/s)\n"
    )

    # Wyświetl wyniki
    for idx, original, anonymized in results:
        print(f"\n{'─' * 80}")
        print(f"📝 PRZYKŁAD {idx}/{total}")
        print(f"{'─' * 80}")

        # Wyświetl oryginał (skrócony jeśli za długi)
        display_text = (
            original if len(original) <= 500 else original[:500] + "..."
        )
        print(f"\n🔵 ORYGINAŁ:\n{display_text}")

        # Wyświetl wynik (skrócony jeśli za długi)
        display_result = (
            anonymized if len(anonymized) <= 500 else anonymized[:500] + "..."
        )
        print(f"\n🟢 ZANONIMIZOWANE:\n{display_result}")

    print(f"\n{'=' * 80}")
    print("                              KONIEC TESTÓW")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    run_test()
